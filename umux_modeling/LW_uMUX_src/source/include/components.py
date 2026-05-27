import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import root_scalar

### ENTITIES ###
class MicrowaveDevice:
    def __init__(self, name = None):
        self.name = name
    def Z(self, f): raise NotImplementedError
    
    def get_abcd(self, f, mode='series'):
        z = self.Z(f)
        if np.isscalar(z):
            if mode == 'series':
                return np.array([[1, z], [0, 1]], dtype=complex)
            else:  # shunt
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.array([[1, 0], [1/z, 1]], dtype=complex)
        else:
            N = len(z)
            abcd = np.zeros((N, 2, 2), dtype=complex)
            abcd[:, 0, 0] = 1
            abcd[:, 1, 1] = 1
            if mode == 'series':
                abcd[:, 0, 1] = z
            else:  # shunt
                with np.errstate(divide='ignore', invalid='ignore'):
                    abcd[:, 1, 0] = 1 / z
            return abcd

class SQUID(MicrowaveDevice):
    """
    Models an RF-SQUID as a flux-tunable inductor. Incorporates
    self-screening physics via the Kepler equation.
    """
    def __init__(self, L_j, L_s, lamb=None,  phi_ext=0.0):
        self.name = "SQUID"
        self.L_s = L_s
        if lamb is not None:
            self.lamb = lamb
            self.L_j = L_s / lamb
        else:
            self.L_j = L_j
            self.lamb = L_s / L_j
            

        self.phi_ext = phi_ext
        self.phi_rf_amp = 0.0
        self.PHI_0 = 2.068e-15  
        self.flux_lines = {}
        
    def add_flux_line(self, name, M, initial_current=0.0):
        line = FluxLine(name, M, initial_current)
        self.flux_lines[name] = line
        return line
        
    def set_current(self, name, new_I):
        if name in self.flux_lines:
            line = self.flux_lines[name]
            line.I = new_I  # This now automatically updates line.phi via the property setter
        else:
            raise KeyError(f"Flux line '{name}' does not exist on this SQUID.")
        
    def get_total_flux(self):
        total_flux = self.phi_ext
        for line in self.flux_lines.values():
            total_flux += line.get_flux()
        return total_flux

    def phi_of_phie(self, phie):
        if self.lamb < 1e-6: return phie
        phi = np.copy(phie) 
        
        for _ in range(15):  
            step = (phi + self.lamb * np.sin(phi) - phie) / (1.0 + self.lamb * np.cos(phi))
            if np.max(np.abs(step)) < 1e-9: break
            phi -= step
            
        return phi

    def Z(self, f):
        """
        Calculates the SQUID impedance at a given frequency f.

        If RF probe power is specified (via phi_rf_amp > 0), this method
        solves the full nonlinear time-domain SQUID equation to find the
        complex impedance Z = V/I at the fundamental frequency. This is
        more accurate than linear approximations but computationally slower.

        If no RF power is applied, it returns the small-signal inductance.
        """
        phie_dc = 2.0 * np.pi * self.get_total_flux() / self.PHI_0
        is_array = isinstance(phie_dc, np.ndarray) and phie_dc.ndim > 0

        # Handle DC case or when no RF power is applied (small-signal limit)
        if f == 0 or self.phi_rf_amp == 0:
            phi_dc = self.phi_of_phie(phie_dc)
            L_squid_small_signal = self.L_s / (1.0 + self.lamb * np.cos(phi_dc))
            return 1j * 2 * np.pi * f * L_squid_small_signal

        # --- Time-domain numerical solution for impedance ---
        omega = 2 * np.pi * f
        num_points = 128  # Points per RF cycle for simulation
        t = np.linspace(0, 1/f, num_points, endpoint=False)

        # Create the time-varying external flux drive
        phi_rf_radians = 2.0 * np.pi * self.phi_rf_amp / self.PHI_0
        phie_t = phie_dc + phi_rf_radians * np.sin(omega * t)
        if is_array:
            phie_t = phie_dc[:, np.newaxis] + phi_rf_radians * np.sin(omega * t)

        # Solve for total flux vs. time by solving the transcendental equation at each time step
        phi_total_t = self.phi_of_phie(phie_t)

        # Calculate time-domain voltage and screening current
        axis = 1 if is_array else 0
        V_t = (self.PHI_0 / (2 * np.pi)) * np.gradient(phi_total_t, t, axis=axis, edge_order=2)
        I_c = self.PHI_0 / (2 * np.pi * self.L_j)
        I_scr_t = I_c * np.sin(phi_total_t)

        # Use rFFT to get the fundamental components of V and I
        V_fft = np.fft.rfft(V_t, axis=axis)
        I_fft = np.fft.rfft(I_scr_t, axis=axis)

        V_fundamental = V_fft.take(1, axis=axis)
        I_fundamental = I_fft.take(1, axis=axis)

        # Calculate impedance Z = V/I. FFT result needs conjugation for standard phasor def.
        z_squid = np.full_like(I_fundamental, complex(0, float('inf')), dtype=complex)
        mask = np.abs(I_fundamental) >= 1e-12
        z_squid[mask] = np.conj(V_fundamental[mask] / I_fundamental[mask])

        return z_squid.item() if not is_array else z_squid

    def get_L(self):
        total_flux = self.get_total_flux()
        phie = 2.0 * np.pi * total_flux / self.PHI_0
        phi = self.phi_of_phie(phie)
        cos_term = math.cos(phi)
        L_j_tuned = float('inf') if abs(cos_term) < 1e-12 else self.L_j / cos_term
        return self.L_s + L_j_tuned
    
    def add_rf_flux_func(self, func):
        self.phi_rf_func = func

class FluxLine:
    def __init__(self, name, M, initial_current=0.0):
        self.name = name
        self.M = M
        # Use "private" attributes for internal state
        self._I = 0.0
        self._phi = 0.0
        # Set initial state via the property setter to ensure consistency
        self.I = initial_current
        
    @property
    def I(self):
        """The current applied to the flux line."""
        return self._I

    @I.setter
    def I(self, new_I):
        self._I = np.asanyarray(new_I)
        self._phi = self.M * self._I

    @property
    def phi(self):
        """The magnetic flux generated by the current in the line."""
        return self._phi

    @phi.setter
    def phi(self, new_phi):
        self._phi = np.asanyarray(new_phi)
        if self.M != 0:
            self._I = self._phi / self.M
        else:
            self._I = np.zeros_like(self._phi)

    def get_flux(self):
        return self.phi
    
    def set_flux(self, phi):
        self.phi = phi

class Resonator(MicrowaveDevice):
    def __init__(self, length, load, v_p=1.15e8, Qi=100000, Qr = None, Qc = None,  Z0=50):
        self.name = "Resonator"
        self.length = length
        self.load = load
        self.v_p = v_p
        self.Qi = Qi
        if Qr is not None:
            self.Qr = Qr
        if Qc is not None:
            self.Qc = Qc
        self.Z0 = Z0
        
    def Z(self, f):
        if np.isscalar(f) and f == 0: return float('inf')
        beta = 2 * np.pi * f / self.v_p
        alpha = beta / (2 * self.Qi)
        gamma = alpha + 1j * beta
        
        zl_val = self.load.Z(f) 
        
        num = zl_val + self.Z0 * np.tanh(gamma * self.length)
        den = self.Z0 + zl_val * np.tanh(gamma * self.length)
        
        return self.Z0 * (num / den)
    

    
class Inductor(MicrowaveDevice):
    def __init__(self, L):
        self.name = "Inductor"
        self.L = L
        
    def Z(self, f):
        omega = 2 * np.pi * f
        return 1j * omega * self.L

class ScreenedInductor(MicrowaveDevice):
    def __init__(self, primary_inductor, screening_device, M):
        self.name = "ScreenedInductor"
        if not isinstance(primary_inductor, Inductor):
            raise TypeError("The primary_inductor must be an instance of the Inductor class.")
        self.primary_inductor = primary_inductor
        self.screening_device = screening_device
        self.M = M

    def Z(self, f):
        if np.isscalar(f) and f == 0:
            return self.primary_inductor.Z(f)

        omega = 2 * np.pi * f
        z_primary = self.primary_inductor.Z(f)
        z_screening = self.screening_device.Z(f)

        if np.isscalar(z_screening):
            if abs(z_screening) < 1e-12:
                return z_primary + complex(0, float('inf'))
            if abs(z_screening) == float('inf'):
                return z_primary
            z_reflected = (omega * self.M)**2 / z_screening
        else:
            z_reflected = np.zeros_like(z_screening, dtype=complex)
            
            valid_mask = (np.abs(z_screening) >= 1e-12) & (~np.isinf(z_screening))
            z_reflected[valid_mask] = (omega * self.M)**2 / z_screening[valid_mask]
            
            zero_mask = np.abs(z_screening) < 1e-12
            z_reflected[zero_mask] = complex(0, float('inf'))

        return z_primary + z_reflected

class Capacitor(MicrowaveDevice):
    
    def __init__(self, C):
        self.C = C
        self.name = "Capacitor"

    def Z(self, f):
        if np.isscalar(f) and f == 0: return float('inf') 
        omega = 2 * np.pi * f
        return -1j / (omega * self.C)
    
class Terminator(MicrowaveDevice):
    def __init__(self, R):
        self.name = "Term"
        self.R = R

    def Z(self, f):
        return complex(self.R,0)