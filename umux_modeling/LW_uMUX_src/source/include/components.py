import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks
from scipy.optimize import root_scalar

### ENTITIES ###
class MicrowaveDevice:
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
    def __init__(self, L_j, L_s, lamb=None, phi_ext=0.0):
        self.L_s = L_s
        if lamb is not None:
            self.lamb = lamb
            self.L_j = L_s / lamb
        else:
            self.L_j = L_j
            self.lamb = L_s / L_j
            
        self.phi_ext = phi_ext
        self.PHI_0 = 2.068e-15  
        self.flux_lines = {}
        
    def add_flux_line(self, name, M, initial_current=0.0):
        line = FluxLine(name, M, initial_current)
        self.flux_lines[name] = line
        return line
        
    def set_current(self, name, new_I):
        if name in self.flux_lines:
            self.flux_lines[name].I = new_I
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
        phie = 2.0 * np.pi * self.get_total_flux() / self.PHI_0
        phi = self.phi_of_phie(phie)
        L_squid = self.L_s / (1.0 + self.lamb * np.cos(phi))
        return 1j * 2 * np.pi * f * L_squid
    
    def get_L(self):
        total_flux = self.get_total_flux()
        phie = 2.0 * np.pi * total_flux / self.PHI_0
        phi = self.phi_of_phie(phie)
        cos_term = math.cos(phi)
        L_j_tuned = float('inf') if abs(cos_term) < 1e-12 else self.L_j / cos_term
        return self.L_s + L_j_tuned

class FluxLine:
    def __init__(self, name, M, initial_current=0.0):
        self.name = name
        self.M = M
        self.I = initial_current
        self.phi = M*initial_current
        
    def get_flux(self):
        #print("Flux on",self.name,"=",self.M*self.I)
        return float(self.M)*self.I
    
    def set_flux(self, phi):
        self.phi = phi
        self.I = phi/self.M
    
class Resonator(MicrowaveDevice):
    def __init__(self, length, load, v_p=1.15e8, Qi=100000, Z0=50):
        self.length = length
        self.load = load
        self.v_p = v_p
        self.Qi = Qi
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
        self.L = L
        
    def Z(self, f):
        omega = 2 * np.pi * f
        return 1j * omega * self.L

class ScreenedInductor(MicrowaveDevice):
    def __init__(self, primary_inductor, screening_device, M):
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

    def Z(self, f):
        if np.isscalar(f) and f == 0: return float('inf') 
        omega = 2 * np.pi * f
        return -1j / (omega * self.C)
    
class Terminator(MicrowaveDevice):
    def __init__(self, R):
        self.R = R

    def Z(self, f):
        return complex(self.R,0)