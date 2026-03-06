import math
import numpy as np
import cmath

### ENTITIES ###
class MicrowaveDevice:
    def Z(self, f): raise NotImplementedError
    
    def get_abcd(self, f, mode='series'):
        z = self.Z(f)
        if mode == 'series':
            return np.array([[1, z], [0, 1]])
        else: 
            return np.array([[1, 0], [1/z, 1]])
        
    #SQUID Entity: Acts as a non-linear inductor, simply calculates this inductance based off of input flux (given by mutual Inductance) Componants: Mutual Inductance, Inductance. Should be flexible to include multiple inputs with different couplings (Flux Ramp, TES ...) Params: SQUID params, Inputs: Flux, Outputs: Inductance

class SQUID(MicrowaveDevice):
    """
    Models a SQUID as a flux-tunable inductor. Its impedance is determined
    by the total external magnetic flux applied to its loop.

    Args:
       I_c (float): The critical current of the Josephson junction (in Amps).
       L_loop (float): The geometric loop inductance of the SQUID (in Henrys).
       phi_ext (float, optional): The initial external flux bias (in Webers). Defaults to 0.0.
    """
    def __init__(self, I_c, L_loop, phi_ext=0.0):
        self.I_c = I_c
        self.L_loop = L_loop
        self.phi_ext = phi_ext
        self.PHI_0 = 2.067e-15  # Flux quantum
    
    def set_flux(self,phi):
        self.phi_ext = phi

    def get_L(self):
        cos_term = np.cos(np.pi * self.phi_ext / self.PHI_0)

        if abs(cos_term) < 1e-9:
            return float('inf')

        L_j = self.PHI_0 / (2 * np.pi * self.I_c * abs(cos_term))
        
        return self.L_loop + L_j
        
    def Z(self, f):
        omega = 2 * np.pi * f
        return 1j * omega * self.get_L()

#Resonator Entity: A resonant frequency function in a transmission. Resonant frequency changes based on inductive load. : Componants: Resonance Params, Will be coupled to an inductor componant, which is coupled to a SQUID. Params: Resonator Params, Inputs: Inductance, Outputs: Resonant Params, S21
    
class Resonator(MicrowaveDevice):
    """ Summary:
    _id -> resonator number id
    params:[a, tau, fr, Qr, Qc, phi0] -> resonator params as defined by the fit model:
            - fr: resonant frequency
            - bw: resonator bandwidth
            - Qr: quality factor (to expand)
            -  Theta0
    """
    RESONATOR_COUNT = 0
    
    def __init__(self, params):
        self._id = Resonator.RESONATOR_COUNT
        Resonator.RESONATOR_COUNT+=1
        
        self.params = params
    
    def phase_model(self, f):
        fr, Qr, theta0 = self.params[2], self.params[3], self.params[5]
        return -theta0 + 2 * np.arctan(2 * Qr * (1 - f / fr))
    
    def t21_model(self, f):
        a, tau, fr, Qr, Qc, phi0 = self.params[0],self.params[1],self.params[2],self.params[3],self.params[4],self.params[5]
        delay = np.exp(-2 * np.pi * f * tau * 1.0j)
        resonator_term = (np.exp(1.0j * phi0) * Qr / Qc) / (1 + 2.0j * Qr * (f - fr) / fr)
        return a * delay * (1 - resonator_term)
    
    def Z(self, f):
        if f == 0: return 0
        fr, Qr, theta0 = self.params[2], self.params[3], self.params[5]
        ratio = (f / fr) - (fr / f)
        return 50 * (1 + 1j * Qr * ratio)
    
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
        if f == 0:
            return self.primary_inductor.Z(f)

        omega = 2 * np.pi * f
        z_primary = self.primary_inductor.Z(f)
        z_screening = self.screening_device.Z(f)

        if abs(z_screening) < 1e-12: return complex(0, float('inf'))
        if abs(z_screening) == float('inf'): return z_primary

        z_reflected = (omega * self.M)**2 / z_screening
        return z_primary + z_reflected
        

class Capacitor(MicrowaveDevice):
    def __init__(self, C):
        self.C = C

    def Z(self, f):
        if f == 0:
            return float('inf') 
        omega = 2 * np.pi * f
        return -1j / (omega * self.C)
    
class Terminator(MicrowaveDevice):
    def __init__(self, R):
        self.R = R

    def Z(self, f):
        return complex(self.R,0)


### SYSTEMS ###

#VNA System

class VNA_Simulator:
    def __init__(self, z0=50):
        self.z0 = z0
        self.chain = []

    def add(self, component, mode='series'):
        self.chain.append((component, mode))

    def get_s_matrix(self, f):
        if not np.isscalar(f):
            s_params = np.array([self.get_s_matrix(freq) for freq in f])
            return s_params.transpose(1, 2, 0)

        abcd = np.identity(2, dtype=complex)
        for comp, mode in self.chain:
            abcd = abcd @ comp.get_abcd(f, mode)

        A, B, C, D = abcd.flatten()
        z0 = self.z0
        denom = A + B/z0 + C*z0 + D
        s11 = (A + B/z0 - C*z0 - D) / denom
        s12 = (2 * (A*D - B*C)) / denom
        s21 = 2 / denom
        s22 = (-A + B/z0 - C*z0 + D) / denom
        return np.array([[s11, s12], [s21, s22]])

#Channel System: Can simply add in line a resonator, inductor, SQUID, and input current
class Channel(MicrowaveDevice):
    def __init__(self, name="Channel"):
        self.name = name
        self.components = []

    def add(self, component):
        self.components.append(component)

    def Z(self, f):
        if not self.components:
            return float('inf') 
        return sum(comp.Z(f) for comp in self.components)

#Readout System: Can couple many channels, stack their response, demodulate. Probably the most intensive part.


### HELPERS ###

    #Flux Ramp Generator: At first a simple current function gnerator, but could couple in resonant frequency as tone tracking and such.

    #Pulse Generator

    #TES Coupler
    
    #Resonance Loader
