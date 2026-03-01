import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Physical Constants ---
PHI_0 = 2.067e-15        # Flux quantum
M_IN = 250e-12           # TES-to-SQUID Mutual Inductance (250 pH)

# --- Resonator & SQUID Parameters ---
f_center = 5.0e9         # 5 GHz probe frequency
df_max = 100e3           # 100 kHz peak-to-peak SQUID swing
Q_tot = 50000            # Total Quality Factor
Q_c = 60000              # Coupling Quality Factor

# --- DAQ & Ramp Parameters ---
fs = 2.4e6               # 2.4 MHz sample rate
t_duration = 0.002       # 2 ms simulation time
f_ramp = 400e3          # 40 kHz flux ramp frequency
n_phi0 = 4              # Number of Phi_0 to sweep per ramp cycle

I_C = 1.0e-6        # Critical current (1 uA)
L_LOOP = 50e-12     # SQUID loop inductance (50 pH)
M_SQ = 20e-12       # Mutual inductance to resonator (20 pH)
L_RES = 2e-9        # Resonator inductance (2 nH)
C_RES = 0.5e-12     # Resonator capacitance (0.5 pF)

def flux_SQ(I_in):
    M_in = 250e-12
    return M_in * I_in

def freq_res(phi, f0, f_max):
    phi0 = 2.067e-15
    phase = 2*np.pi*phi/phi0
    return f0 + f_max*np.cos(phase)

def calc_freq_res(I_in):
    f0 = 6e9
    f_max = 100e3
    
    phi = flux_SQ(I_in)
    fr = freq_res(phi, f0, f_max)
    
    return fr

def L_SQ(phi, I_c, L_loop):
    Lj = PHI_0 / (2 * np.pi * I_c * np.cos(np.pi * phi / PHI_0))
    return L_loop + Lj

def L_eff(phi, I_c, L_loop, L_res, M_sq):
    return L_res - (M_sq**2 / L_SQ(phi, I_c, L_loop))

def fr(phi, I_c, L_loop, L_res, M_sq, C_res):
    Leff = L_eff(phi, I_c, L_loop, L_res, M_sq)
    return 1 / (2 * np.pi * np.sqrt(Leff * C_res))

def flux_SQ(I_in):
    return M_IN * I_in

def generate_flux_ramp(t, f_ramp, n_phi0):
    T_ramp = 1 / f_ramp
    phi_amplitude = n_phi0 * PHI_0
    phi_ramp = (phi_amplitude / T_ramp) * (t % T_ramp)
    
    return phi_ramp


def get_s21(f_probe, f_res, Q_tot, Q_c):
    """
    Calculates complex S21 transmission for a given probe frequency 
    and instantaneous resonance frequency.
    """
    # Fractional frequency shift
    dx = (f_probe - f_res) / f_res
    
    # Standard notch resonator formula
    s21 = 1 - (Q_tot / Q_c) / (1 + 2j * Q_tot * dx)
    return s21