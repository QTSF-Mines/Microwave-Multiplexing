import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks
from scipy.optimize import root_scalar

def extract_physical_params(fit_params, load_device, v_p=1.15e8, Z0=50, Cc_in=None):
    fr, Qr, Qc = fit_params[2], fit_params[3], fit_params[4]
    omega = 2 * np.pi * fr

    Qi = 1 / (1/Qr - 1/Qc) if (Qr > 0 and Qc > 0 and (1/Qr - 1/Qc) > 1e-12) else 1e12 
    
    if Cc_in is not None:
        Cc = Cc_in
    else:
        Cc = (1 / (2 * omega * Z0)) * np.sqrt(np.pi / Qc)
    
    z_load = load_device.Z(fr)
    L_load = np.imag(z_load) / omega
    
    Xc = 1 / (omega * Cc)
    XL = omega * L_load
    
    tan_bl = Z0 * (Xc - XL) / (Z0**2 + Xc * XL)
    
    bl = np.arctan(tan_bl)
    if bl < 0: bl += np.pi 
        
    length = bl / (omega / v_p)
    return length, Cc, Qi

def generate_flux_ramp(t, f_ramp, n_phi0):
    T_ramp = 1 / f_ramp
    PHI_0 = 2.067e-15
    phi_amplitude = n_phi0 * PHI_0
    phi_ramp = (phi_amplitude / T_ramp) * (t % T_ramp)
    
    return phi_ramp

def savefig(filename):
    plt.savefig(filename+".pdf")


def find_all_peaks(f, s21, dist=1.0e6, showplot=False, filename=None, excluded_frequencies=None):
    # Calculate magnitude once to save memory and processing time later
    mag_s21 = np.abs(s21)
    
    # 1. Invert the magnitude array directly instead of subtracting the max.
    # Finding peaks on -mag_s21 perfectly locates the S21 transmission dips.
    peaks_full, _ = find_peaks(-mag_s21, prominence=0.05, distance=10)
    
    # Fast exit if no peaks are found
    if len(peaks_full) == 0:
        return np.array([], dtype=int)
        
    f_peaks = f[peaks_full]
    
    # Create a boolean mask to track which peaks to drop (False = keep, True = drop)
    drop_mask = np.zeros(len(peaks_full), dtype=bool)
    
    # 2. Vectorized Adjacency Check (Replaces the first for-loop)
    # np.diff calculates the distance between adjacent peaks instantly.
    diffs = np.diff(f_peaks)
    too_close = diffs < dist
    
    # If a pair is too close, flag both the left and right peaks for deletion
    drop_mask[:-1] |= too_close 
    drop_mask[1:]  |= too_close  
    
    # 3. Vectorized Exclusion Check (Replaces the nested for-loop)
    if excluded_frequencies is not None and len(excluded_frequencies) > 0:
        excl_arr = np.array(excluded_frequencies)
        
        # Array broadcasting: creates a 2D matrix of all peak-to-exclusion distances at once
        excl_dists = np.abs(f_peaks[:, np.newaxis] - excl_arr)
        
        # If any peak is within `dist` of ANY excluded frequency, flag it
        drop_mask |= np.any(excl_dists < dist, axis=1)
        
    # 4. Filter the peaks array in one fell swoop
    final_peaks = peaks_full[~drop_mask]
    
    if showplot:
        # Re-use the previously calculated magnitude for the dB conversion
        s21_db = 20 * np.log10(mag_s21)
        
        plt.figure()
        plt.vlines(x=f[final_peaks], ymin=s21_db.min(), ymax=s21_db.max(), 
                   color='red', linestyle='--', alpha=0.5, label='Peak Locations')

        plt.plot(f, s21_db)
        plt.ylabel("S21 (dB)")
        plt.xlabel("Freq (Hz)")
        plt.legend()
        
        if filename:
            plt.savefig(filename + "_peak_locations")
            
        plt.show()
        
    return final_peaks



import scipy.optimize as op

plt.ion()

# NOTE: lambda is a reserved word in python, so we use lamb instead.

Phi0 = 2.068e-15    # Magnetic flux quantum
Z1 = 50.0           # 50 Ohm CPW resonator
Ls = 22.4e-12       # SQUID self-inductance (from FastHenry simulation)


def phi_of_phie(phie, lamb):
    """ Solves the self-consistent magnetic flux in the SQUID.

    The magnetic flux in an unshunted rf-SQUID obeys:

    Phi = Phi_external - lambda*sin(2*pi*Phi/Phi0)

    where the second term describes flux applied to the SQUID by its own circulating current.

    This is Kepler's Equation (https://en.wikipedia.org/wiki/Kepler%27s_equation), which is a
    transcendental equation without a closed-form inverse. We therefore solve it numerically,
    assuming lambda < 1. If lambda > 1 the inversion is multi-valued and is not solved with
    this code.

    Args:
        phie      : Externally applied flux in radians (phie = 2*pi*Phi_e/Phi0).
        lamb      : Lambda parameter defined as lambda = 2*pi*Ic*Ls/Phi0. (also known as beta_L)
    
    Returns:
        phi       : Total flux in the SQUID in radians (phi = 2*pi*Phi/Phi0).
    """
    phiguess = phie
    phi = op.fsolve(lambda phi : phi + lamb*np.sin(phi) - phie, phiguess)   # Numerically solve Kepler Eq.

    return phi

def f0_of_I(I,I0,Min,Mc,fb,lamb):
    """ Solves for the resonance frequency as a function of input parameters

    This model comes from Ben Mates' thesis (Eq. 2.16 and 2.50):

    f0(phi) - fb = (4 f0^2 / Z1) * (Mc^2 / Ls) * (lambda cos(phi)) / (1 + lambda cos(phi))

    Args:
        I         : Current applied to the SQUID input (or flux-ramp) coil (A).
        I0        : Offset flux in units of input current (A).
        Min       : Mutual inductance of input (or flux-ramp) coil into the SQUID (H).
        Mc        : Mutual inductance of the resonator into the SQUID (H).
        fb        : "Base" frequency of the resonator, not loaded by the rf-SQUID (Hz).
        lamb      : Lambda parameter defined as lambda = 2*pi*Ic*Ls/Phi0. (also known as beta_L)
    
    Returns:
        f0        : Resonance frequency loaded by the rf-SQUID (Hz).
    """
    phie = 2*np.pi*(I+I0)*Min/Phi0
    phi = phi_of_phie(phie,lamb)
    f0 = fb + (4*(fb**2)/Z1) * (Mc**2)/Ls * lamb*np.cos(phi)/(1 + lamb*np.cos(phi))

    return f0

