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