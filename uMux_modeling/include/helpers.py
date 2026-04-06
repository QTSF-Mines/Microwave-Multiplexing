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

def find_all_peaks(f, s21,dist = 1.0e6 , showplot = False, filename = None):
    s21_flipped = s21.max() - np.abs(s21)

    # Temporary debug plot
    # plt.figure()
    # plt.plot(f, s21_flipped)
    # plt.title("Is this 'flipped' signal showing clear peaks?")
    # plt.show()

    excluded_frequencies = []

    peaks_full, _ = find_peaks(s21_flipped, prominence=0.05, distance=10)

    collided_peak_indices = set()

    for i in range(len(peaks_full)-1):
        peak1_idx = peaks_full[i]
        peak2_idx = peaks_full[i+1]

        if np.abs(f[peak2_idx] - f[peak1_idx]) < dist:
                collided_peak_indices.add(peak1_idx)
                collided_peak_indices.add(peak2_idx)
            
        for j in range(len(excluded_frequencies)):
            if np.abs(excluded_frequencies[j] - f[peak1_idx]) < dist:
                collided_peak_indices.add(peak1_idx)
        

    final_peaks_list = [p for p in peaks_full if p not in collided_peak_indices]
    peaks = np.array(final_peaks_list, dtype=int)
    
    if(showplot):
        s21_db = 20*np.log10(np.abs(s21))
        plt.vlines(x=f[peaks], ymin=s21_db.min(), ymax=s21_db.max(), 
                color='red', linestyle='--', alpha=0.5, label='Peak Locations')

        plt.plot(f, 20*np.log10(np.abs(s21)))
        plt.ylabel("S21 (dB)")
        plt.xlabel("Freq (Hz)")
        
        if(filename != None):
            savefig(filename+"_peak_locations")
            
        plt.show()
        
    return peaks