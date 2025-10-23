import numpy as np
import scipy as sp
import circle_fitting as cf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def savefig(filename):
    plt.savefig("../../figures/"+filename+".pdf")

def fit_phase(f_region, z_t, zc, r, plot_mode='none', filename=None):
    phase = np.angle(z_t)

    def phase_func(f, fr, Qr, theta0):
        return -theta0 + 2 * np.arctan(2 * Qr * (1 - f / fr))

    fr_guess = f_region[np.argmin(np.abs(phase))]
    Qr_guess = 50000.0
    theta0_guess = 0.0
    p0 = [fr_guess, Qr_guess, theta0_guess]

    try:
        popt, pcov = curve_fit(phase_func, f_region, phase, p0=p0)
    except RuntimeError:
        return [np.nan] * 5, [np.nan] * 5

    fr_fit, Qr_fit, theta0_fit = popt
    perr = np.sqrt(np.diag(pcov))
    fr_err, Qr_err, theta0_err = perr

    Qc_fit = Qr_fit * (np.abs(zc) + r) / (2 * r)
    phi0_fit = theta0_fit - np.angle(zc)
    Qc_err = np.nan
    phi0_err = np.nan

    if plot_mode == 'all':
        print(f"Initial Guesses (p0):")
        print(f" fr = {p0[0]:.4e} Hz")
        print(f" Qr = {p0[1]:.1f}")
        print(f" theta0 = {p0[2]:.4f} rad\n")
        print("Fit Results:")
        print(f" fr = {fr_fit:.4e} ± {fr_err:.1e} Hz")
        print(f" Qr = {Qr_fit:.1f} ± {Qr_err:.1f}")
        print(f" theta0 = {theta0_fit:.4f} ± {theta0_err:.4f} rad\n")
        print(f"Derived Parameters:")
        print(f" Qc = {Qc_fit:.1f}")
        print(f" phi0 = {phi0_fit:.4f} rad")

        plt.figure(figsize=(8, 6))
        plt.plot(f_region, phase, 'o', label='Data', markersize=4)
        plt.plot(f_region, phase_func(f_region, *popt), 'r-', label='Fit')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (radians)")
        plt.title("Phase Angle Fit")
        plt.legend()
        plt.grid(True)
        if filename is not None:
            savefig(filename + "_phase_fit")
        plt.show()

    params = [fr_fit, Qr_fit, theta0_fit, Qc_fit, phi0_fit]
    params_unc = [fr_err, Qr_err, theta0_err, Qc_err, phi0_err]
    return params, params_unc

def initial_fit(f_region, s21_region, tau, plot_mode='none', filename=None):
    s21_real = np.real(s21_region)
    s21_imag = np.imag(s21_region)
    arg = 2 * np.pi * tau * f_region
    s21_p = s21_region * (1.0 * np.cos(arg) + 1.0j * np.sin(arg))
    s21_p_real = np.real(s21_p)
    s21_p_imag = np.imag(s21_p)
    xc, yc, r = cf.fit_circle(s21_p_real, s21_p_imag)
    zc = 1.0 * xc + 1.0j * yc
    alpha = np.angle(zc)
    z_t = (zc - s21_p) * np.exp(-1j * alpha)
    
    xc_t, yc_t, r_t = cf.fit_circle(np.real(z_t), np.imag(z_t))
    zc_t = 1.0 * xc_t + 1.0j * yc_t

    if plot_mode == 'all':
        plt.figure(figsize=(8, 8))
        cf.plot_circle(xc, yc, r, 100)
        cf.plot_circle(xc_t, yc_t, r, 100)
        plt.plot(np.real(z_t), np.imag(z_t), label="Transformed Data")
        plt.plot(s21_p_real, s21_p_imag, '.', label="Delay-Corrected Data")
        plt.xlabel("Real(S21')")
        plt.ylabel("Imag(S21')")
        plt.title("Initial Circle Fit")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        if filename is not None:
            savefig(filename + "_init_fit")
        plt.show()
    return zc, z_t, r

def refined_fit(f_region, s21_region, tau, plot_mode='none', filename=None):
    def t21_func(f, a, tau, fr, Qr, Qc, phi0):
        delay = np.exp(-2 * np.pi * f * tau * 1.0j)
        resonator_term = (np.exp(1.0j * phi0) * Qr / Qc) / (1 + 2.0j * Qr * (f - fr) / fr)
        return a * delay * (1 - resonator_term)

    def t21_wrapper_func(f_stacked, a, tau, fr, Qr, Qc, phi0):
        N = len(f_stacked) // 2
        f_unique = f_stacked[:N]
        t21_complex = t21_func(f_unique, a, tau, fr, Qr, Qc, phi0)
        return np.concatenate((np.real(t21_complex), np.imag(t21_complex)))

    s21_stacked_real = np.concatenate((np.real(s21_region), np.imag(s21_region)))
    f_stacked = np.concatenate((f_region, f_region))
    zc, z_t, r = initial_fit(f_region, s21_region, tau, plot_mode=plot_mode, filename=filename)
    params, _ = fit_phase(f_region, z_t, zc, r, plot_mode=plot_mode, filename=filename)
    fr_fit_phase, Qr_fit_phase, theta0_fit_phase, Qc_fit_phase, phi0_fit_phase = params
    
    try:
        if np.isnan(fr_fit_phase):
            raise RuntimeError("Phase fitting failed, cannot proceed with refined fit.")
            
        num_edge_points = max(1, int(len(s21_region) * 0.1))
        edge_magnitudes = np.concatenate([
            np.abs(s21_region[:num_edge_points]),
            np.abs(s21_region[-num_edge_points:])
        ])
        a_guess = np.mean(edge_magnitudes)
        params_guess = [a_guess, tau, fr_fit_phase, Qr_fit_phase, Qc_fit_phase, phi0_fit_phase]

        popt, pcov = curve_fit(t21_wrapper_func, f_stacked, s21_stacked_real,
                               p0=params_guess,
                               maxfev=5000)
                               
        a_fine, tau_fine, fr_fine, Qr_fine, Qc_fine, phi0_fine = popt
        
        
        perr = np.sqrt(np.diag(pcov))
        dfr_fine = perr[2]
        t21_fit = t21_func(f_region, *popt)

        if plot_mode == 'all':
            print(f"Fit successful! Resonance Frequency: {fr_fine*1e-9:.6f} +/- {dfr_fine*1e-9:.6f} GHz")
            plt.figure(figsize=(8, 8))
            plt.plot(np.real(s21_region), np.imag(s21_region), '.', label='Data')
            plt.plot(np.real(t21_fit), np.imag(t21_fit), 'r-', label='Fit to Gao')
            plt.title('Refined Resonance Circle Fit')
            plt.xlabel('Real(S21)')
            plt.ylabel('Imag(S21)')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            if filename is not None:
                savefig(filename + "_refined_circle_fit")
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.scatter(f_region, np.abs(s21_region), label='Data')
            plt.plot(f_region, np.abs(t21_fit), color='r', label='Fit')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude |S21|")
            plt.title("Magnitude Fit")
            plt.legend()
            plt.grid(True)
            if filename is not None:
                savefig(filename + "_refined_mag_fit")
            plt.show()

        elif plot_mode == 'condensed':
            print(f"Fit successful! Resonance Frequency: {fr_fine*1e-9:.6f} +/- {dfr_fine*1e-9:.6f} GHz")
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
            ax1.plot(np.real(s21_region), np.imag(s21_region), '.', label='Data')
            ax1.plot(np.real(t21_fit), np.imag(t21_fit), 'r-', label='Fit')
            ax1.set_title('Resonance Circle Fit')
            ax1.set_xlabel('Real(S21)')
            ax1.set_ylabel('Imag(S21)')
            ax1.legend()
            ax1.axis('equal')
            ax1.grid(True)
            ax2.plot(f_region * 1e-9, np.abs(s21_region), '.', label='Data')
            ax2.plot(f_region * 1e-9, np.abs(t21_fit), color='r', label='Fit')
            ax2.set_xlabel("Frequency (GHz)")
            ax2.set_ylabel("Magnitude |S21|")
            ax2.set_title("Magnitude Fit")
            ax2.legend()
            ax2.grid(True)

            def phase_func_plot(f, fr, Qr, theta0):
                return -theta0 + 2 * np.arctan(2 * Qr * (1 - f / fr))

            phase_data = np.angle(z_t)
            phase_fit_params = [fr_fit_phase, Qr_fit_phase, theta0_fit_phase]
            ax3.plot(f_region * 1e-9, phase_data, 'o', markersize=4, label='Data')
            ax3.plot(f_region * 1e-9, phase_func_plot(f_region, *phase_fit_params), 'r-', label='Fit')
            ax3.set_xlabel("Frequency (GHz)")
            ax3.set_ylabel("Phase (radians)")
            ax3.set_title("Initial Phase Fit")
            ax3.legend()
            ax3.grid(True)
            plt.tight_layout()
            if filename is not None:
                savefig(filename + "_condensed_fit")
            plt.show()

        chi2 = residues(f_region, s21_region, t21_fit, plot_mode=plot_mode, filename=filename)
        
        return a_fine, tau_fine, fr_fine, Qr_fine, Qc_fine, phi0_fine, chi2

    except RuntimeError as e:
        print(f"WARNING: Fit failed for this region. Skipping. (Error: {e})")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def residues(f_region, s21_region, t21_fit, plot_mode='none', filename=None):
    res = s21_region - t21_fit
    n_complex_points = len(res)
    
    m = int(len(f_region)* 0.02)
    if m >= len(s21_region):
        m = len(s21_region) - 1

    sigmaz2 = np.sum(np.abs(np.diff(s21_region[:m + 1]))**2) / (2 * m)
    sigma_real_sq = sigmaz2 / 2.0
    k = 6 
    dof = 2 * n_complex_points - k
    chi2_val = np.sum(np.abs(res)**2) / sigma_real_sq
    reduced_chi2 = chi2_val / dof
    
    if plot_mode in ['all']:
        print(f"Number of complex points (n): {n_complex_points}")
        print(f"Degrees of Freedom (2n - k): {dof}")
        print(f"Estimated single-quadrature noise variance (sigma_0^2): {sigma_real_sq:.3e}")
        print(f"Reduced Chi-Squared (chi_nu^2): {reduced_chi2:.4f}")

        # Your plotting code from here is great for visualization...
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(f_region * 1e-9, np.real(res), 'o', alpha=0.7, label='Real Part')
        ax.plot(f_region * 1e-9, np.imag(res), 'o', alpha=0.7, label='Imaginary Part')
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
        
        noise_std_dev_est = np.sqrt(sigma_real_sq)
        ax.axhspan(-noise_std_dev_est, noise_std_dev_est, color='gray', alpha=0.2, label=r'1$\sigma$ Noise Level')
        
        ax.set_title(fr'Fit Residuals ($\chi^2_\nu$ = {reduced_chi2:.2f})', fontsize=18)
        ax.set_xlabel('Frequency (GHz)', fontsize=14)
        ax.set_ylabel('Residual (Data - Fit)', fontsize=14)
        ax.legend(loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        if filename is not None:
            savefig(filename + "_residuals")
        plt.show()

    return reduced_chi2