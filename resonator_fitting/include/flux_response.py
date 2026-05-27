import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

plt.ion()

def savefig(filename):
    plt.savefig(filename + ".pdf")

def fit_all_flux_responses(ibias, freqs_2d, lf, showplot=False, filename=None, condensed_plots_ids=None):
    """
    Iterates over a 2D array of flux responses, fits them, and compiles the parameters.
    
    Args:
        ibias (1D array): Applied bias current array (A).
        freqs_2d (2D array): Array of shape (N_resonators, len(ibias)) containing frequency shifts (Hz).
        lf (module): Your library containing 'fit_lambda' and 'f0_of_I'.
        showplot (bool): Whether to generate aggregate summary plots at the end.
        filename (str): Base filename for saving plots.
        condensed_plots_ids (list): List of indices (e.g., [0, 5, 12]) to plot individually.
        
    Returns:
        tuple of arrays: (I0fits, Minfits, Mcfits, fbfits, lambfits)
    """
    I0fits, Minfits, Mcfits, fbfits, lambfits = [], [], [], [], []
    
    for i in range(len(freqs_2d)):
        plot_mode = 'none'
        individual_filename = None
        
        if condensed_plots_ids is not None and i in condensed_plots_ids:
            plot_mode = 'condensed'
            if filename is not None:
                individual_filename = f"{filename}_res{i}"
                
        # Call the refined fit function for this specific resonator
        I0, Min, Mc, fb, lamb = refined_flux_fit(
            ibias, freqs_2d[i], lf, 
            plot_mode=plot_mode, 
            filename=individual_filename, 
            showdata=True
        )
        
        I0fits.append(I0)
        Minfits.append(Min)
        Mcfits.append(Mc)
        fbfits.append(fb)
        lambfits.append(lamb)
        
    if showplot:
        make_flux_result_plots(I0fits, Minfits, Mcfits, fbfits, lambfits, filename)
        
    return np.array(I0fits), np.array(Minfits), np.array(Mcfits), np.array(fbfits), np.array(lambfits)


def refined_flux_fit(ibias, freqs, lf, plot_mode='none', filename=None, showdata=False):
    """ Fits a single uMux flux response curve and handles individual plotting/residuals. """
    try:
        I0fit, Minfit, Mcfit, fbfit, lambfit = lf.fit_lambda(ibias, freqs)
        
        if plot_mode == 'none':
            return I0fit, Minfit, Mcfit, fbfit, lambfit
            
        f_fit_points = lf.f0_of_I(ibias, I0fit, Minfit, Mcfit, fbfit, lambfit)
        residuals = freqs - f_fit_points
        
        i_smooth = np.linspace(np.min(ibias), np.max(ibias), 1000)
        f_smooth = lf.f0_of_I(i_smooth, I0fit, Minfit, Mcfit, fbfit, lambfit)
        
        if plot_mode in ['all', 'condensed']:
            textstr = ""
            if showdata:
                textstr = '\n'.join((
                    r'Fit Parameters:',
                    f'$\\lambda = {lambfit:.3f}$',
                    f'$M_{{in}} = {Minfit*1e12:.1f}$ pH',
                    f'$M_c = {Mcfit*1e12:.1f}$ pH',
                    f'$I_0 = {I0fit*1e6:.1f}$ $\\mu$A',
                    f'$f_b = {fbfit*1e-6:.3f}$ MHz'
                ))
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='lightgray')

        if plot_mode == 'condensed' or plot_mode == 'all':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Main Fit Plot
            ax1.plot(ibias * 1e3, freqs * 1e-6, 'o', markersize=5, label='Data', color='C0', alpha=0.8)
            ax1.plot(i_smooth * 1e3, f_smooth * 1e-6, '-', linewidth=2.5, label='Fit', color='C1')
            ax1.set_xlabel("Input Current (mA)", fontsize=12)
            ax1.set_ylabel("Frequency Shift (MHz)", fontsize=12)
            ax1.set_title("Flux Response Fit", fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.6)
            if showdata:
                ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
            ax1.legend(loc='lower right')

            # Residuals Plot
            ax2.plot(ibias * 1e3, residuals * 1e-6, 'o-', markersize=4, color='C3')
            ax2.axhline(0, color='black', linestyle='--', linewidth=1)
            ax2.set_xlabel("Input Current (mA)", fontsize=12)
            ax2.set_ylabel("Residuals (MHz)", fontsize=12)
            ax2.set_title("Residuals", fontsize=14)
            ax2.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            if filename is not None:
                savefig(filename + "_condensed")
            plt.show()

        return I0fit, Minfit, Mcfit, fbfit, lambfit

    except Exception as e:
        print(f"WARNING: Fit failed. Returning NaNs. (Error: {e})")
        return np.nan, np.nan, np.nan, np.nan, np.nan


# ==============================================================================
# Aggregate Plotting Functions
# ==============================================================================

def plot_lambdas(lambfits, filename=None):
    lambs_clean = np.array(lambfits)[~np.isnan(lambfits)]
    avg_lamb = np.average(lambs_clean)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(lambs_clean, bins=15, edgecolor='black', color='mediumpurple', alpha=0.8)
    ax.axvline(avg_lamb, color='r', linestyle='--', linewidth=2, label=f'Avg = {avg_lamb:.3f}')
    
    ax.set_title(r'$\lambda$ Parameter Distribution', fontsize=16)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if filename is not None:
        savefig(filename + "_lambdas")
    plt.show()

def plot_mutual_inductances(Minfits, Mcfits, filename=None):
    min_clean = np.array(Minfits)[~np.isnan(Minfits)] * 1e12 # Convert to pH
    mc_clean = np.array(Mcfits)[~np.isnan(Mcfits)] * 1e12   # Convert to pH
    
    avg_min = np.average(min_clean)
    avg_mc = np.average(mc_clean)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Mutual Inductance Distributions', fontsize=16)

    ax1.hist(min_clean, bins=15, edgecolor='black', color='skyblue', alpha=0.9)
    ax1.axvline(avg_min, color='r', linestyle='--', linewidth=2, label=f'Avg = {avg_min:.1f} pH')
    ax1.set_title(r'Input Mutual Inductance ($M_{in}$)')
    ax1.set_xlabel('Inductance (pH)')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.hist(mc_clean, bins=15, edgecolor='black', color='lightgreen', alpha=0.9)
    ax2.axvline(avg_mc, color='r', linestyle='--', linewidth=2, label=f'Avg = {avg_mc:.1f} pH')
    ax2.set_title(r'Coupling Mutual Inductance ($M_c$)')
    ax2.set_xlabel('Inductance (pH)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if filename is not None:
        savefig(filename + "_mutual_inductances")
    plt.show()

def plot_fbs(fbfits, filename=None):
    fb_clean = np.array(fbfits)[~np.isnan(fbfits)]
    avg_fb = np.average(fb_clean)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(fb_clean, bins=15, edgecolor='black', color='salmon', alpha=0.8)
    ax.axvline(avg_fb, color='r', linestyle='--', linewidth=2, label=f'Avg = {avg_fb/1e9:.4f} GHz')
    
    ax.set_title('Base Resonator Frequency ($f_b$) Distribution', fontsize=16)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Count')
    ax.xaxis.set_major_formatter(EngFormatter(unit='Hz'))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if filename is not None:
        savefig(filename + "_fbs")
    plt.show()

def plot_I0s(I0fits, filename=None):
    I0_clean = np.array(I0fits)[~np.isnan(I0fits)] * 1e6 # Convert to uA
    avg_I0 = np.average(I0_clean)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(I0_clean, bins=15, edgecolor='black', color='gold', alpha=0.8)
    ax.axvline(avg_I0, color='r', linestyle='--', linewidth=2, label=f'Avg = {avg_I0:.2f} $\\mu$A')
    
    ax.set_title('Offset Current ($I_0$) Distribution', fontsize=16)
    ax.set_xlabel(r'Current ($\mu$A)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if filename is not None:
        savefig(filename + "_I0s")
    plt.show()

def make_flux_result_plots(I0fits, Minfits, Mcfits, fbfits, lambfits, filename=None):
    """ Wrapper to call all summary plotters """
    print("\n--- uMux Flux Response Summary ---")
    plot_lambdas(lambfits, filename)
    plot_mutual_inductances(Minfits, Mcfits, filename)
    plot_fbs(fbfits, filename)
    plot_I0s(I0fits, filename)