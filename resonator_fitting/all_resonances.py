import numpy as np
import scipy as sp
import circle_fitting as cf
import resonance_fitting as rf
import time
import matplotlib.pyplot as plt
plt.ion()
from scipy.signal import find_peaks
from matplotlib.ticker import EngFormatter

def savefig(filename):
    plt.savefig("../../figures/NIST_resonators_test/"+filename+".pdf")

def fit_all_resonators(f, s21, tau, dist = 1.0e6, showplot = False, filename = None, condensed_plots_ids = None):
    peaks = find_all_peaks(f, s21, dist = dist, showplot = showplot, filename = filename)
    aas = []
    taus = []
    frs = []
    Qrs = []
    Qcs = []
    Qis = []
    phi0s = []
    chi2s = []
    bws = []

    width = 0.5e6

    for i in range(len(peaks)):
        f_start = f[peaks[i]] - width/2
        f_stop = f[peaks[i]] + width/2
        f_range = f_stop-f_start
        mask = (f >= f_start) & (f <= f_stop)
        f_region = f[mask]
        s21_region = s21[mask]
        plot_mode = 'none'
        individual_filename = None
        if(condensed_plots_ids != None):
            if(i in condensed_plots_ids):
                plot_mode = 'condensed'
                individual_filename = filename

        a, tau, fr, Qr, Qc, phi0, chi2 = rf.refined_fit(f_region, s21_region, tau, plot_mode=plot_mode, filename = individual_filename)
        bw = fr/Qr
        Qi = 1/(1/Qr - 1/Qc)
        aas.append(a)
        taus.append(tau)
        frs.append(fr)
        Qrs.append(Qr)
        Qcs.append(Qc)
        phi0s.append(phi0)
        chi2s.append(chi2)
        bws.append(bw)
        Qis.append(Qi)
    if(showplot):
        make_result_plots(aas, taus, frs, Qrs, Qcs, Qis, bws, phi0s, chi2s, filename)
        
    return aas, taus, frs, Qrs, Qcs, Qis, bws, phi0s, chi2s        
    

def find_all_peaks(f, s21,dist = 1.0e6 , showplot = False, filename = None):
    s21_flipped = s21.max() - np.abs(s21)

    excluded_frequencies = [5.772e9]

    peaks_full, _ = find_peaks(s21_flipped, prominence=0.7, distance=60)

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
    peaks = np.array(final_peaks_list)
    
    if(showplot):
        plt.vlines(x=f[peaks], ymin=-10, ymax=21, color='red', linestyle='--', 
           label='Peak Locations')

        plt.plot(f, 20*np.log10(np.abs(s21)))
        plt.ylabel("S21 (dB)")
        plt.xlabel("Freq (Hz)")
        
        if(filename != None):
            savefig(filename+"_peak_locations")
            
        plt.show()
        
    return peaks



def plot_frs(frs, filename = None):
    """
    Analyzes and plots resonator frequencies and their spacing distribution.

    This function creates a two-panel plot:
    1. A plot of resonator frequencies vs. their index.
    2. A histogram of the spacing between adjacent resonators.

    It also calculates and annotates the average spacing on the histogram.

    Args:
        frs (array-like): A list or array of resonator frequencies in Hz.
    """
    # --- Data Preparation ---
    # Ensure frequencies are sorted for correct spacing calculation
    frs_sorted = np.sort(frs)
    
    # Calculate spacing between adjacent resonators
    fr_spacing = np.diff(frs_sorted)

    # Filter out unusually large gaps (>10 MHz) to get a more representative average
    fr_spacing_filtered = fr_spacing[fr_spacing < 10e6]
    avg_spacing = np.average(fr_spacing_filtered)

    print("--- Resonator Analysis ---")
    print(f"Number of resonators: {len(frs_sorted)}")
    print(f"Average frequency: {np.average(frs_sorted)/1e9:.4f} GHz")
    print(f"Average spacing (filtered < 10 MHz): {avg_spacing/1e6:.2f} MHz\n")

    # --- Plotting ---
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(10, 8), 
        gridspec_kw={'height_ratios': [1, 1]}
    )
    fig.suptitle('Resonator Frequency Analysis', fontsize=16)

    # --- Top Plot: Resonator Frequencies ---
    resonator_indices = np.arange(len(frs_sorted))
    ax1.plot(resonator_indices, frs_sorted, marker='o', linestyle='-', color='b')
    ax1.set_title('Resonator Frequencies vs. Index')
    ax1.set_xlabel('Resonator Index')
    ax1.set_ylabel('Frequency')
    ax1.yaxis.set_major_formatter(EngFormatter(unit='Hz'))
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Bottom Plot: Spacing Histogram ---
    ax2.hist(fr_spacing_filtered, bins=20, edgecolor='black', color='c')
    ax2.set_title('Frequency Spacing Distribution')
    ax2.set_xlabel('Spacing')
    ax2.set_ylabel('Count')
    ax2.xaxis.set_major_formatter(EngFormatter(unit='Hz'))

    # Add a vertical line to mark the average spacing
    ax2.axvline(
        avg_spacing, 
        color='r', 
        linestyle='--', 
        linewidth=2, 
        label=f'Average = {avg_spacing/1e6:.2f} MHz'
    )
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout for a clean look and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if(filename != None):
        filename_fr = filename + "_frs_"
        savefig(filename_fr)
    plt.show()

def plot_bws(bws, filename):
    """
    Analyzes and plots a histogram of resonator bandwidths.

    This function creates a histogram of the provided bandwidths and
    annotates it with the average value.

    Args:
        bws (array-like): A list or array of resonator bandwidths in Hz.
    """
    # --- Data Preparation ---
    bws_array = np.asarray(bws)
    avg_bw = np.average(bws_array)

    print("--- Bandwidth Analysis ---")
    print(f"Number of resonators: {len(bws_array)}")
    print(f"Average bandwidth: {avg_bw/1e3:.2f} kHz")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the histogram of the bandwidths
    ax.hist(bws_array, bins=15, edgecolor='black', color='lightgreen')
    
    # Add a vertical line to mark the average bandwidth
    ax.axvline(
        avg_bw, 
        color='r', 
        linestyle='--', 
        linewidth=2, 
        label=f'Average = {avg_bw/1e3:.2f} kHz'
    )
    
    # --- Formatting and Labels ---
    ax.set_title('Resonator Bandwidth Distribution', fontsize=16)
    ax.set_xlabel('Bandwidth')
    ax.set_ylabel('Count')
    # Use an engineering formatter for clean axis units (e.g., kHz)
    ax.xaxis.set_major_formatter(EngFormatter(unit='Hz'))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if(filename != None):
        filename_bw = filename + "_bws_"
        savefig(filename_bw)
    plt.show()

def plot_Qs(qcs,qrs,qis, filename):


    # Calculate averages
    avg_qc = np.average(qcs)
    avg_qr = np.average(qrs)
    avg_qi = np.average(qis)
    
    print("--- Q Factor Analysis ---")
    print(f"Average Coupling Q (Qc): {avg_qc:,.0f}")
    print(f"Average Internal Q (Qi): {avg_qi:,.0f}")
    print(f"Average Loaded Q (Qr):   {avg_qr:,.0f}\n")

    # --- Plotting ---
    # Create a figure with three subplots arranged horizontally
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.suptitle('Resonator Q Factor Distributions', fontsize=16)

    # --- Helper function for plotting to avoid repeating code ---
    def plot_hist(ax, data, title, color):
        avg_val = np.average(data)
        ax.hist(data, bins=15, color=color, edgecolor='black', alpha=0.9)
        ax.axvline(
            avg_val,
            color='r',
            linestyle='--',
            linewidth=2,
            label=f'Avg = {avg_val/1e3:.1f}k'
        )
        ax.set_title(title)
        ax.set_xlabel('Q Factor')
        ax.set_ylabel('Count')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    # --- Create each subplot ---
    plot_hist(axes[0], qcs, 'Coupling Q ($Q_c$)', 'skyblue')
    plot_hist(axes[1], qis, 'Internal Q ($Q_i$)', 'lightgreen')
    plot_hist(axes[2], qrs, 'Loaded Q ($Q_r$)', 'salmon')
    
    # Adjust layout for a clean look and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if(filename != None):
        filename_qs = filename + "_qs_"
        savefig(filename_qs)
    plt.show()

def plot_chi2s(chi2s, filename):
    """
    Analyzes and plots a histogram of chi-squared values from fits.

    This function is ideal for visualizing the goodness-of-fit across a
    set of resonators. It creates a histogram and annotates it with the
    average and standard deviation.

    Args:
        chi2s (array-like): A list or array of chi-squared values.
    """
    # --- Data Preparation ---
    chi2s_array = np.asarray(chi2s)
    avg_chi2 = np.average(chi2s_array)
    std_chi2 = np.std(chi2s_array)

    print("--- Chi-Squared Analysis ---")
    print(f"Number of fits: {len(chi2s_array)}")
    print(f"Average Chi-Squared: {avg_chi2:.2f}")
    print(f"Std. Dev. of Chi-Squared: {std_chi2:.2f}\n")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the histogram of chi-squared values
    ax.hist(chi2s_array, bins=15, edgecolor='black', color='mediumpurple', alpha=0.8)

    # Add a vertical line to mark the average value
    ax.axvline(
        avg_chi2,
        color='r',
        linestyle='--',
        linewidth=2,
        label=f'Average = {avg_chi2:.2f}'
    )

    # --- Formatting and Labels ---
    # Use LaTeX for the chi-squared symbol
    ax.set_title(r'Chi-Squared ($\chi^2$) Distribution of Fits', fontsize=16)
    ax.set_xlabel(r'$\chi^2$ Value')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a text box for the standard deviation for more context
    stats_text = f'Std. Dev. = {std_chi2:.2f}'
    ax.text(0.95, 0.95, stats_text, 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    
    if(filename != None):
        filename_chi2 = filename + "_chi2s_"
        savefig(filename_chi2)
    plt.show()
    
    
def make_result_plots(aas, taus, frs, Qrs, Qcs, Qis, bws, phi0s, chi2s, filename = None):
    plot_frs(frs, filename)
    plot_bws(bws, filename)
    plot_Qs(Qcs,Qrs,Qis, filename)
    plot_chi2s(chi2s, filename)


