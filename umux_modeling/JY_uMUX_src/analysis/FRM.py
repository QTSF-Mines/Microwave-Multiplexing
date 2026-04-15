# =========================
# Imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
from scipy.signal import correlate, find_peaks
from scipy.optimize import curve_fit


from iminuit import Minuit
from iminuit.cost import LeastSquares
import iminuit


# =========================
# Periodicity Fit Functions
# =========================

def cosine_model(t, A, period, phase, offset):
        """Cosine function: A * cos(2*pi*t/period + phase) + offset"""
        return A * np.cos(2 * np.pi * t / period + phase) + offset


def periodicity_iminuit_fit(time, trace, period_guess, sigma=None, plot=False):
    """
    Estimate the periodicity of the flux ramp modulated signal using iminuit method.
    
    Parameters:
    time (np.ndarray): The time axis corresponding to the trace.
    trace (np.ndarray): The input signal trace to be tested.
    period_guess (float): The initial guess for the period of the signal.
    plot (bool): Whether to plot the fitting result.
    Returns:
    period_fit (float): The estimated period of the signal.
    period_error (float): The uncertainty of the estimated period.
    """

    # Use iminuit to fit the cosin model
    # The uncertainty of the initial parameter is unknown, so assume it's equally weighted between data points.
    # Then set the uncertainty to be a free parameter in the likelihood function. Then try to fit for it along with other cosine parameters.

    
    A_guess = (np.max(trace) - np.min(trace)) / 2
    offset_guess = np.mean(trace)
    phase_guess = 0
    period_guess = period_guess

    if sigma==None:

        sigma_guess = np.std(trace - np.mean(trace))
        
        # Need to custom write the least squares function to include sigma as a parameter
        def nll(sigma, A, period, phase, offset):
            if sigma <= 0:  # enforce positivity
                return np.inf
            r = trace - cosine_model(time, A, period, phase, offset)
            n = len(trace)
            return np.sum(r**2) / sigma**2 + n * np.log(sigma**2)

        m = Minuit(nll, A=A_guess, period=period_guess, phase=phase_guess, offset=offset_guess, sigma=sigma_guess)
        m.migrad()
        m.minos()
        period_fit = m.values['period']
        sigma = m.values['sigma']
    else:

        least_squares = LeastSquares(time, trace, sigma, cosine_model)
        m = Minuit(least_squares, A=A_guess, period=period_guess, phase=phase_guess, offset=offset_guess)
        m.migrad()
        m.minos()
        period_fit = m.values['period']

    if plot:
        plt.figure()
        plt.errorbar(time*1e3, trace, yerr=sigma, label='Raw signal')
        plt.plot(time*1e3, cosine_model(time, m.values['A'], m.values['period'], m.values['phase'], m.values['offset']), label='Fitted cosine', linestyle='--')
        plt.xlabel('Time (ms)')
        plt.ylabel('Signal amplitude')
        plt.title(f'Fitted Period: {period_fit*1e6:.2f} us')
        plt.legend()
        plt.show()
    
    return period_fit, m.errors['period'], sigma

def periodicity_curve_fit_bulk(data_ds, sweep_dims, start_pt_ds, period_guess, domain='imag', sigma=None, plot=False, alpha_guess=0.2, ramp_num=(0, 100)):
    """
    Estimate the periodicity of the flux ramp modulated signal using iminuit method for bulk data.
    
    Parameters:
    data_ds (xr.Dataset): The input xarray Dataset containing the signal trace to be tested.
    sweep_dims (list): List of dimension names that are sweep parameters.
    period_guess (float): The initial guess for the period of the signal.
    domain (str): 'imag' or 'real' to specify which part of the signal to use.
    sigma (float): The uncertainty of the signal. If None, it will be fitted.
    plot (bool): Whether to plot the fitting result for each trace.
    alpha_guess (float): Initial guess for the alpha parameter in the fitting.
    Returns:
    data_ds (xr.Dataset): The input xarray Dataset with an additional coordinate 'fitted_period' containing the estimated period of the flux ramp for each resonator.
    """


    sweeps = [data_ds[dim].values for dim in sweep_dims]
    num_sweeps = np.prod([len(sweep) for sweep in sweeps])
    pts_per_ramp = int(np.round(1/data_ds.ramp_freq*data_ds.samp_freq))
    ramp_idx_list = np.linspace(ramp_num[0], ramp_num[1], ramp_num[1] - ramp_num[0] + 1, dtype=int)  # use the provided ramp_num parameter

    fitted_periods = np.zeros((num_sweeps, len(ramp_idx_list)))
    fitted_errors = np.zeros((num_sweeps, len(ramp_idx_list)))

    for i in tqdm(range(num_sweeps), desc="Estimating periodicity for traces"):
        # Create a dictionary to hold the current sweep parameters
        sweep_params = {dim: sweeps[idx][val] for idx, (dim, val) in enumerate(zip(sweep_dims, np.unravel_index(i, [len(sweep) for sweep in sweeps])))}
        start_pt_trace = int(start_pt_ds.start_idx.sel(sweep_params, method='nearest').values)

        # Partition the time trace into each ramp interval based on ramp_freq, samp_freq. Then get rid of the start of each ramp based on alpha_guess
        for j in range(len(ramp_idx_list)):
            start_pt = int(ramp_idx_list[j]*pts_per_ramp + alpha_guess*pts_per_ramp + start_pt_trace)
            end_pt = int((ramp_idx_list[j]+1)*pts_per_ramp + start_pt_trace)
            if domain == 'imag':
                trace_segment = data_ds.S21_norm.sel(sweep_params, method='nearest').values.imag[start_pt:end_pt]
            elif domain == 'real':
                trace_segment = data_ds.S21_norm.sel(sweep_params, method='nearest').values.real[start_pt:end_pt]
            time_segment = data_ds.time_trace.values[start_pt:end_pt] - data_ds.time_trace.values[start_pt]

            # Call the periodicity_curve_fit function with the current sweep parameters
            period_fit, period_error = periodicity_curve_fit(time_segment, trace_segment, period_guess, plot=plot)
            fitted_periods[i, j] = period_fit
            fitted_errors[i, j] = period_error
            


    # Make the fitted_periods and fitted_errors into one new xr.Dataset with 2 variables
    # Need to reshape fitted_periods and fitted_errors into the shape of sweep_dim1 x sweep_dim2 x ... x ramp_idx
    fitted_periods=xr.Dataset(
        {
            "fitted_period": (sweep_dims + ["ramp_idx"], fitted_periods.reshape(*[len(sweep) for sweep in sweeps], len(ramp_idx_list))),
            "fitted_error": (sweep_dims + ["ramp_idx"], fitted_errors.reshape(*[len(sweep) for sweep in sweeps], len(ramp_idx_list))),
        },
        coords={dim: sweeps[idx] for idx, dim in enumerate(sweep_dims)}
        | {"ramp_idx": ramp_idx_list},
    )

    return fitted_periods


def periodicity_curve_fit(time, trace, period_guess, plot=False, plot_period=None):
    """
    Fit the periodicity of the flux ramp modulated signal using curve fitting method.

    Parameters: 
    time (np.ndarray): The time axis corresponding to the trace.
    trace (np.ndarray): The input signal trace to be fitted.
    period_guess (float): The initial guess for the period of the signal.

    Returns:
    period_fit (float): The fitted period of the signal.
    """

    # Initial parameter guesses: [amplitude, period, phase, offset]
    A_guess = (np.max(trace) - np.min(trace)) / 2
    offset_guess = np.mean(trace)
    phase_guess = 0
    p0 = [A_guess, period_guess, phase_guess, offset_guess]
    
    # Perform curve fitting
    popt, pcov = curve_fit(cosine_model, time, trace, p0=p0)
    period_fit = popt[1]
    period_error = np.sqrt(np.diag(pcov))[1]

    if plot:
        plt.figure()
        plt.plot(time*1e6, trace, label='Raw signal')
        plt.plot(time*1e6, cosine_model(time, *popt), label='Fitted cosine', linestyle='--')
        if plot_period is not None:
            plt.plot(time*1e6, cosine_model(time, p0[0], plot_period, p0[2], p0[3]), color='r', linestyle='--', label='Other period guess')
        plt.xlabel('Time (μs)')
        plt.ylabel('Signal amplitude')
        plt.title(f'Fitted Period: {period_fit*1e6:.2f} us')
        plt.legend()
        plt.show()
    
    return period_fit, period_error



def periodicity_FFT_fit(time, trace, period_guess, plot=False):
    """
    Estimate the periodicity of the flux ramp modulated signal using FFT method.
    
    Parameters:
    time (np.ndarray): The time axis corresponding to the trace.
    trace (np.ndarray): The input signal trace to be tested.
    period_guess (float): The initial guess for the period of the signal.
    
    Returns:
    period_fit (float): The estimated period of the signal.
    """

    samp_time = time[1] - time[0]

    # Compute FFT
    N = len(trace)
    fft_vals = np.fft.fft(trace - np.mean(trace))
    fft_freqs = np.fft.fftfreq(N, d=samp_time)
    fft_magnitudes = np.abs(fft_vals)

    # Only consider positive frequencies
    pos_mask = fft_freqs > 0
    fft_freqs = fft_freqs[pos_mask]
    fft_magnitudes = fft_magnitudes[pos_mask]
    # Find the peak frequency
    peak_idx = np.argmax(fft_magnitudes)
    freq_peak = fft_freqs[peak_idx]
    period_fit = 1 / freq_peak

    if plot:
        plt.figure()
        plt.plot(fft_freqs, fft_magnitudes)
        plt.axvline(1/period_guess, color='r', linestyle='--', label='Initial guess')
        plt.axvline(freq_peak, color='g', linestyle='--', label='Estimated peak frequency')
        plt.axvline(freq_peak*2, color='b', linestyle='--', label='2nd Harmonic')
        plt.axvline(freq_peak*3, color='m', linestyle='--', label='3rd Harmonic')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('FFT Magnitude')
        plt.title(f'Estimated Period: {period_fit*1e6:.2f} us')
        plt.loglog()
        plt.legend()
        plt.show()


    return period_fit







# =========================
# Plotting Functions
# =========================
def plot_raw_data_triangle(time, trace, alpha, ramp_freq, n_period, idx_list, start_pt=0, shift=False):
    samp_time = time[1] - time[0]
    pts_per_ramp = 1/ramp_freq/samp_time/2
    free_period = (1-alpha) / (ramp_freq*2) / n_period
    trace = trace[start_pt:]
    time = time[start_pt:] - time[start_pt]

    plt.figure()
    
    for i in idx_list:
        # Find the start and end points of the ramp
        start_pt = i/(ramp_freq*2)/samp_time
        end_pt = int(np.floor((i+1)/(ramp_freq*2)/samp_time))

        # Throw away the first alpha fraction of points
        start_pt += alpha*(end_pt-start_pt)
        start_pt = int(np.floor(start_pt))
        
        # Plot the signal
        trace_norm = trace[start_pt:end_pt]-(np.max(trace[start_pt:end_pt])+np.min(trace[start_pt:end_pt]))/2
        trace_norm = trace_norm/np.max(trace_norm)

        if i == idx_list[0]:
            plt.plot(np.linspace(0, len(trace_norm), len(trace_norm), endpoint=False), trace_norm, alpha=0.5, c='b', label='Raw signal')
        else:
            if i % 2 == 0:
                plt.plot(np.linspace(0, len(trace_norm), len(trace_norm), endpoint=False), trace_norm, alpha=0.5, c='b')
            else:
                plt.plot(np.linspace(0, len(trace_norm), len(trace_norm), endpoint=False), trace_norm[::-1], alpha=0.5, c='b')


    free_sin = np.sin(2*np.pi/free_period*(time[start_pt:end_pt]-time[start_pt]))
    plt.plot(np.linspace(0, len(trace_norm), len(trace_norm), endpoint=False), free_sin, alpha=0.5, c='r', label='Free oscillation')

    if shift:
        phi_residual = alpha/((1-alpha)/n_period)*2*np.pi
        free_sin_shifted = np.sin(2*np.pi/free_period*(time[start_pt:end_pt]-time[start_pt]+phi_residual))
        plt.plot(np.linspace(0, len(free_sin_shifted), len(free_sin_shifted), endpoint=False), free_sin_shifted, alpha=0.5, c='g', label='Shifted free oscillation')
        print(f'Shifted by {phi_residual/np.pi:.2f} pi')

    plt.xlabel('index') 
    plt.ylabel('Normalized amplitude')
    plt.legend(loc='upper right')
    return 

def plot_raw_data_sawtooth(time, trace, alpha, ramp_freq, samp_freq, n_period, idx_list, start_pt=0):
    pts_per_sawtooth = 1/ramp_freq*samp_freq
    free_period = (1-alpha) / (ramp_freq) / n_period
    trace = trace[start_pt:]
    time = time[start_pt:] - time[start_pt]

    plt.figure()
    
    for i in idx_list:
        # Find the start and end points of the sawtooth
        start_pt = i/ramp_freq*samp_freq
        end_pt = int(np.floor((i+1)/ramp_freq*samp_freq))

        # Throw away the first alpha fraction of points
        start_pt += alpha*(end_pt-start_pt)
        start_pt = int(np.floor(start_pt))
        
        # Plot the signal
        trace_norm = trace[start_pt:end_pt]-(np.max(trace[start_pt:end_pt])+np.min(trace[start_pt:end_pt]))/2
        trace_norm = trace_norm/np.max(trace_norm)


        if i == idx_list[0]:
            plt.plot(np.linspace(0, len(trace_norm), len(trace_norm), endpoint=False), trace_norm, alpha=0.5, c='b', label='Raw signal')
        else:
            plt.plot(np.linspace(0, len(trace_norm), len(trace_norm), endpoint=False), trace_norm, alpha=0.5, c='b')

        free_sin = np.sin(2*np.pi/free_period*(time[start_pt:end_pt]-time[start_pt]))
        free_cos = np.cos(2*np.pi/free_period*(time[start_pt:end_pt]-time[start_pt]))
        plt.plot(np.linspace(0, len(trace_norm), len(trace_norm), endpoint=False), free_sin, alpha=0.5, c='r', label='Free oscillation')

        demod_value = np.angle(np.sum((trace_norm * free_cos)) + 1j*np.sum((trace_norm * free_sin)))
        print(demod_value)

    plt.xlabel('index') 
    plt.ylabel('Normalized amplitude')
    plt.legend(loc='upper right')
    return


# =========================
# Linearity Test Functions
# =========================


def estimate_period_single_trace(trace, vol_bias):
    """
    Estimate the periodicity of the flux ramp modulated signal using autocorrelation method.
    
    Parameters:
    trace (np.ndarray): The input signal trace to be tested.
    vol_bias (np.ndarray): The voltage bias corresponding to the trace.
    
    Returns:
    vol_period (float): The estimated voltage period of the flux ramp.
    """

    # subtract mean to avoid large zero-lag spike
    trace = trace - np.mean(trace)

    # compute autocorrelation
    corr = correlate(trace, trace, mode="full")
    lags = np.arange(-len(trace)+1, len(trace))
    corr = corr / np.max(corr)  # normalize

    # only keep positive lags
    corr = corr[len(corr)//2:]
    lags = lags[len(lags)//2:]

    # find peaks in the autocorrelation
    peaks, _ = find_peaks(corr)

    if len(peaks) < 2:
        return None  # no clear periodicity

    # first peak is at lag 0, so use difference between the next peaks
    vol_period = np.diff(vol_bias[peaks])[0]
    return vol_period, corr, peaks

def periodicity_calibr_bulk(data_ds, domain='imag'):
    """
    Estimate the periodicity of the flux ramp modulated signal using autocorrelation method for bulk data.
    
    Parameters:
    data_ds (xr.Dataset): The input xarray Dataset containing the signal trace to be tested.
    
    Returns:
    data_ds (xr.Dataset): The input xarray Dataset with an additional coordinate 'vol_period' containing the estimated voltage period of the flux ramp for each resonator.
    """

    num_res = data_ds.sizes['center_freq']

    vol_period = np.zeros(num_res)

    for res_idx in tqdm(range(num_res), desc="Estimating periodicity for resonators"):
        # Take both the I and Q channel data for better estimation

        if domain == 'imag':
            vol_period_est, _, _ = estimate_period_single_trace(data_ds.S21_norm.sel(center_freq=data_ds.center_freq[res_idx], method='nearest').values.imag, data_ds.vol_bias.values)
        elif domain == 'real':
            vol_period_est, _, _ = estimate_period_single_trace(data_ds.S21_norm.sel(center_freq=data_ds.center_freq[res_idx], method='nearest').values.real, data_ds.vol_bias.values)

        if vol_period_est is not None:
            vol_period[res_idx] = vol_period_est
        else:
            vol_period[res_idx] = np.nan
            print(f"Warning: No clear periodicity found for resonator at {data_ds.center_freq[res_idx].values} Hz")

    data_ds = data_ds.assign_coords(vol_period=("center_freq", vol_period))

    return data_ds

def triangle_wave(t, A, ramp_freq, offset, phase=0):
    """
    Generate a triangle wave.
    
    Parameters:
    t : array-like
        Time array.
    A : float
        Amplitude of the triangle wave. (peak to peak)
    ramp_freq : float
        Frequency of the ramp (number of ramps per second).
    offset : float
        Vertical offset of the triangle wave. Offset=0 means the wave oscillates between a and A.
    phase : float, optional
        Phase shift of the triangle wave, in cycles (0 to 1). Default is 0.
    
    Returns:
    array-like
        Triangle wave values at time t.
        
    """
    
    return A * np.abs(2 * (t * ramp_freq + phase - np.floor(t * ramp_freq + 0.5 + phase))) + offset

def linearity_trace_cut(trace, ramp_freq, samp_time, waveform='triangle', plot=False):
    """
    Cut the trace to only include the linear region. 
    
    Parameters:
    trace (np.ndarray): The input signal trace to be cut.
    ramp_freq (float): The frequency of the ramp in Hz.
    samp_length (float): The length of the sample in seconds.

    Returns:
    trace_cut (np.ndarray): The cut signal trace.
    """

    time_axis = np.linspace(0, samp_time, len(trace), endpoint=False)
    ramp_period = 1/ramp_freq

    if waveform == 'triangle':
        A_guess = (np.max(trace)-np.min(trace))
        offset_guess = np.min(trace)

        # Find the guess for phase
        # Only look at the first period
        pts_per_ramp = int(1/(ramp_freq)/samp_time*len(trace))
        phase_guess = 1-time_axis[trace[:pts_per_ramp].argmin()]/ramp_period

        # Fit a triangle wave with ramp_freq to the time axis
        popt, _ = curve_fit(lambda t, A, offset, phase: triangle_wave(t, A, ramp_freq, offset, phase), time_axis, trace, 
                            p0=[A_guess, offset_guess, phase_guess], bounds=([0, -np.inf, 0], [np.inf, np.inf, 1]))
        A, offset, phase = popt

        if phase <= 0.5:
            linear_start_time = (0.5-phase) * ramp_period
            linear_end_time = (1-phase) * ramp_period
        else:
            linear_start_time = (1-phase) * ramp_period
            linear_end_time = (1.5-phase) * ramp_period

        linear_start_idx = int(np.floor(linear_start_time/samp_time*len(trace)))
        linear_end_idx = int(np.floor(linear_end_time/samp_time*len(trace)))

        # Then move the trace so that the linear region starts from zero
        # Fit a line to the linear region to find the offset

        trace_cut = np.abs(trace[linear_start_idx:linear_end_idx]-trace[linear_start_idx])
        popt_line, _ = curve_fit(lambda x, k, b: k*x + b, time_axis[linear_start_idx:linear_end_idx]-time_axis[linear_start_idx], trace_cut)
        trace_cut = trace_cut - popt_line[1]



        #trace_cut = np.abs(trace[linear_start_idx:linear_end_idx]-trace[linear_start_idx])

    if plot:
        plt.figure()
        plt.plot(time_axis, trace, label='Raw trace')
        plt.plot(time_axis, triangle_wave(time_axis, popt[0], ramp_freq, popt[1], popt[2]), label='Fitted triangle wave')
        plt.plot(time_axis, triangle_wave(time_axis, A_guess, ramp_freq, offset_guess, phase_guess), label='Initial guess')
        plt.axvline(linear_start_time, color='r', linestyle='--', label='Linear region start')
        plt.axvline(linear_end_time, color='g', linestyle='--', label='Linear region end')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal amplitude')
        plt.legend()
        plt.title('Linearity region cut based on triangle wave fit')
        plt.show()


    return trace_cut

def linearity_trace_cut_bulk(data_ds, sweep_dims, trace_cut_args):
    """
    Cut the traces in a dataset to only include the linear region.

    Parameters:
    data_ds (xarray.Dataset): The input dataset containing traces.
    sweep_dims (list): List of dimension names that are sweep parameters.
    trace_cut_args (dict): Dictionary of arguments to be passed to the linearity_trace_cut function.

    Returns:
    xarray.Dataset: The dataset with cut traces.
    """

    sweeps = [data_ds[dim].values for dim in sweep_dims]
    num_sweeps = np.prod([len(sweep) for sweep in sweeps])

    cut_traces = []

    for i in range(num_sweeps):
        # Create a dictionary to hold the current sweep parameters
        sweep_params = {dim: sweeps[idx][val] for idx, (dim, val) in enumerate(zip(sweep_dims, np.unravel_index(i, [len(sweep) for sweep in sweeps])))}
        # Call the linearity_trace_cut function with the current sweep parameters
        
        cut_trace = linearity_trace_cut(data_ds.FRM_demod.sel(sweep_params, method='nearest').values, **trace_cut_args)
        cut_traces.append(cut_trace)
    
    # Trim the cut_traces to have the same length
    min_length = min([len(trace) for trace in cut_traces])
    cut_traces = [trace[:min_length] for trace in cut_traces]
    cut_traces = np.array(cut_traces)
    
    cut_traces = xr.DataArray(cut_traces.reshape([len(sweep) for sweep in sweeps] + [min_length]), 
                              coords = {sweep_dims[i]: sweeps[i] for i in range(len(sweep_dims))} 
                              | {'time_trace': np.linspace(0, min_length*trace_cut_args['samp_time']/min_length, min_length, endpoint=False)}, 
                              dims = sweep_dims + ['time_trace'])
    return cut_traces


# =========================
# Demodulation Functions
# =========================
def FRM_demod_triangle_single_trace(trace, start_pt, alpha, ramp_freq, samp_freq, n_period):

    """
    Demodulate the flux ramp modulated signal using the triangle wave demodulation method.
    
    Parameters:
    trace (np.ndarray): The input signal trace to be demodulated.
    time (np.ndarray): The time axis corresponding to the trace.
    alpha (float): Fraction of the ramp to be thrown away at the beginning of each ramp.
    ramp_freq (float): Frequency of the flux ramp in Hz.
    n_period (int): Number of periods to include in the demodulation.

    Returns:
    FRM_demod_time (np.ndarray): The time axis corresponding to the demodulated data
    FRM_demod_data (np.ndarray): The demodulated data 
    """

    # Normalize the trace and remove DC offset. Also remove the points before start_pt
    trace = trace - (np.max(trace) + np.min(trace)) / 2
    trace = trace / np.max(np.abs(trace))
    trace = trace[start_pt:]

    free_period = (1-alpha) / (ramp_freq*2) / n_period
    pts_per_ramp = int(np.round(1/ramp_freq*samp_freq/2))
    num_ramp = len(trace)//(pts_per_ramp*2)*2 # want even number of rise and fall ramps
    

    # Trim the trace to have an integer number of ramps
    trace = trace[:int(num_ramp*pts_per_ramp)]

    demod_time = np.linspace(0, num_ramp, num_ramp)/ramp_freq

    # First reshape the trace into ramps, then demodulate each ramp
    trace = trace.reshape((num_ramp, int(pts_per_ramp)))
    trace_rise = trace[::2]
    trace_fall = trace[1::2]
    ramp_start_pt = int(alpha*pts_per_ramp)
    ramp_end_pt = int(pts_per_ramp)

    time = np.linspace(0, (ramp_end_pt - ramp_start_pt)/samp_freq, ramp_end_pt - ramp_start_pt, endpoint=False)
    free_cos = np.cos(2*np.pi/free_period*(time))
    free_sin = np.sin(2*np.pi/free_period*(time))
    
    demod_trace_rise = np.angle(np.sum(trace_rise[:, ramp_start_pt:ramp_end_pt]*free_cos, axis=1) + 1j*np.sum(trace_rise[:, ramp_start_pt:ramp_end_pt]*free_sin, axis=1))
    demod_trace_fall = np.angle(np.sum(trace_fall[:, ramp_start_pt:ramp_end_pt]*free_cos, axis=1) + 1j*np.sum(trace_fall[:, ramp_start_pt:ramp_end_pt]*free_sin, axis=1))


    return demod_time, np.unwrap(demod_trace_rise), np.unwrap(demod_trace_fall)

def FRM_demod_sawtooth_single_trace(trace, start_pt, alpha, ramp_freq, samp_freq, n_period):

    """
    Demodulate the flux ramp modulated signal using the sawtooth demodulation method.
    
    Parameters:
    trace (np.ndarray): The input signal trace to be demodulated.
    time (np.ndarray): The time axis corresponding to the trace.
    alpha (float): Fraction of the ramp to be thrown away at the beginning of each ramp.
    sawtooth_freq (float): Frequency of the flux ramp in Hz.
    n_period (int): Number of periods to include in the demodulation.

    Returns:
    FRM_demod_time (np.ndarray): The time axis corresponding to the demodulated data
    FRM_demod_data (np.ndarray): The demodulated data 
    """

    # Normalize the trace and remove DC offset. Also remove the points before start_pt
    trace = trace - (np.max(trace) + np.min(trace)) / 2
    trace = trace / np.max(np.abs(trace))
    trace = trace[start_pt:]

    free_period = (1-alpha) / (ramp_freq) / n_period
    pts_per_ramp = int(np.round(1/ramp_freq*samp_freq))
    num_ramp = len(trace) // pts_per_ramp
    demod_time = np.linspace(0, num_ramp, num_ramp)/ramp_freq
    demod_trace = np.zeros(num_ramp)


    trace = trace[:int(num_ramp*pts_per_ramp)]
    trace = trace.reshape((num_ramp, int(pts_per_ramp)))
    ramp_start_pt = int(alpha*pts_per_ramp)
    ramp_end_pt = int(pts_per_ramp)
    time = np.linspace(0, (ramp_end_pt - ramp_start_pt)/samp_freq, ramp_end_pt - ramp_start_pt, endpoint=False)

    # Calculate the free oscillation signal for this sawtooth
    free_cos = np.cos(2*np.pi/free_period*(time))
    free_sin = np.sin(2*np.pi/free_period*(time))

    # Demodulate the signal
    demod_trace = -np.angle(np.sum(trace[:, ramp_start_pt:ramp_end_pt]*free_cos, axis=1) 
                               + 1j*np.sum(trace[:, ramp_start_pt:ramp_end_pt]*free_sin, axis=1))

    return demod_time, np.unwrap(demod_trace)


def FRM_demod_triangle_PSD_bulk(data_ds, alpha, n_period, start_pt_ds, align='avg'):
    """
    Demodulate the flux ramp modulated signal using the triangle wave demodulation method.
    
    Parameters:
    data_ds (xr.Dataset): The input xarray Dataset containing the signal trace to be demodulated.
    alpha (np.ndarray): Array of fractions of the ramp to be thrown away at the beginning of each ramp, shape (num_res,).
    ramp_freq (float): Frequency of the flux ramp in Hz.
    n_period (np.ndarray): Array of number of periods to include in the demodulation, shape (num_res,).

    Returns:
    FRM_demod_ds (xr.Dataset): The demodulated data stored in an xarray Dataset.
    """

    num_res = data_ds.sizes['center_freq']
    # Check for the dimensions other than center_freq and time_trace
    sweep_dims = [dim for dim in data_ds.dims if dim not in ['center_freq', 'time_trace']]
    if len(sweep_dims) == 0:
        num_reps = 1
    else:
        num_reps = data_ds.sizes[sweep_dims[0]]


    # Prepare the output xarray Dataset
    demod_time = np.linspace(0, int(np.floor(data_ds.time_trace.values[-1]*data_ds.samp_freq))*2, int(np.floor(data_ds.time_trace.values[-1]*data_ds.ramp_freq))*2)/data_ds.ramp_freq
    FRM_demod_data = []
    
    for res_idx in tqdm(range(num_res), desc="Demodulating resonators"):
        for rep_idx in range(num_reps):

            if num_reps == 1:
                trace = data_ds.S21_norm.sel(center_freq=data_ds.center_freq[res_idx], method='nearest').values.imag
                start_pt = int(start_pt_ds.start_idx.sel(center_freq=data_ds.center_freq[res_idx], method='nearest').values)
                
            else:
                trace = data_ds.S21_norm.sel(center_freq=data_ds.center_freq[res_idx], rep=data_ds.rep[rep_idx], method='nearest').values.imag
                start_pt = int(start_pt_ds.start_idx.sel(center_freq=data_ds.center_freq[res_idx], rep=start_pt_ds.rep[rep_idx], method='nearest').values)

            _, demod_trace_rise, demod_trace_fall = FRM_demod_triangle_single_trace(trace, start_pt, alpha[res_idx], data_ds.ramp_freq, data_ds.samp_freq, n_period[res_idx])
            
            # Align the rise and fall traces in time assuming the first point of the rising and falling edge demodulation have the same value
            # Then put the two traces together
            if align == 'avg':
                demod_trace_fall = -demod_trace_fall + np.mean(demod_trace_rise) + np.mean(demod_trace_fall)
            elif align == 'first':
                demod_trace_fall = -demod_trace_fall + demod_trace_rise[0] + demod_trace_fall[0]
            
            demod_trace_full = np.zeros(len(demod_trace_rise) + len(demod_trace_fall))
            demod_trace_full[::2] = demod_trace_rise
            demod_trace_full[1::2] = demod_trace_fall   
            FRM_demod_data.append(demod_trace_full)

    # Trim FRM_demod_data to have the same length in each trace
    min_length = min([len(trace) for trace in FRM_demod_data])
    FRM_demod_data = np.array([trace[:min_length] for trace in FRM_demod_data])
    demod_time = demod_time[:min_length]

    
    FRM_demod_ds = xr.Dataset(
        {"FRM_demod": (("center_freq", "rep", "time_trace"), FRM_demod_data.reshape((num_res, num_reps, -1)))},
        coords={            
            "center_freq": data_ds.center_freq,
            "rep": data_ds.rep if 'rep' in data_ds.dims else np.array([0]),
            "time_trace": demod_time
        }
    )



    return FRM_demod_ds


def FRM_demod_sawtooth_PSD_bulk(data_ds, alpha, n_period, start_pt_ds):  
    """
    Demodulate the flux ramp modulated signal using the sawtooth demodulation method.
    
    Parameters:
    data_ds (xr.Dataset): The input xarray Dataset containing the signal trace to be demodulated.
    alpha (np.ndarray): Array of fractions of the ramp to be thrown away at the beginning of each ramp, shape (num_res,).
    sawtooth_freq (float): Frequency of the flux ramp in Hz.
    n_period (np.ndarray): Array of number of periods to include in the demodulation, shape (num_res,).

    Returns:
    FRM_demod_ds (xr.Dataset): The demodulated data stored in an xarray Dataset.
    """

    num_res = data_ds.sizes['center_freq']
    # Check for the dimensions other than center_freq and time_trace
    sweep_dims = [dim for dim in data_ds.dims if dim not in ['center_freq', 'time_trace']]
    if len(sweep_dims) == 0:
        num_reps = 1
    else:
        num_reps = data_ds.sizes[sweep_dims[0]]

    # Prepare the output xarray Dataset
    demod_time = np.linspace(0, int(np.floor(data_ds.time_trace.values[-1]*data_ds.ramp_freq)), int(np.floor(data_ds.time_trace.values[-1]*data_ds.ramp_freq)))/data_ds.ramp_freq
    FRM_demod_data = []
    
    for res_idx in tqdm(range(num_res), desc="Demodulating resonators"):
        for rep_idx in range(num_reps):
            
            if num_reps == 1:
                trace = data_ds.S21_norm.sel(center_freq=data_ds.center_freq[res_idx], method='nearest').values.imag
                start_pt = int(start_pt_ds.start_idx.sel(center_freq=data_ds.center_freq[res_idx], method='nearest').values)
            else:
                trace = data_ds.S21_norm.sel(center_freq=data_ds.center_freq[res_idx], rep=data_ds.rep[rep_idx], method='nearest').values.imag
                start_pt = int(start_pt_ds.start_idx.sel(center_freq=data_ds.center_freq[res_idx], rep=start_pt_ds.rep[rep_idx], method='nearest').values)


            _, demod_trace = FRM_demod_sawtooth_single_trace(trace, start_pt, alpha[res_idx], data_ds.ramp_freq, data_ds.samp_freq, n_period[res_idx])
            
            FRM_demod_data.append(demod_trace)

    # Trim FRM_demod_data to have the same length in each trace
    min_length = min([len(trace) for trace in FRM_demod_data])
    FRM_demod_data = np.array([trace[:min_length] for trace in FRM_demod_data])
    demod_time = demod_time[:min_length]  


    FRM_demod_ds = xr.Dataset(
        {"FRM_demod": (("center_freq", "rep", "time_trace"), FRM_demod_data.reshape((num_res, num_reps, -1)))},
        coords={            
            "center_freq": data_ds.center_freq,
            "rep": data_ds.rep if 'rep' in data_ds.dims else np.array([0]),
            "time_trace": demod_time
        }
    )


    return FRM_demod_ds



# =========================
# Uncertainty propagation Functions
# =========================

def FRM_err_sawtooth_single_trace(trace, period, period_err, start_pt, ramp_freq, samp_freq, n_period):

    """
    Propagate the uncertainty in period to uncertainty in flux ramp demodulated phase for a single trace.
    Parameters:
    trace_cut (np.ndarray): The input signal trace to be demodulated.
    period (float): The estimated period of the signal.
    period_err (float): The uncertainty of the estimated period.
    ramp_freq (float): Frequency of the flux ramp in Hz.
    samp_freq (float): Sampling frequency of the signal in Hz.
    n_period (int): Number of periods to include in the demodulation.
    Returns:
    FRM_demod_err (np.ndarray): The uncertainty in the demodulated phase.
    """

    # Estimate the FRM demodulated phase uncertainty by calculating FRM_phase(period + period_err) - FRM_phase(period)

    alpha = 1-(n_period*ramp_freq*period)
    alpha_err = 1-(n_period*ramp_freq*(period+period_err))


    _, demod_trace = FRM_demod_sawtooth_single_trace(trace, start_pt, alpha, ramp_freq, samp_freq, n_period)
    _, demod_trace_err = FRM_demod_sawtooth_single_trace(trace, start_pt, alpha_err, ramp_freq, samp_freq, n_period)


    demod_err = demod_trace_err - demod_trace

    return demod_trace, demod_err

def FRM_err_sawtooth_PSD_bulk(data_ds, sweep_dims, period_ds, start_pt_ds, n_period):

    """
    Docstring for FRM_err_sawtooth_PSD_bulk
    
    :param data_ds: Description
    :param period_ds: Description
    :param start_pt_ds: Description
    :param n_period: Description
    """ 

    sweeps = [data_ds[dim].values for dim in sweep_dims]
    num_sweeps = np.prod([len(sweep) for sweep in sweeps])
    num_ramps = int(len(data_ds.time_trace)/(data_ds.samp_freq/data_ds.ramp_freq))-10


    # Prepare the output xarray Dataset
    demod_idx = np.linspace(0, num_ramps, num_ramps)
    FRM_demod_data = np.zeros((num_sweeps, num_ramps))  # leave some margin
    FRM_demod_err = np.zeros((num_sweeps, num_ramps))
    
    for sweep_idx in tqdm(range(num_sweeps), desc="Calculating demodulation uncertainty for resonators"):
        # Create a dictionary to hold the current sweep parameters
        sweep_params = {dim: sweeps[idx][val] for idx, (dim, val) in enumerate(zip(sweep_dims, np.unravel_index(sweep_idx, [len(sweep) for sweep in sweeps])))}

        trace = data_ds.S21_norm.sel(sweep_params, method='nearest').values.imag
        start_pt = start_pt_ds.start_idx.sel(sweep_params, method='nearest').values
        period = period_ds.fitted_period.sel(sweep_params, method='nearest').values
        period_err = period_ds.fitted_error.sel(sweep_params, method='nearest').values

        demod_trace, demod_trace_err = FRM_err_sawtooth_single_trace(trace, period, period_err, start_pt, data_ds.ramp_freq, data_ds.samp_freq, n_period)
        
        FRM_demod_data[sweep_idx] = demod_trace[:num_ramps]
        FRM_demod_err[sweep_idx] = demod_trace_err[:num_ramps]    

    # Trim FRM_demod_data to have the same length in each trace

    FRM_demod_ds = xr.Dataset(
        {"FRM_demod": (sweep_dims + ["FRM_time"], FRM_demod_data.reshape([len(sweep) for sweep in sweeps] + [-1])),
         "FRM_demod_err": (sweep_dims + ["FRM_time"], FRM_demod_err.reshape([len(sweep) for sweep in sweeps] + [-1]))},
        coords={        
            **{sweep_dims[i]: sweeps[i] for i in range(len(sweep_dims))},
            "FRM_time": demod_idx/data_ds.ramp_freq
        }
    )
    return FRM_demod_ds


# =========================
# Trigger/Start Point Detection Functions
# =========================
def trigger_start_triangle(trigger_data, ramp_freq, samp_freq, threshold=(2,1), search_range=1000):
    '''
    Find the start point of the flux ramping data based on the trigger signal.
    The trigger signal is a square wave that is high when flux signal is above 0V and low when flux signal is below 0V.
    Assign the starting point to the first falling edge that has enough points before it to fill in the alpha fraction of the ramp.

    Parameters:
    trigger: xr.Dataset, the trigger signal stored in the 'trigger' variable
    alpha: float, the fraction of the ramp to throw away
    ramp_freq: float, the frequency of the flux ramp
    samp_freq: float, the sampling frequency of the data
    treshold: tuple, the treshold for the trigger signal, (number of points, voltage drop)

    Returns:
    start_pt: xr.Dataset, the starting point of the flux ramping data
    '''

    sweep_dims = [dim for dim in trigger_data.dims if dim not in ("time_trace",)]
    flat_trigger = trigger_data["trigger"].stack(traces=sweep_dims)
    flat_trigger = flat_trigger.transpose("traces", "time_trace").values

    start_idx = np.zeros(flat_trigger.shape[0], dtype=int)

    for i in range(flat_trigger.shape[0]):
        pts_per_ramp = int(np.ceil(samp_freq/ramp_freq))
        trigger_trace = flat_trigger[i, :search_range]

        # Trigger treshold: the voltage drop within 2 points is greater than 1V (2,1)
        # Trigger on the falling edge, but the start point is defined as the point pts_per_ramp*1/4 after the falling edge
        trigger_diff = np.array([trigger_trace[i+threshold[0]] - trigger_trace[i] for i in range(len(trigger_trace)-threshold[0])])
        trigger_idx = np.where(trigger_diff < -threshold[1])[0][0]
        start_idx[i] = trigger_idx + int(pts_per_ramp/4)

    # Reshape start_idx to the original sweep dimensions
    start_idx_xr = xr.Dataset({"start_idx": (sweep_dims, start_idx.reshape([trigger_data.sizes[dim] for dim in sweep_dims]))},
                              coords={dim: trigger_data[dim] for dim in sweep_dims})
    start_idx_xr = start_idx_xr.assign_coords(sweep_freq=('center_freq', np.array(trigger_data["sweep_freq"])))

    return start_idx_xr


def trigger_start_sawtooth(trigger_data, threshold=(2,1), search_range=1000, symmetry=1):
    '''
    Find the start point of the flux ramping data based on the trigger signal.
    The trigger signal is a square wave that is high when flux signal is above 0V and low when flux signal is below 0V.
    Assign the starting point to the first falling edge.

    Parameters:
    trigger: xr.Dataset, the trigger signal stored in the 'trigger' variable
    alpha: float, the fraction of the ramp to throw away
    ramp_freq: float, the frequency of the flux ramp
    samp_freq: float, the sampling frequency of the data
    treshold: tuple, the treshold for the trigger signal, (number of points, voltage drop)

    Returns:
    start_pt: xr.Dataset, the starting point of the flux ramping data
    '''

    sweep_dims = [dim for dim in trigger_data.dims if dim not in ("time_trace",)]
    flat_trigger = trigger_data["trigger"].stack(traces=sweep_dims)
    flat_trigger = flat_trigger.transpose("traces", "time_trace").values

    start_idx = np.zeros(flat_trigger.shape[0], dtype=int)
    pts_per_ramp = int(trigger_data.samp_freq/trigger_data.ramp_freq)
    pad = int((1-symmetry)/2*pts_per_ramp)

    for i in range(flat_trigger.shape[0]):
        trigger_trace = flat_trigger[i, :search_range]

        # Trigger treshold: the voltage drop within 2 points is greater than 1V (2,1)
        # Trigger on the falling edge, but the start point is defined as the point pts_per_ramp*1/4 after the falling edge
        trigger_diff = np.array([trigger_trace[i+threshold[0]] - trigger_trace[i] for i in range(len(trigger_trace)-threshold[0])])
        zero_crossings = np.where(trigger_diff < -threshold[1])[0] - pad
        start_idx[i] = zero_crossings[np.where(zero_crossings > 0)][0]


    # Reshape start_idx to the original sweep dimensions
    start_idx_xr = xr.Dataset({"start_idx": (sweep_dims, start_idx.reshape([trigger_data.sizes[dim] for dim in sweep_dims]))},
                              coords={dim: trigger_data[dim] for dim in sweep_dims})
    start_idx_xr = start_idx_xr.assign_coords(sweep_freq=('center_freq', np.array(trigger_data["sweep_freq"])))

    return start_idx_xr


