import numpy as np
from scipy import signal
import pandas as pd

from iminuit import Minuit
from iminuit.cost import LeastSquares
import iminuit

def welch_PSD(data, fs, freq_cut, freq_resol, window='boxcar', scaling='density', axis=-1, detrend=False):
    """
    Compute the Power Spectral Density (PSD) using Welch's method.

    Parameters:
    - data: 1D array-like, the input signal
    - fs: float, the sampling frequency of the signal
    - freq_cut: list of floats, the frequency cutoffs for the PSD estimation
    - freq_resol: list of floats, the frequency resolutions for each segment
    - window: str or array_like, the window to apply to each segment
    - scaling: str, the scaling of the PSD
    - axis: int, the axis along which to compute the PSD
    - detrend: str or function, the detrending method

    Returns:
    - freqs: 1D array, the frequencies corresponding to the PSD
    - psd: 1D array, the estimated PSD of the signal
    """

    if freq_cut[0] < int(np.round(fs / data.shape[axis])):
        print(fs / data.shape[axis])
        raise ValueError("The smallest element of freq_cut must be greater than or equal to the baseline frequency resolution.")
    
    else:
        freq_axis = np.empty((0))
        Sxx = np.empty(data.shape[:-1] + (0,))

        for i in range(len(freq_resol)):
            f, Sxx_crt = signal.welch(data, fs=fs, nperseg=int(fs/freq_resol[i]), window=window, scaling=scaling, axis=axis, detrend=detrend)
            mask = (f>=freq_cut[i]) & (f<freq_cut[i+1])
            freq_axis = np.concatenate((freq_axis, np.arange(freq_cut[i], freq_cut[i+1], freq_resol[i])))
            Sxx = np.concatenate((Sxx, np.take(Sxx_crt, np.where(mask)[0], axis=-1)), axis=-1)

    return freq_axis, Sxx

def welch_CSD(data1, data2, fs, freq_cut, freq_resol, window='boxcar', scaling='density', axis=-1, detrend=False):
    """
    Compute the cross Power Spectral Density (CSD) using Welch's method.

    Parameters:
    - data1: 1D array-like, the first input signal
    - data2: 1D array-like, the second input signal
    - fs: float, the sampling frequency of the signals
    - freq_cut: list of floats, the frequency cutoffs for the CSD estimation
    - freq_resol: list of floats, the frequency resolutions for each segment
    - window: str or array_like, the window to apply to each segment
    - scaling: str, the scaling of the CSD
    - axis: int, the axis along which to compute the CSD
    - detrend: str or function, the detrending method

    Returns:
    - freqs: 1D array, the frequencies corresponding to the CSD
    - csd: 1D array, the estimated CSD of the signals
    """

    if freq_cut[0] < fs / data1.shape[axis]:
        raise ValueError("The smallest element of freq_cut must be greater than or equal to the baseline frequency resolution.")
    else:
        freq_axis = np.empty((0))
        Sxy = np.empty(data1.shape[:-1] + (0,))

        for i in range(len(freq_resol)):
            f, Sxy_crt = signal.csd(data1, data2, fs=fs, nperseg=int(fs/freq_resol[i]), window=window, scaling=scaling, axis=axis, detrend=detrend)
            mask = (f>=freq_cut[i]) & (f<freq_cut[i+1])
            freq_axis = np.concatenate((freq_axis, np.arange(freq_cut[i], freq_cut[i+1], freq_resol[i])))
            Sxy = np.concatenate((Sxy, np.take(Sxy_crt, np.where(mask)[0], axis=-1)), axis=-1)

    return freq_axis, Sxy


def noise_model(f, S_w, S_f, n):
    return np.sqrt(S_w + f**(-n) * S_f)


def remove_noise_peaks(freq_axis, noisy_data, continuum, threshold, noise_check, window=2):
    """
    Remove noise peaks from the noisy data based on the continuum and a threshold.

    Parameters:
    -----------
    freq_axis : 1D array
        The frequency axis for the noisy data.
    noisy_data : 1D array
        The noisy data from which to remove peaks.
    continuum : 1D array
        The continuum data to subtract from the noisy data.
    threshold : float
        The threshold above which to consider a point as a noise peak.
    noise_check : 1D array
        The frequencies at which to check for noise peaks.
    window : int
        The number of points to remove on each side of a detected noise peak.       
    
    Returns:
    -----------
    freq_axis_clean : 1D array
        The frequency axis after removing noise peaks.
    clean_data : 1D array
        The noisy data after removing noise peaks.
    """



    # Subtract the continuum from the noisy data
    residual = noisy_data - continuum

    # At each index in check_idx, check if the residual is above the threshold
    check_idx = np.argmin(np.abs(freq_axis[:, None] - noise_check[None, :]), axis=0)
    noise_idx = check_idx[(residual[check_idx] > threshold)]

    # Remove the data points within the window +-2 points at those indices
    remove_idx = []
    for idx in noise_idx:
        remove_idx.extend(range(max(0, idx-window), min(len(noisy_data), idx+window)))

    clean_data = np.delete(noisy_data, remove_idx)
    freq_axis_clean = np.delete(freq_axis, remove_idx)

    return freq_axis_clean, clean_data


def fit_noise_model_bulk(freq_axis, noise, noise_err, freq_cut, fixed_params={}):

    """
    Fit the noise model to multiple noise spectra and return the fitted parameters in a dataframe.

    Parameters:
    -----------
    freq_axis : list of 1D arrays
        The frequency axis for the noise spectra, each of shape (num_freqs,).
    noise : 2D array
        The noise spectra to fit, shape (num_res, num_freqs).
    noise_err : 2D array
        The errors on the noise spectra, shape (num_res, num_freqs).
    freq_cut : 1D array
        The frequency cut-offs for the fit, shape (num_freqs,).

    Returns:
    -----------
    results_df : pandas DataFrame
        A dataframe containing the fitted parameters S_w, S_f, and n for each noise spectrum
    """

    num_res = len(noise)
    
    S_w = np.zeros(num_res)
    S_f = np.zeros(num_res)
    n = np.zeros(num_res)

    if noise_err is None:

        for i in range(num_res):
            freq_axis_cut = freq_axis[i][(freq_axis[i]>=freq_cut[i][0]) & (freq_axis[i]<=freq_cut[i][1])]
            noise_cut = noise[i][(freq_axis[i]>=freq_cut[i][0]) & (freq_axis[i]<=freq_cut[i][1])]

            def nll(sigma, S_w, S_f, n):
                if np.any(sigma <= 0):
                    return np.inf
                r = noise_cut - noise_model(freq_axis_cut, S_w, S_f, n)
                n = len(noise_cut)
                return np.sum(r**2) / sigma**2 + n * np.log(sigma**2)
            
            # Set up the minimizer
            S_w_guess = np.min(noise_cut)**2
            S_f_guess = noise_cut[0]**2
            params_guess = {'S_w': S_w_guess,
                            'S_f': S_f_guess,
                            'n': 1,
                            'sigma': np.std(noise_cut)}
            
            for param, value in fixed_params.items():
                if param in params_guess:
                    params_guess[param] = value[i]
            
            m = Minuit(nll, **params_guess)
            
            for param, value in fixed_params.items():
                m.fixed[param] = True

            # Add limits and initial values
            m.limits['S_w'] = (0, None)
            m.limits['S_f'] = (0, None)
            m.limits['n'] = (0, 5)
            m.limits['sigma'] = (0, None)

            # Perform the fit
            m.migrad()

            # Extract the fitted parameters and store them in a pdataframe
            S_w[i] = m.values['S_w']
            S_f[i] = m.values['S_f']
            n[i] = m.values['n']

            results_df = pd.DataFrame({'S_w': np.sqrt(S_w), 'S_f': np.sqrt(S_f), 'n': n})
            results_err = pd.DataFrame({'S_w_err': m.errors['S_w']/2/np.sqrt(S_w), 'S_f_err': m.errors['S_f']/2/np.sqrt(S_f), 'n_err': m.errors['n']})
        
    else:

        for i in range(num_res):
            freq_axis_cut = freq_axis[i][(freq_axis[i]>=freq_cut[i][0]) & (freq_axis[i]<=freq_cut[i][1])]
            noise_cut = noise[i][(freq_axis[i]>=freq_cut[i][0]) & (freq_axis[i]<=freq_cut[i][1])]
            noise_err_cut = noise_err[i][(freq_axis[i]>=freq_cut[i][0]) & (freq_axis[i]<=freq_cut[i][1])]

            cost = LeastSquares(freq_axis_cut, noise_cut, noise_err_cut, noise_model)
            
            # Set up the minimizer
            S_w_guess = np.min(noise_cut)**2
            S_f_guess = noise_cut[0]**2
            params_guess = {'S_w': S_w_guess,
                            'S_f': S_f_guess,
                            'n': 1}
            
            for param, value in fixed_params.items():
                if param in params_guess:
                    params_guess[param] = value[i]
            
            m = Minuit(cost, **params_guess)
            
            for param, value in fixed_params.items():
                m.fixed[param] = True

            # Add limits and initial values
            m.limits['S_w'] = (0, None)
            m.limits['S_f'] = (0, None)
            m.limits['n'] = (0, 5)

            # Perform the fit
            m.migrad()

            # Extract the fitted parameters and store them in a pdataframe
            S_w[i] = m.values['S_w']
            S_f[i] = m.values['S_f']
            n[i] = m.values['n']

            results_df = pd.DataFrame({'S_w': np.sqrt(S_w), 'S_f': np.sqrt(S_f), 'n': n})
            results_err = pd.DataFrame({'S_w_err': m.errors['S_w']/2/np.sqrt(S_w), 'S_f_err': m.errors['S_f']/2/np.sqrt(S_f), 'n_err': m.errors['n']})

    return results_df, results_err