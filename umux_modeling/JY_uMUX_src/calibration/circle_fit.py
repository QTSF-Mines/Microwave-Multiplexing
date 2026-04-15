import numpy as np
from tqdm import tqdm
from si_prefix import si_format
from resonator_tools import circuit
from scipy.optimize import curve_fit
import xarray as xr


from dspx_helpers import resonatorhelper

def delay_fit(freq, trace):
    " ************* Under development *********"
    def model_func(f, delay, amp, alpha, phi0):
        return amp * np.exp(1j*(2*np.pi*f*delay + alpha)) + phi0

    f0 = freq[np.argmin(np.abs(trace))]
    initial_guess = [10e-9, np.max(np.abs(trace)), 0, 0]
    popt, pcov = curve_fit(lambda f, delay, amp, alpha, phi0: model_func(f, delay, amp, alpha, phi0).view(np.float64), freq, trace.view(np.float64), p0=initial_guess)
    delay_fit = popt[0]
    amp_fit = popt[1]
    alpha_fit = popt[2]
    phi0_fit = popt[3]

    return delay_fit, amp_fit, alpha_fit, phi0_fit


def circle_fit_trace(freq_axis, trace, electric_delay='auto', plot=False, print_results=False):
    "Do a circle fit to the complex S21 data on resonance and return the resonant frequencies and Q factors"
    fr_guess = freq_axis[np.argmin(20*np.log10(np.abs(trace)))]
    port = circuit.notch_port()
    port.add_data(freq_axis,trace)
    if electric_delay == 'auto':
        electric_delay = port.get_delay(freq_axis,trace)[0]
    else:
        electric_delay = electric_delay

    port.autofit(fr_guess=fr_guess, electric_delay=electric_delay, fcrop=None)

    results = port.fitresults
    if plot == True:
        fig, ax = resonatorhelper.plotall(port,'fit: {}Hz'.format(si_format(results['fr'],precision=6)), 'resonator')
    if print_results == True:
        print(results)

        
    results_out = [results['fr'], results['Ql'], results['Qc_dia_corr'], results['Qi_dia_corr']]
    errors_out = [results['fr_err'], results['Ql_err'], results['absQc_err'], results['Qi_dia_corr_err']]

    data_sim = port.z_data_sim
    data_sim_norm = port.z_data_sim_norm

    delay, amp_norm, alpha, fr, Ql, A2, frcal =\
				port.do_calibration(port.f_data[port._fid],port.z_data_raw[port._fid],ignoreslope=True,guessdelay=False,fixed_delay=electric_delay,Ql_guess=results['Ql'], fr_guess=results['fr'])

    calibration_results = [delay, amp_norm, alpha, results['phi0']]

    return results_out, errors_out, calibration_results


def circle_fit_bulk(S21_data, electric_delay='auto'):
    """
    Perform circle fitting on bulk S21 data for multiple resonators. No plotting nor printing of results.
    Assuming the name of the S21 data variable is "S21" in the xarray Dataset.
    Assuming the name of the center frequency dimension is given by "center_freq".
    Assuming the name of the sweep frequency dimension is given by "sweep_freq" (default).
    
    Parameters:
    S21_data: xr.Dataset
        xarray Dataset containing S21 data and dims (..., frequency), where ... are any number of dimensions representing different resonators, flux bias, etc.
    electric_delay: xr.Dataset or 'auto'
        Electric delay to use for each resonator. If 'auto', the delay will be fitted for each resonator individually.
    """

    if "S21" not in S21_data:
        raise ValueError("S21_data must contain a variable named 'S21'")

    # Flatten all sweep dimensions except frequency
    sweep_dims = [dim for dim in S21_data.dims if dim not in ("sweep_freq",)]
    flat_s21 = S21_data["S21"].stack(traces=sweep_dims)
    flat_s21 = flat_s21.transpose("traces", "sweep_freq").values  # shape (num_traces, num_freqs)

    # Prepare dictionaries to hold results
    fit_results = {
        "fr": np.zeros(flat_s21.shape[0]),
        "Ql": np.zeros(flat_s21.shape[0]),
        "Qc": np.zeros(flat_s21.shape[0]),
        "Qi": np.zeros(flat_s21.shape[0])
    }
    fit_errors = {
        "fr_err": np.zeros(flat_s21.shape[0]),
        "Ql_err": np.zeros(flat_s21.shape[0]),
        "Qc_err": np.zeros(flat_s21.shape[0]),
        "Qi_err": np.zeros(flat_s21.shape[0])
    }
    calib_params = {
        "delay": np.zeros(flat_s21.shape[0]),
        "amp_norm": np.zeros(flat_s21.shape[0]),
        "alpha": np.zeros(flat_s21.shape[0]),
        "phi0": np.zeros(flat_s21.shape[0])
    }

    for i in tqdm(range(flat_s21.shape[0])):
        trace = flat_s21[i, :]

        # identify the center frequency and sweep frequency for this trace
        center_freq = S21_data[sweep_dims].isel({dim: i // np.prod([S21_data.sizes[d] for d in sweep_dims if d != dim]) % S21_data.sizes[dim] for dim in sweep_dims}).center_freq.values
        # extract the index of the resonator from center_freq
        resonator_index = np.argmin(np.abs(S21_data.center_freq.values - center_freq))
        sweep_freqs = S21_data["sweep_freq"].values
        freqs = center_freq + sweep_freqs
        
        if np.any(electric_delay == 'auto'):
            electric_delay_fit = 'auto'
        else:
            electric_delay_fit = electric_delay[resonator_index]

        results_out, errors_out, calib_out = circle_fit_trace(freqs, trace, electric_delay=electric_delay_fit, plot=False, print_results=False)
        fit_results["fr"][i], fit_results["Ql"][i], fit_results["Qc"][i], fit_results["Qi"][i] = results_out
        fit_errors["fr_err"][i], fit_errors["Ql_err"][i], fit_errors["Qc_err"][i], fit_errors["Qi_err"][i] = errors_out
        calib_params["delay"][i], calib_params["amp_norm"][i], calib_params["alpha"][i], calib_params["phi0"][i] = calib_out

    # Reshape results back to original sweep dimensions
    fit_results_ds = xr.Dataset({key: (sweep_dims, fit_results[key].reshape([S21_data.sizes[dim] for dim in sweep_dims])) for key in fit_results},
                                coords={dim: S21_data[dim] for dim in sweep_dims})
    fit_errors_ds = xr.Dataset({key: (sweep_dims, fit_errors[key].reshape([S21_data.sizes[dim] for dim in sweep_dims])) for key in fit_errors},
                                coords={dim: S21_data[dim] for dim in sweep_dims})
    calib_params_ds = xr.Dataset({key: (sweep_dims, calib_params[key].reshape([S21_data.sizes[dim] for dim in sweep_dims])) for key in calib_params},
                                coords={dim: S21_data[dim] for dim in sweep_dims})

    return fit_results_ds, fit_errors_ds, calib_params_ds


def do_normalization_trace(S21_data, calib_params, ):
    return



def do_normalization_bulk(S21_data, calib_params, vol_bias_sel=0, sweep_freq=True, avg_trace=True):

    """
    Perform normalization on bulk S21 data for multiple resonators using provided calibration parameters.
    Assuming the name of the S21 data variable is "S21" in the xarray.
    Assuming the name of the center frequency dimension is given by "center_freq".
    Assuming the name of the sweep frequency dimension is given by "sweep_freq" (default).
    Assuming the calibration parameters are provided in an array with shape (num_resonators, 4) corresponding to (delay, amp_norm, alpha, phi0).

    Parameters:
    S21_data: xr.Dataset
        xarray Dataset containing S21 data and dims (..., frequency), where ... are any number of dimensions representing different resonators, flux bias, etc.
    calib_params: np.ndarray
        Array of shape (num_resonators, 4) containing calibration parameters for each resonator.

    Returns:
    S21_data_normalized: xr.Dataset
        xarray Dataset containing normalized S21 data with the same dims as input.
    """
         
    if "S21" not in S21_data:
        raise ValueError("S21_data must contain a variable named 'S21'")
    

    if sweep_freq:

        if avg_trace==True:
            # Flatten all sweep dimensions except frequency
            sweep_dims = [dim for dim in S21_data.dims if dim not in ("sweep_freq", "vol_bias")]
            flat_s21 = S21_data["S21"].stack(traces=sweep_dims)
            if "vol_bias" in S21_data.dims:
                flat_s21 = flat_s21.transpose("traces", "vol_bias", "sweep_freq").values  # shape (num_traces, num_vol_bias, num_freqs)
            else:
                flat_s21 = flat_s21.transpose("traces", "sweep_freq").values  # shape (num_traces, num_freqs)

            S21_normalized = np.zeros_like(flat_s21, dtype=complex)

            for i in tqdm(range(flat_s21.shape[0])):
                trace = flat_s21[i]

                # Unravel the index in sweep_dims from i
                unravel_idx = np.zeros(len(sweep_dims), dtype=int)
                for j, dim in enumerate(sweep_dims):
                    unravel_idx[j] = i // np.prod([S21_data.sizes[d] for d in sweep_dims if d != dim]) % S21_data.sizes[dim]
                # Build a dictionary to select the calibration parameters
                sel_dict = {dim: S21_data[dim].values[unravel_idx[j]] for j, dim in enumerate(sweep_dims)}
                sel_dict["vol_bias"] = vol_bias_sel  # Select a specific vol_bias
                delay = calib_params.sel(**sel_dict)["delay"].values
                amp_norm = calib_params.sel(**sel_dict)["amp_norm"].values
                alpha = calib_params.sel(**sel_dict)["alpha"].values
                phi0 = calib_params.sel(**sel_dict)["phi0"].values

                # Perform normalization
                freqs = sel_dict['center_freq'] + S21_data["sweep_freq"].values
                S21_normalized[i] = 1-(1-trace/amp_norm*np.exp(1j*(-alpha+2.*np.pi*delay*freqs)))*np.exp(-1j*phi0)

            if "vol_bias" in S21_data.dims:
                S21_normalized = xr.Dataset({"S21_norm": (sweep_dims + ["vol_bias", "sweep_freq"], S21_normalized.reshape([S21_data.sizes[dim] for dim in sweep_dims] + [S21_data.sizes["vol_bias"], S21_data.sizes["sweep_freq"]]))},
                                            coords={dim: S21_data[dim] for dim in sweep_dims} | {"vol_bias": S21_data["vol_bias"], "sweep_freq": S21_data["sweep_freq"]})
            else:
                S21_normalized = xr.Dataset({"S21_norm": (sweep_dims + ["sweep_freq"], S21_normalized.reshape([S21_data.sizes[dim] for dim in sweep_dims] + [S21_data.sizes["sweep_freq"]]))},
                                            coords={dim: S21_data[dim] for dim in sweep_dims} | {"sweep_freq": S21_data["sweep_freq"]})

        else:
            # Flatten all sweep dimensions except frequency
            sweep_dims = [dim for dim in S21_data.dims if dim not in ("sweep_freq", "vol_bias", "time_trace")]
            flat_s21 = S21_data["S21"].stack(traces=sweep_dims)
            if "vol_bias" in S21_data.dims:
                flat_s21 = flat_s21.transpose("traces", "vol_bias", "sweep_freq", "time_trace").values  # shape (num_traces, num_vol_bias, num_freqs)
            else:
                flat_s21 = flat_s21.transpose("traces", "sweep_freq", "time_trace").values  # shape (num_traces, num_freqs)

            S21_normalized = np.zeros_like(flat_s21, dtype=complex)

            for i in tqdm(range(flat_s21.shape[0])):
                trace = flat_s21[i]

                # Unravel the index in sweep_dims from i
                unravel_idx = np.zeros(len(sweep_dims), dtype=int)
                for j, dim in enumerate(sweep_dims):
                    unravel_idx[j] = i // np.prod([S21_data.sizes[d] for d in sweep_dims if d != dim]) % S21_data.sizes[dim]

                
                # Build a dictionary to select the calibration parameters
                sel_dict = {dim: S21_data[dim].values[unravel_idx[j]] for j, dim in enumerate(sweep_dims)}
                sel_dict["vol_bias"] = vol_bias_sel  # Select a specific vol_bias
                delay = calib_params.sel(**sel_dict)["delay"].values
                amp_norm = calib_params.sel(**sel_dict)["amp_norm"].values
                alpha = calib_params.sel(**sel_dict)["alpha"].values
                phi0 = calib_params.sel(**sel_dict)["phi0"].values


                # Perform normalization
                freqs = sel_dict['center_freq'] + S21_data["sweep_freq"].values
                freqs_tile = np.tile(freqs[np.newaxis, :, np.newaxis], (len(S21_data.vol_bias),1, len(S21_data.time_trace)))
                S21_normalized[i] = 1-(1-trace/amp_norm*np.exp(1j*(-alpha+2.*np.pi*delay*freqs_tile)))*np.exp(-1j*phi0)
                

            if "vol_bias" in S21_data.dims:
                S21_normalized = xr.Dataset({"S21_norm": (sweep_dims + ["vol_bias", "sweep_freq", "time_trace"], S21_normalized.reshape([S21_data.sizes[dim] for dim in sweep_dims] + [S21_data.sizes["vol_bias"], S21_data.sizes["sweep_freq"], S21_data.sizes["time_trace"]]))},
                                            coords={dim: S21_data[dim] for dim in sweep_dims} | {"vol_bias": S21_data["vol_bias"], "sweep_freq": S21_data["sweep_freq"], "time_trace": S21_data["time_trace"]})
            else:
                S21_normalized = xr.Dataset({"S21_norm": (sweep_dims + ["sweep_freq", "time_trace"], S21_normalized.reshape([S21_data.sizes[dim] for dim in sweep_dims] + [S21_data.sizes["sweep_freq"], S21_data.sizes["time_trace"]]))},
                                            coords={dim: S21_data[dim] for dim in sweep_dims} | {"sweep_freq": S21_data["sweep_freq"], "time_trace": S21_data["time_trace"]})


    
    
    else:
        sweep_dims = [dim for dim in S21_data.dims if dim not in ("center_freq",)]
        flat_s21 = S21_data["S21"].stack(traces=sweep_dims)
        flat_s21 = flat_s21.transpose("center_freq", "traces").values

        S21_normalized = np.zeros_like(flat_s21, dtype=complex)
       

        for i in tqdm(range(flat_s21.shape[0])):
            freq = S21_data.center_freq.values[i] + S21_data.sweep_freq.values[i]
            sel_dict = {"center_freq": S21_data.center_freq.values[i],
                        "vol_bias": vol_bias_sel}
            delay = calib_params.sel(**sel_dict)["delay"].values
            amp_norm = calib_params.sel(**sel_dict)["amp_norm"].values
            alpha = calib_params.sel(**sel_dict)["alpha"].values
            phi0 = calib_params.sel(**sel_dict)["phi0"].values

            # Perform normalization
            S21_normalized[i] = 1-(1-flat_s21[i]/amp_norm*np.exp(1j*(-alpha+2.*np.pi*delay*freq)))*np.exp(-1j*phi0)
        
        S21_normalized = xr.Dataset({"S21_norm": (["center_freq"] + sweep_dims, S21_normalized.reshape([S21_data.sizes["center_freq"]] + [S21_data.sizes[dim] for dim in sweep_dims]))},
                                    coords={"center_freq": S21_data["center_freq"]} | {dim: S21_data[dim] for dim in sweep_dims})

    # Copy the attrs from S21_data to S21_normalized
    S21_normalized.attrs = S21_data.attrs
    
    return S21_normalized


def S21_calc(fit_results, sweep_freqs):
    """
    Calculate the theoretical S21 response based on fit results and frequency axis.
    
    Parameters:
    fit_results: xr.Dataset
        xarray Dataset containing fit results with keys 'fr', 'Ql', 'Qc', 'Qi'. They also contain a dimension
        of center_freq if multiple resonators are fitted together.
    freqs: np.ndarray
        Array of frequency values at which to calculate S21.
    
    Returns:
    S21_theoretical: xr.Dataset
        Array of calculated S21 values.
    """

    sweep_dims = [dim for dim in fit_results.dims if dim not in ("center_freq",)]
    flat_fr = fit_results['fr'].stack(traces=sweep_dims)
    flat_Ql = fit_results['Ql'].stack(traces=sweep_dims)
    flat_Qc = fit_results['Qc'].stack(traces=sweep_dims)
    flat_fit_results = xr.concat([flat_fr, flat_Ql, flat_Qc], dim="param").values  # shape (num_traces, num_resonators)
    print(flat_fit_results.shape)

    S21_theoretical = np.zeros((flat_fit_results.shape[1], flat_fit_results.shape[2], len(sweep_freqs)), dtype=complex)

    for i in range(flat_fit_results.shape[1]):
        for j in range(flat_fit_results.shape[2]):
            freqs = fit_results.center_freq.values[i] + sweep_freqs
            fr = flat_fit_results[0, i, j]
            Ql = flat_fit_results[1, i, j]
            Qc = flat_fit_results[2, i, j]
            S21_theoretical[i, j, :] = 1 - (Ql/Qc) / (1 + 2j*Ql*(freqs - fr)/fr)

    S21_theoretical = xr.Dataset({"S21_theoretical": (["center_freq"] + sweep_dims + ["sweep_freq"], S21_theoretical)},
                                coords={"center_freq": fit_results["center_freq"], **{dim: fit_results[dim] for dim in sweep_dims}, "sweep_freq": sweep_freqs})
    return S21_theoretical

