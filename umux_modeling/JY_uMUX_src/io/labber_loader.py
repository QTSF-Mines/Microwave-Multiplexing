import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import xarray as xr

from dspx_helpers import helper



def load_IQ_data(h5file, channel_names, sweep_params, fixed_params=None, samp_freq_ch=None, avg=False):
    """
    Load IQ data from a Labber HDF5 file. Adopt the xarray data structure for easy data handling.
    Assume each resonator uses a fixed sweep frequency.
     
    Parameters:
    h5file (str or Path): Path to the HDF5 file.
    channel_names (dict): Dictionary with keys "I", "Q", and "trigger" mapping to their respective channel names.
    sweep_params (dict): Dictionary mapping sweep parameter names to their channel names.

    Returns:
    xr.Dataset: xarray Dataset containing S21 data with time trace dimension.
    """

    f = helper.LabberLog(h5file)
    instr_config = f.instrument_config
    step_pd = f.step_df
    sweep_param_names = [name for name in sweep_params.keys()]
    sweep_param_chs = [sweep_params[name] for name in sweep_param_names]
    sweep_param_values = [np.array(step_pd[ch].unique()) for ch in sweep_param_chs]
    sweep_param_sizes = [len(vals) for vals in sweep_param_values]
    num_sweeps = np.prod(sweep_param_sizes)
    num_res = len(step_pd[sweep_params["center_freq"]].unique())

    if avg == False:
        samp_freq = parse_samp_freq(instr_config[samp_freq_ch[0]][samp_freq_ch[1]])

    if avg:
        len_trace = 1
    else:
        len_trace = len(f.getTrace(log_ch=channel_names["I"], tr_nr=0))

    S21_data = np.zeros((num_sweeps, len_trace), dtype=complex)

    if fixed_params is not None:
        fixed_params_names = [name for name in fixed_params.keys()]
        fixed_params_chs = [fixed_params[name] for name in fixed_params_names]
        fixed_params_values = [np.zeros(num_res) for _ in fixed_params_names]

    for i in tqdm(range(num_sweeps)):
        tr_nr = np.argmax(np.all([step_pd[ch] == vals[idx] for ch, vals, idx in zip(sweep_param_chs, sweep_param_values, np.unravel_index(i, sweep_param_sizes))], axis=0))
        if avg:
            S21_data[i] = np.average(f.getTrace(log_ch=channel_names['I'], tr_nr=tr_nr)) + 1j * np.average(f.getTrace(log_ch=channel_names['Q'], tr_nr=tr_nr))
        else:   
            S21_data[i] = f.getTrace(log_ch=channel_names['I'], tr_nr=tr_nr) + 1j * f.getTrace(log_ch=channel_names['Q'], tr_nr=tr_nr)
        res_idx = np.where(step_pd[sweep_params["center_freq"]].unique() == step_pd[sweep_params["center_freq"]].iloc[tr_nr])[0][0]

        if fixed_params is not None:
            for j, ch in enumerate(fixed_params_chs):
                fixed_params_values[j][res_idx] = step_pd[ch].iloc[tr_nr]

    # Put I and Q data into one xarray DataArray with two variables, with the last dimension being time trace (if not averaged)
    data_ds = xr.Dataset(
        {
            "S21": (sweep_param_names + (["time_trace"] if not avg else []), S21_data.reshape(sweep_param_sizes + ([len_trace] if not avg else []))),
        },
        coords={name: vals for name, vals in zip(sweep_param_names, sweep_param_values)} | ({"time_trace": np.linspace(0, len_trace/samp_freq, len_trace)} if not avg else {})
    )

    if fixed_params is not None:
        for name, vals in zip(fixed_params_names, fixed_params_values):
            data_ds = data_ds.assign_coords({name: ("center_freq", vals)})

    return data_ds

def parse_samp_freq(samp_freq_str):
    """
    Parse the sampling frequency string from Labber instrument configuration to a float value in Hz.
    Example input: '1 kS/s' -> output: 1000.0, '2 MS/s' -> output: 2000000.0
    """
    units = {'kS/s': 1e3, 'MS/s': 1e6, 'GS/s': 1e9}
    for unit, factor in units.items():
        if unit in samp_freq_str:
            return float(samp_freq_str.replace(unit, '').strip()) * factor
    raise ValueError(f"Unknown sampling frequency unit in string: {samp_freq_str}")


def load_FRM_data(h5file, channel_names, sweep_params, locked_params, fixed_params):
    """
    Load IQ data from a Labber HDF5 file. Adopt the xarray data structure for easy data handling.
    Assume each resonator uses a fixed sweep frequency.
     
    Parameters:
    h5file (str or Path): Path to the HDF5 file.
    channel_names (dict): Dictionary with keys "I", "Q", and "trigger" mapping to their respective channel names.
    sweep_params (dict): Dictionary mapping sweep parameter names to their channel names.
    samp_freq_chs (dict): Dictionary mapping "DAQ_instr", "AWG_instr", "samp_freq_ch", and "ramp_freq_ch" to their respective channel names.

    Returns:
    xr.Dataset: xarray Dataset containing S21 data with time trace dimension.
    """

    f = helper.LabberLog(h5file)
    instr_config = f.instrument_config
    step_pd = f.step_df
    sweep_param_names = [name for name in sweep_params.keys()]
    sweep_param_chs = [sweep_params[name] for name in sweep_param_names]
    sweep_param_values = [np.array(step_pd[ch].unique()) for ch in sweep_param_chs]
    sweep_param_sizes = [len(vals) for vals in sweep_param_values]
    num_sweeps = np.prod(sweep_param_sizes)
    num_res = len(step_pd[sweep_params["center_freq"]].unique())

    locked_params_names = [name for name in locked_params.keys()]
    locked_params_chs = [locked_params[name] for name in locked_params_names]
    locked_params_values = [np.zeros(num_res) for _ in locked_params_names]

    fixed_params_names = [name for name in fixed_params.keys()]
    fixed_params_values = np.zeros(len(fixed_params_names))

    for i, ch in enumerate(fixed_params.values()):
        if fixed_params_names[i] == "samp_freq":
            fixed_params_values[i] = parse_samp_freq(instr_config[ch[0]][ch[1]])
        else:
            fixed_params_values[i] = instr_config[ch[0]][ch[1]]

    len_trace = len(f.getTrace(log_ch=channel_names["I"], tr_nr=0))

    S21_data = np.zeros((num_sweeps, len_trace), dtype=complex)
    trigger_data = np.zeros((num_sweeps, len_trace))

    for i in tqdm(range(num_sweeps)):
        tr_nr = np.argmax(np.all([step_pd[ch] == vals[idx] for ch, vals, idx in zip(sweep_param_chs, sweep_param_values, np.unravel_index(i, sweep_param_sizes))], axis=0))

        S21_data[i] = f.getTrace(log_ch=channel_names['I'], tr_nr=tr_nr) + 1j * f.getTrace(log_ch=channel_names['Q'], tr_nr=tr_nr)
        trigger_data[i] = f.getTrace(log_ch=channel_names['trigger'], tr_nr=tr_nr)
        res_idx = np.where(step_pd[sweep_params["center_freq"]].unique() == step_pd[sweep_params["center_freq"]].iloc[tr_nr])[0][0]
        for j, ch in enumerate(locked_params_chs):
            locked_params_values[j][res_idx] = step_pd[ch].iloc[tr_nr]
    
    samp_freq = fixed_params_values[fixed_params_names.index("samp_freq")]

    # Put I and Q data into one xarray DataArray with two variables, with the last dimension being time trace (if not averaged)
    data_ds = xr.Dataset(
        {
            "S21": (sweep_param_names + ["time_trace"], S21_data.reshape(sweep_param_sizes + [len_trace])),
        },
        coords={name: vals for name, vals in zip(sweep_param_names, sweep_param_values)} | {"time_trace": np.linspace(0, len_trace/samp_freq, len_trace)}  # Time axis in seconds
    )
    trigger_ds = xr.Dataset(
        {
            "trigger": (sweep_param_names + ["time_trace"], trigger_data.reshape(sweep_param_sizes + [len_trace])),
        },
        coords={name: vals for name, vals in zip(sweep_param_names, sweep_param_values)} | {"time_trace": np.linspace(0, len_trace/samp_freq, len_trace)}
    )
    

    for name, vals in zip(locked_params_names, locked_params_values):
        data_ds = data_ds.assign_coords({name: ("center_freq", vals)})
        trigger_ds = trigger_ds.assign_coords({name: ("center_freq", vals)})

    for i, name in enumerate(fixed_params_names):
        data_ds = data_ds.assign_attrs({name: fixed_params_values[i]})
        trigger_ds = trigger_ds.assign_attrs({name: fixed_params_values[i]})

    return data_ds, trigger_ds 




def load_VNA_data(h5file, S21_ch, sweep_params):
    """
    Load VNA S21 data from a Labber HDF5 file. Adopt the xarray data structure for easy data handling.
     
    Parameters:
    h5file (str or Path): Path to the HDF5 file.
    S21_ch (str): Name of the S21 data channel.
    sweep_params (dict): Dictionary mapping sweep parameter names to their channel names.

    Returns:
    tuple: A tuple containing:
        - S21_data (xr.DataArray): xarray DataArray of S21 data after averaging over time trace.
    """

    f = helper.LabberLog(h5file)
    step_pd = f.step_df
    sweep_param_names = list(sweep_params.keys())
    sweep_param_chs = list(sweep_params.values())
    sweep_param_values = [np.array(step_pd[ch].unique()) for ch in sweep_param_chs]
    sweep_param_sizes = [len(vals) for vals in sweep_param_values]
    num_sweeps = np.prod(sweep_param_sizes)
    len_trace = len(f.getTrace(log_ch=S21_ch, tr_nr=0))

    freq_axis = f.getXaxis(log_ch=S21_ch, tr_nr=0)
    sweep_freqs = freq_axis - freq_axis[0] - (freq_axis[-1]-freq_axis[0])/2  # Centered sweep frequencies


    S21_data = np.zeros((num_sweeps, len_trace), dtype=complex)

    for i in tqdm(range(num_sweeps)):
        tr_nr = np.argmax(np.all([step_pd[ch] == vals[idx] for ch, vals, idx in zip(sweep_param_chs, sweep_param_values, np.unravel_index(i, sweep_param_sizes))], axis=0))
        S21_data[i, :] = f.getTrace(log_ch=S21_ch, tr_nr=tr_nr)

    # Create the xarray Data with the last dimension being sweep frequency

    S21_data_xr = xr.Dataset(
        {
            "S21": (sweep_param_names + ["sweep_freq"], S21_data.reshape(sweep_param_sizes + [len_trace])),
        },
        coords={name: vals for name, vals in zip(sweep_param_names, sweep_param_values)} | {"sweep_freq": sweep_freqs} 
    )

    return S21_data_xr