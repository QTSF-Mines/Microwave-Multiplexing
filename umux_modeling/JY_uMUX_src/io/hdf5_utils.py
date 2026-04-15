import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import xarray as xr

from dspx_helpers import helper

def print_nested_dict(d, indent=0):
    for key, value in d.items():
        print('    ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print()
            print_nested_dict(value, indent + 1)
        else:
            print(value)
        


def hdf5_print_instr_config(h5file):
    """
    List all the step channels, log channels, and instrument configuration in a Labber HDF5 file.
    """
    f = helper.LabberLog(h5file)
    instr_config = f.instrument_config

    print_nested_dict(instr_config)

    return

def hdf5_list_channels(h5file):
    """
    List all the step channels and log channels in a Labber HDF5 file.
    """
    f = helper.LabberLog(h5file)
    step_pd = f.step_df
    log_chs = f.getLogChannels()

    print("Step channels:")
    for ch in step_pd.columns:
        print(f" - {ch}")

    print("\nLog channels:")
    for ch in log_chs:
        print(f" - {ch}")

    return