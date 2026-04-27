import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.signal import butter, filtfilt
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.optimize import root_scalar
from umux_modeling.LW_uMUX_src.source.include.components import *

class Flux_Ramp:
    def __init__(self, nPhi0, freq, time_array):
        self.n = nPhi0
        self.f = freq
        self.t = time_array
        
        T_ramp = 1 / self.f
        PHI_0 = 2.067e-15
        phi_amplitude = self.n * PHI_0
        self.phi_ramp = (phi_amplitude / T_ramp) * (self.t % T_ramp)
  
    

class Pulse_Generator:
    def __init__(self, t_array, tau_rise, tau_fall, amplitude, offset=0):
        # Prevent division by zero and log errors
        self.tau_r = max(tau_rise, 1e-12)
        self.tau_f = tau_fall
        if self.tau_r == self.tau_f:
            self.tau_f += 1e-12
            
        self.amp = amplitude
        self.t_array = t_array
        self.offset = offset
        
        # 1. Initialize output
        output = np.zeros_like(self.t_array)
        
        # 2. Define the mask for time >= offset
        mask = self.t_array >= self.offset
        
        # 3. Calculate the Peak Normalization (independent of the time array)
        t_peak = (self.tau_r * self.tau_f / (self.tau_f - self.tau_r)) * np.log(self.tau_f / self.tau_r)
        norm = np.exp(-t_peak / self.tau_f) - np.exp(-t_peak / self.tau_r)
        
        # 4. Calculate pulse only for masked indices to avoid index mismatch
        t_masked = self.t_array[mask] - self.offset
        pulse_shape = np.exp(-t_masked / self.tau_f) - np.exp(-t_masked / self.tau_r)
        
        # 5. Assign values
        output[mask] = self.amp * (pulse_shape / norm)
        self.I_pulse = output
        

class Demodulator:
    def __init__(self, flux_ramp_rate,sim_time, low_pass_ratio = 0.75, fs_required = 100e6):
        dt = 1.0 / fs_required  
        self.t_array = np.arange(0, sim_time, dt) 
        
        self.flux_ramp_rate = flux_ramp_rate
        self.low_pass_ratio = low_pass_ratio
        self.fs_required = fs_required
        self.sim_time = sim_time

    def mixer(self, signal_ac):
        w_fr = 2 * np.pi * self.flux_ramp_rate  
        ref_I = np.cos(w_fr * self.t_array) 
        ref_Q = np.sin(w_fr * self.t_array) 
        
        mixed_I = signal_ac * ref_I
        mixed_Q = signal_ac * ref_Q
        self.unfiltered_phase = np.unwrap(np.arctan2(mixed_Q, mixed_I))
        
        return mixed_I, mixed_Q
    
    def low_pass_filter(self, mixed_I, mixed_Q):
        cutoff_hz = self.flux_ramp_rate * self.low_pass_ratio
        b, a = butter(4, cutoff_hz, btype='low', fs=self.fs_required)
        
        filtered_I = filtfilt(b, a, mixed_I)
        filtered_Q = filtfilt(b, a, mixed_Q)
        
        return filtered_I, filtered_Q
    
    #To Do:
    #def average_over_period()s
        
    def phase_offset(self, filtered_I, filtered_Q):
        phase_rad = np.arctan2(filtered_Q, filtered_I)
        phase_rad_unwrapped = np.unwrap(phase_rad)
        phase_rad_unwrapped -= phase_rad_unwrapped[0] 
        self.filtered_phase = phase_rad_unwrapped
        
        return phase_rad_unwrapped
    
    def python_sampling(self, phase_rad_unwrapped):
        step_size = int(self.fs_required / self.flux_ramp_rate) #number of simulated points per flux ramp (simulated ADC sampling rate)
        t_sampled = self.t_array[::step_size]
        phase_sampled = -phase_rad_unwrapped[::step_size]
        
        return t_sampled, phase_sampled
        
    def demodulate(self, signal_ac):
        mixed_I, mixed_Q = self.mixer(signal_ac)
        filtered_I, filtered_Q = self.low_pass_filter(mixed_I, mixed_Q)
        phase_rad_unwrapped = self.phase_offset(filtered_I, filtered_Q)
        
        t_sampled, phase_sampled = self.python_sampling(phase_rad_unwrapped)
        
        return t_sampled, phase_sampled
    
        
    

        