import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from tqdm import tqdm
from si_prefix import si_format
from resonator_tools import circuit
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import xarray as xr
import pandas as pd

from iminuit import Minuit
from iminuit.cost import LeastSquares
import iminuit

from uncertainties import ufloat
from uncertainties import umath
from uncertainties import unumpy

Phi_0 = 2.067833831e-15  # Magnetic flux quantum in Wb (or V·s)
k_B = 1.38e-23  # Boltzmann constant in J/K

a = np.array([1, -1/2, -1/8, 3/8, 1/8, -5/16, 1/16, -5/32, 9/32, -5/64, 3/16, -17/64,
     -15/512, 57/512, -115/512, 133/512, 21/512, -77/512, 137/512, -267/1024,
     21/1024, -35/512, 103/512, -651/2048, 547/2048, -63/2048, 27/256, -1089/4096,
     193/512, -1139/4096, -105/8192, 435/8192, -2595/16384, 5705/16384, -7317/16384, 4807/16384])
b = np.array([0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5,
              6, 6, 6, 6, 7, 7, 7, 7,
              8, 8, 8, 8, 8, 9, 9, 9, 
              9, 9, 10, 10, 10, 10, 10, 10])
c = np.array([1, 2, 1, 3, 2, 4, 1, 3, 5, 2, 4, 6,
              1, 3, 5, 7, 2, 4, 6, 8,
              1, 3, 5, 7, 9, 2, 4, 6,
              8, 10, 1, 3, 5, 7, 9, 11])

def optimal_vol_bias(S21_norm, fit_results, flux_cut):
    """
    Find the optimal flux bias point that maximizes the slope of Im(S21) to flux transfer function.

    Parameters
    ----------
    S21_norm : xarray.DataArray
        Normalized S21 data with dimensions (sweep_freq, center_freq, vol_bias).
    fit_results : xarray.Dataset
        Fitting results containing 'fr' (resonance frequency) with dimensions (center_freq, vol_bias).
    flux_cut : list of tuples
        List of (min, max) tuples specifying the flux bias range for each resonator.

    Returns
    -------
    optimal_flux_points : list
        List of optimal flux bias points for each resonator.
    """
    num_res = len(fit_results.center_freq)
    optimal_flux_points = np.zeros(num_res)
    optimal_sweep_freqs = np.zeros(num_res)
    for i in range(num_res):

        # Apply the flux bias cut
        flux_min, flux_max = flux_cut[i]
        mask = (S21_norm.vol_bias >= flux_min) & (S21_norm.vol_bias <= flux_max)
        fr = fit_results['fr'].sel(center_freq=fit_results.center_freq[i], method='nearest').where(mask, drop=True)
        
        vol_bias_cut = S21_norm['vol_bias'].where(mask, drop=True)
        dS21_dflux = S21_norm['S21_norm'].sel(center_freq=fit_results.center_freq[i], method='nearest').where(mask, drop=True).imag.differentiate('vol_bias')
        dS21_dflux_fr = dS21_dflux.sel(sweep_freq=fr-fit_results.center_freq[i], method='nearest')

        # Find the flux bias that maximizes the absolute value of the derivative
        optimal_index = np.argmax(np.abs(np.array(dS21_dflux_fr)))
        optimal_flux = vol_bias_cut.values[optimal_index]
        optimal_flux_points[i] = optimal_flux
        optimal_sweep_freqs[i] = fr.values[optimal_index] - fit_results.center_freq[i].values

    return optimal_flux_points, optimal_sweep_freqs


def res_freq_finite_power(bias, Phi_rf, f_off, df_mod, M_mod, Phi_ext_off, beta):
    "Calculate the resonant frequency of the resonator from the bias current at finite rf powers"

    Phi_ext = M_mod * bias + Phi_ext_off

    phi_ext = 2*np.pi*Phi_ext
    phi_rf = 2*np.pi*Phi_rf

    p = np.zeros((len(a), len(phi_ext)))
    for i in range(len(a)):
        p[i, :] = a[i] * beta**b[i] * sp.j1(c[i]*phi_rf) * np.cos(c[i]*phi_ext)

    return f_off + df_mod*2*beta/phi_rf * np.sum(p, axis=0)


def solve_phi_tot(phi_tot, phi_ext, beta=0.):
    "Solve for the total flux in the resonator"
    return phi_tot + (beta/(2*np.pi))*np.sin(2*np.pi*phi_tot) - phi_ext


def res_freq_low_power(bias, f_off, df_mod, M_mod, Phi_ext_off, beta):
    "Calculate the resonant frequency of the resonator from the bias current in the low power limit"
    Phi_ext = M_mod * bias + Phi_ext_off
    Phi_tot = fsolve(solve_phi_tot, Phi_ext, args=(Phi_ext, beta))
    phi = 2*np.pi*Phi_tot

    return f_off + df_mod*(beta*np.cos(phi))/(1+beta*np.cos(phi))

def uMUX_low_pwr_fit(fit_results, fit_errors, period_guess, beta_guess, plot=False):
    """
    uMUX model for resonance frequency as a function of flux bias, in the low power limit.
    
    Parameters
    ----------
    fit_results : xarray.Dataset
        Fitting results containing 'fr' (resonance frequency) with dimensions (center_freq, vol_bias).
        
    Returns
    ----------
    model_params : dict
        Dictionary containing fitted parameters for each resonator.
    """

    num_res = len(fit_results.center_freq)
    fit_params = {
        'f_off': np.zeros(num_res),
        'df_mod': np.zeros(num_res),
        'M_mod': np.zeros(num_res),
        'Phi_ext_off': np.zeros(num_res),
        'beta': np.zeros(num_res)
    }
    fit_params_errors = {
        'f_off': np.zeros(num_res),
        'df_mod': np.zeros(num_res),
        'M_mod': np.zeros(num_res),
        'Phi_ext_off': np.zeros(num_res),
        'beta': np.zeros(num_res)
    }

    for i in range(num_res): 
        vol_bias = fit_results['vol_bias'].values
        fr = fit_results['fr'].sel(center_freq=fit_results.center_freq[i], method='nearest').values
        fr_err = fit_errors['fr_err'].sel(center_freq=fit_results.center_freq[i], method='nearest').values

        loss_function = LeastSquares(vol_bias, fr, fr_err, res_freq_low_power)

        # Initial guess for parameters
        fr_range = np.max(fr) - np.min(fr)
        f_off_guess = np.min(fr) + fr_range*0.8
        M_mod_guess = 1/period_guess  
        Phi_ext_off_guess = -fit_results['vol_bias'][np.argmax(fr)].values*M_mod_guess # Assuming the minimum is at -0.5 flux quanta
        # Make sure all guesses are within [-1, 1]
        if np.abs(Phi_ext_off_guess) > 1:
            Phi_ext_off_guess -= int(Phi_ext_off_guess)

        m = Minuit(loss_function, f_off=f_off_guess, df_mod = fr_range, M_mod=M_mod_guess, Phi_ext_off=Phi_ext_off_guess, beta=beta_guess)
        m.limits = [(f_off_guess*0.999, f_off_guess*1.001),
                    (fr_range*0.5, fr_range*2),
                    (0.5*M_mod_guess, 2*M_mod_guess),
                    (-1, 1),
                    (0.1, 1)]
        m.migrad()
        m.hesse()
        m.minos()

        fit_params['f_off'][i] = m.values['f_off']
        fit_params['df_mod'][i] = m.values['df_mod']
        fit_params['M_mod'][i] = m.values['M_mod']
        fit_params['Phi_ext_off'][i] = m.values['Phi_ext_off']
        fit_params['beta'][i] = m.values['beta']


        fit_params_errors['f_off'][i] = m.errors['f_off']
        fit_params_errors['df_mod'][i] = m.errors['df_mod']
        fit_params_errors['M_mod'][i] = m.errors['M_mod']
        fit_params_errors['Phi_ext_off'][i] = m.errors['Phi_ext_off']
        fit_params_errors['beta'][i] = m.errors['beta']

        if plot:
            fr_fit = res_freq_low_power(vol_bias, m.values['f_off'], m.values['df_mod'], m.values['M_mod'], m.values['Phi_ext_off'], m.values['beta'])
            plt.figure(figsize=(6,4))
            plt.errorbar(vol_bias, fr, yerr=fr_err, fmt='o', label='Data')
            plt.plot(vol_bias, fr_fit, '-', label='Fit', zorder=10)
            plt.xlabel('Flux Bias (V)')
            plt.ylabel('Resonance Frequency (Hz)')
            plt.title(f'Resonator {i+1}')
            plt.legend()
            plt.grid()
            plt.show()

    return fit_params, fit_params_errors


def uMUX_finite_pwr_fit(fit_results, fit_errors, Phi_rf_guess, period_guess, beta_guess, plot=False):
    """
    uMUX model for resonance frequency as a function of flux bias, considering finite power effects.

    Parameters
    ----------
    fit_results : xarray.Dataset
        Fitting results containing 'fr' (resonance frequency) with dimensions (center_freq, vol_bias).

    Returns
    -------
    model_params : dict
        Dictionary containing fitted parameters for each resonator.
    """

    num_res = len(fit_results.center_freq)
    fit_params = {
        'f_off': np.zeros(num_res),
        'df_mod': np.zeros(num_res),
        'M_mod': np.zeros(num_res),
        'Phi_ext_off': np.zeros(num_res),
        'beta': np.zeros(num_res),
        'Phi_rf': np.zeros(num_res)
    }

    for i in range(num_res):
        vol_bias = fit_results['vol_bias'].values
        fr = fit_results['fr'].sel(center_freq=fit_results.center_freq[i], method='nearest').values
        fr_err = fit_errors['fr_err'].sel(center_freq=fit_results.center_freq[i], method='nearest').values

        # Select the fr values that are within the range +-0.5 MHz around the median value of fr
        fr_median = np.median(fr)
        mask = (fr >= fr_median - 0.5e6) & (fr <= fr_median + 0.5e6)
        vol_bias = vol_bias[mask]
        fr = fr[mask]
        fr_err = fr_err[mask]

        loss_function = LeastSquares(vol_bias, fr, fr_err, res_freq_finite_power)

        # Initial guess for parameters
        fr_range = np.max(fr) - np.min(fr)
        f_off_guess = np.min(fr) + fr_range*0.8
        M_mod_guess = 1/period_guess
        Phi_ext_off_guess = -fit_results['vol_bias'][np.argmin(fr)]*M_mod_guess-0.5

        if Phi_ext_off_guess < -1:
            Phi_ext_off_guess = -Phi_ext_off_guess


        m = Minuit(loss_function, Phi_rf=Phi_rf_guess, f_off=f_off_guess, df_mod = fr_range, M_mod=M_mod_guess, Phi_ext_off=Phi_ext_off_guess, beta=beta_guess)
        m.limits = [(0.1, 0.6),
                    (f_off_guess*0.999, f_off_guess*1.001),
                    (fr_range*0.6, fr_range*2),
                    (0.5*M_mod_guess, 1.5*M_mod_guess),
                    (-1, 1),
                    (beta_guess-0.2, beta_guess+0.2)]
        m.migrad()
        m.hesse()
        try:
            m.minos()
        except Exception as e:
            print(f"Minuit fitting failed for resonator {i+1}: {e}")

        fit_params['Phi_rf'][i] = m.values['Phi_rf']
        fit_params['f_off'][i] = m.values['f_off']
        fit_params['df_mod'][i] = m.values['df_mod']
        fit_params['M_mod'][i] = m.values['M_mod']
        fit_params['Phi_ext_off'][i] = m.values['Phi_ext_off']
        fit_params['beta'][i] = m.values['beta']

        if plot:
            fr_fit = res_freq_finite_power(vol_bias, m.values['Phi_rf'], m.values['f_off'], m.values['df_mod'], m.values['M_mod'], m.values['Phi_ext_off'], m.values['beta'])

            plt.figure(figsize=(6,4))
            plt.errorbar(vol_bias, fr, yerr=fr_err, fmt='o', label='Data')
            plt.plot(vol_bias, fr_fit, '-', label='Fit', zorder=10)
            plt.xlabel('Flux Bias (V)')
            plt.ylabel('Resonance Frequency (Hz)')
            plt.title(f'Resonator {i+1}')
            plt.legend()
            plt.grid()
            plt.show()


    return fit_params

def pick_flux_points(flux_axis, num_points=3):
    """
    Pick evenly spaced flux points from the flux axis between integer flux quanta and half-integer flux quanta.

    Parameters
    ----------
    flux_axis : array-like
        Array of flux bias values.
    num_points : int, optional
        Number of flux points to pick. Default is 3.

    Returns
    -------
    selected_flux_points : array
        Array of the indices of the selected flux points.
    """
    min_flux = np.min(np.ceil(flux_axis))
    max_flux = min_flux + 0.5
    if max_flux > np.max(flux_axis):
        max_flux = np.min(np.ceil(flux_axis))
        min_flux = max_flux - 0.5
    selected_flux_points = np.linspace(min_flux, max_flux, num_points, endpoint=True)
    return selected_flux_points

def attenuation_fit(Phi_ext_power, alpha, f_off, df_mod, beta):
    "Calculate the resonant frequency of the resonator from the bias current at finite rf powers"

    Phi_ext, vna_powers = Phi_ext_power
    
    phi_ext = 2*np.pi*Phi_ext
    Phi_rf = alpha * np.sqrt(10**(vna_powers/10)/1000)
    phi_rf = 2*np.pi*Phi_rf/Phi_0

    p = np.zeros((len(a), len(phi_ext)))
    for i in range(len(a)):
        p[i, :] = a[i] * beta**b[i] * sp.j1(c[i]*phi_rf) * np.cos(c[i]*phi_ext)

    return f_off + df_mod*2*beta/phi_rf * np.sum(p, axis=0)


def attenuation_fit_bulk(fit_results, fit_errors, uMUX_model_fit_params, uMUX_model_fit_params_err, flux_pt_num=3):

    """
    Fit the attenuation using the uMUX model and circle fit results.

    Parameters
    ----------
    fit_results : xarray.Dataset
        Fitting results containing 'fr' (resonance frequency) with dimensions (center_freq, probe_pwr, vol_bias).
    fit_errors : xarray.Dataset
        Fitting errors containing 'fr_err' (error in resonance frequency) with dimensions (center_freq, probe_pwr, vol_bias).
    uMUX_model_fit_params : dict
        Fitted parameters from the uMUX model.
    uMUX_model_fit_params_err : dict
        Errors in the fitted parameters from the uMUX model.
    plot : bool, optional
        Whether to plot the fit results for each resonator. Default is False.

    Returns
    -------
    model_params : dict
        Dictionary containing fitted parameters for each resonator.
    """
    res_num = len(fit_results.center_freq)
    alpha = np.zeros(res_num)
    alpha_err = np.zeros(res_num)
    colors = plt.cm.viridis(np.linspace(0, 1, flux_pt_num))

    flux_sel_list = np.zeros((res_num, flux_pt_num))
    exp_data_list = np.zeros((res_num, flux_pt_num, len(fit_results.probe_pwr)))
    exp_data_err_list = np.zeros((res_num, flux_pt_num, len(fit_results.probe_pwr)))
    fit_data_list = np.zeros((res_num, flux_pt_num, len(fit_results.probe_pwr)))
    

    for i in tqdm(range(res_num)):

        flux_axis = fit_results['vol_bias'].values * uMUX_model_fit_params['M_mod'][i] + uMUX_model_fit_params['Phi_ext_off'][i]
        print(flux_axis[0], flux_axis[-1])
        flux_sel = pick_flux_points(flux_axis, num_points=flux_pt_num)
        vol_bias_sel = (flux_sel - uMUX_model_fit_params['Phi_ext_off'][i]) / uMUX_model_fit_params['M_mod'][i]

        exp_data = fit_results['fr'].sel(center_freq=fit_results.center_freq[i], vol_bias=vol_bias_sel, method='nearest').values
        exp_data_err = fit_errors['fr_err'].sel(center_freq=fit_results.center_freq[i], vol_bias=vol_bias_sel, method='nearest').values


        # Need to flatten everything into a 1D array for iminuit
        output_powers_grid, Phi_ext_grid = np.meshgrid(fit_results['probe_pwr'], flux_sel)

        
        Phi_ext_flatten = Phi_ext_grid.flatten()
        output_powers_flatten = output_powers_grid.flatten()
        exp_data_flatten = exp_data.flatten()
        exp_data_err_flatten = exp_data_err.flatten()
        
        loss_function_att = LeastSquares((Phi_ext_flatten, output_powers_flatten), exp_data_flatten, exp_data_err_flatten, attenuation_fit)
        
        m_att = Minuit(loss_function_att, alpha=1e-15, f_off=uMUX_model_fit_params['f_off'][i], df_mod=uMUX_model_fit_params['df_mod'][i], beta=uMUX_model_fit_params['beta'][i])
        m_att.fixed['f_off'] = False
        m_att.fixed['df_mod'] = False
        m_att.fixed['beta'] = False

        m_att.limits = [(1e-20, 1e-10), 
                        (uMUX_model_fit_params['f_off'][i]*0.999, uMUX_model_fit_params['f_off'][i]*1.001),
                        (uMUX_model_fit_params['df_mod'][i]*0.1, uMUX_model_fit_params['df_mod'][i]*4),
                        (0.05, 0.95)]
        m_att.migrad()
        m_att.hesse()
        m_att.minos()

        alpha[i] = m_att.values['alpha']
        alpha_err[i] = m_att.errors['alpha']

        fit_data = attenuation_fit((Phi_ext_flatten, output_powers_flatten), m_att.values['alpha'], m_att.values['f_off'], m_att.values['df_mod'], m_att.values['beta']) 
        fit_data = fit_data.reshape((len(flux_sel), len(fit_results.probe_pwr)))

        flux_sel_list[i, :] = flux_sel
        exp_data_list[i, :, :] = exp_data
        exp_data_err_list[i, :, :] = exp_data_err
        fit_data_list[i, :, :] = fit_data

        #att_dB = 10*np.log10((alpha/(np.sqrt(16/np.pi*uMUX_model_fit_params['Q_l']**2/uMUX_model_fit_params['Q_c']/uMUX_model_fit_params['Z_0'])*uMUX_model_fit_params['M_T']))**2)

    alpha_uarray = unumpy.uarray(alpha, alpha_err)
    Q_l_uarray = unumpy.uarray(uMUX_model_fit_params['Q_l'], uMUX_model_fit_params_err['Q_l'])
    Q_c_uarray = unumpy.uarray(uMUX_model_fit_params['Q_c'], uMUX_model_fit_params_err['Q_c'])
    M_T_uarray = unumpy.uarray(uMUX_model_fit_params['M_T'], uMUX_model_fit_params_err['M_T'])
    #att = (alpha_uarray/(unumpy.sqrt(16/np.pi*Q_l_uarray**2/Q_c_uarray/uMUX_model_fit_params['Z_0'][i])*M_T_uarray))**2
    att = (alpha_uarray/M_T_uarray)**2 * (np.pi*uMUX_model_fit_params['Z_0'][i]*Q_c_uarray) / (16 * Q_l_uarray**2)
      
    #return att_dB, alpha, alpha_err, flux_sel_list, exp_data_list, exp_data_err_list, fit_data_list
    return att, alpha, alpha_err, flux_sel_list, exp_data_list, exp_data_err_list, fit_data_list



def internal_Q(vol_bias, Q_offset, A, beta, M_mod, Phi_ext_off):
    "Calculate the internal Q of the resonator from the bias current"
    Phi_ext = M_mod * vol_bias + Phi_ext_off
    Phi_tot = fsolve(solve_phi_tot, Phi_ext, args=(Phi_ext, beta))

    Qi = 1/(1/Q_offset + 1/(A * (1+beta*np.cos(2*np.pi*Phi_tot))**2))
    
    return Qi
