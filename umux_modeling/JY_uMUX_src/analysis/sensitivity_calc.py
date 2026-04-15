import numpy as np
import scipy.special as sp
import matplotlib.colors as colors
import scipy.integrate as spi


Phi_0 = 2.07e-15
k_B = 1.38e-23


def create_cmap_norm(vmin: float, vmax: float, num_levels: int):
    '''
    Create a colormap and normalization for a given range and number of levels.

    vmin: float
        The minimum value for the colormap
    vmax: float
        The maximum value for the colormap
    num_levels: int
        The number of levels in the colormap
    return: tuple
        A tuple containing the levels and normalization for the colormap
    '''
    if not isinstance(vmin, float):
        raise TypeError("vmin must be a float")

    lev_exp = np.linspace(np.floor(np.log10(vmin)-1), np.ceil(np.log10(vmax)+1), num_levels)
    levs = np.power(10, lev_exp)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    return levs, norm



# Coefficients for the Bessel function expansion
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

class resonators:
    def __init__(self, beta: float, f_off: float, df_mod: float, Q_c: float, Q_l: float, Z_0: float, M_T: float):

        self.beta = beta
        self.f_off = f_off
        self.df_mod = df_mod
        self.Q_c = Q_c
        self.Q_l = Q_l
        self.Q_i = 1/(1/self.Q_l - 1/self.Q_c)
        self.Z_0 = Z_0
        self.M_T = M_T

    def MaxAmplitude_fres(self, Phi_ext: float, P_exc_dBm: float):
        '''
        Compute the resonant frequency of the resonator given the bias flux and rf power.
        This is done assuming maximum anti-node current in the resonator termination.

        Phi_ext: The external flux bias point in units of Phi0
        P_exc_dBm: The probe tone power at device in dBm
        returns: The maximum amplitude of the resonator in units of Phi0
        '''

        P_exc = 10**(P_exc_dBm/10)/1000 # convert dBm to W
        I_T = np.sqrt(16*self.Q_l**2*P_exc/(np.pi*self.Q_c*self.Z_0))
        phi_rf = 2*np.pi/Phi_0 * self.M_T*I_T
        phi_ext = 2*np.pi*Phi_ext
        p = a * self.beta**b * sp.j1(c*phi_rf)*np.cos(c*phi_ext)
        p_deriv = - a * self.beta**b * sp.j1(c*phi_rf)*c*2*np.pi*np.sin(c*phi_ext)
        d_fr_d_Phi = self.df_mod * 2 * self.beta/phi_rf * np.sum(p_deriv)
        f_res = self.f_off + self.df_mod * 2 * self.beta / phi_rf * np.sum(p)

        return f_res, d_fr_d_Phi
    
    def S21_to_flux_analytic(self, f_exc: float, Phi_ext: float, P_exc_dBm: float, demod_domain: str):
        '''
        Calculate the d(S21)/d(Phi) transfer function analytically

        f_exc: The frequency of the probe tone
        Phi_ext: The external flux bias in units of Phi0
        P_exc_dBm: The probe tone power at device in dBm
        T_N: The noise temperature in Kelvin 
        demod_domain: The domain in which to calculate the sensitivity ('imag' or 'real')
        returns: The amplifier noise in flux units
        '''

        f_res, d_fr_d_Phi = self.MaxAmplitude_fres(Phi_ext, P_exc_dBm)
        x = (f_exc-f_res)/f_res
        
        # These partial derivatives are calculated analytically from Eq 13 and 14 in the paper
        if demod_domain == 'imag':
            d_S21_d_fr = - 2*self.Q_l**2/self.Q_c * (1-4*self.Q_l**2*x**2)/(1+4*self.Q_l**2*x**2)**2 * f_exc/f_res**2
        elif demod_domain == 'real':
            d_S21_d_fr = - 8*self.Q_l**3/self.Q_c * x/(1+4*self.Q_l**2*x**2)**2 * f_exc/f_res**2

        d_S21_d_Phi = d_S21_d_fr * d_fr_d_Phi

        return d_S21_d_Phi

    def S21_to_flux_numeric(self, f_exc: float, Phi_ext: float, P_exc_dBm: float, demod_domain: str, Phi_ext_step: float=1e-5):
        '''
        Calculate the amplifier noise converted to flux units in the imaginary or real domain of resonator S21.
        This is done numerically for cross-checking purposes.

        f_exc: The frequency of the probe tone
        Phi_ext: The external flux bias in units of Phi0
        P_exc_dBm: The probe tone power at device in dBm
        T_N: The noise temperature in Kelvin
        demod_domain: The domain in which to calculate the sensitivity ('imag' or 'real')
        returns: The amplifier noise in flux units
        '''

        Phi_ext_low = Phi_ext - Phi_ext_step/2
        Phi_ext_high = Phi_ext + Phi_ext_step/2

        f_res_low, _ = self.MaxAmplitude_fres(Phi_ext_low, P_exc_dBm)
        S21_low = 1-(self.Q_l/self.Q_c*np.exp(1j*Phi_0))/(1+2j*self.Q_l*(f_exc-f_res_low)/f_res_low)
        f_res_high, _ = self.MaxAmplitude_fres(Phi_ext_high, P_exc_dBm)
        S21_high = 1-(self.Q_l/self.Q_c*np.exp(1j*Phi_0))/(1+2j*self.Q_l*(f_exc-f_res_high)/f_res_high)

        if demod_domain == 'imag':
            d_S21_d_Phi = (S21_high.imag - S21_low.imag) / Phi_ext_step
        elif demod_domain == 'real':
            d_S21_d_Phi = (S21_high.real - S21_low.real) / Phi_ext_step

        return d_S21_d_Phi

    def amp_noise_analytic(self, f_exc: float, Phi_ext: float, P_exc_dBm: float, T_N: float, demod_domain: str):
        '''
        Calculate the amplifier noise converted to flux units in the imaginary or real domain of resonator S21.
        This is done analytically

        f_exc: The frequency of the probe tone
        Phi_ext: The external flux bias in units of Phi0
        P_exc_dBm: The probe tone power at device in dBm
        T_N: The noise temperature in Kelvin 
        demod_domain: The domain in which to calculate the sensitivity ('imag' or 'real')
        returns: The amplifier noise in flux units
        '''

        P_exc = 10**(P_exc_dBm/10)/1000 # convert dBm to W
        d_S21_d_Phi = self.S21_to_flux_analytic(f_exc, Phi_ext, P_exc_dBm, demod_domain)

        noise = np.sqrt(k_B*T_N/P_exc)

        flux_noise = noise / np.abs(d_S21_d_Phi)

        return flux_noise


    def amp_noise_numeric(self, f_exc: float, Phi_ext: float, P_exc_dBm: float, T_N: float, demod_domain: str, Phi_ext_step: float=1e-5):
        '''
        Calculate the amplifier noise converted to flux units in the imaginary or real domain of resonator S21.
        This is done numerically for cross-checking purposes.

        f_exc: The frequency of the probe tone
        Phi_ext: The external flux bias in units of Phi0
        P_exc_dBm: The probe tone power at device in dBm
        T_N: The noise temperature in Kelvin
        demod_domain: The domain in which to calculate the sensitivity ('imag' or 'real')
        returns: The amplifier noise in flux units
        '''
        P_exc = 10**(P_exc_dBm/10)/1000 # convert dBm to W
        d_S21_d_Phi = self.S21_to_flux_numeric(f_exc, Phi_ext, P_exc_dBm, demod_domain, Phi_ext_step)

        noise = np.sqrt(k_B*T_N/P_exc)

        flux_noise = noise / np.abs(d_S21_d_Phi)

        return flux_noise
    

    def FRM_noise(self, f_exc: float, P_exc_dBm: float, T_N: float, alpha: float, demod_domain: str):
        '''
        Calculate the flux ramping noise in the resonator assuming the only source of noise is from the amplifier.

        f_exc: The frequency of the probe tone
        P_exc_dBm: The probe tone power at device in dBm
        T_N: The noise temperature in Kelvin
        demod_domain: The domain in which to calculate the sensitivity ('imag' or 'real')
        alpha: the part of the flux ramping interval that you keep
        returns: The flux ramping noise in flux units
        '''

        transfer_func_avg = spi.quad(lambda x: (self.S21_to_flux_analytic(f_exc, x, P_exc_dBm, demod_domain))**2, 0, 1)[0]
        P_exc = 10**(P_exc_dBm/10)/1000 # convert dBm to W
        S_sqrt = np.sqrt(k_B*T_N/P_exc)

        FRM_noise = 1/np.sqrt(alpha) * S_sqrt / np.sqrt(transfer_func_avg)

        return FRM_noise, np.sqrt(transfer_func_avg)