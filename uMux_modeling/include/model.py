import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks

### ENTITIES ###
class MicrowaveDevice:
    def Z(self, f): raise NotImplementedError
    
    def get_abcd(self, f, mode='series'):
        z = self.Z(f)
        if np.isscalar(z):
            if mode == 'series':
                return np.array([[1, z], [0, 1]], dtype=complex)
            else:  # shunt
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.array([[1, 0], [1/z, 1]], dtype=complex)
        else:
            N = len(z)
            abcd = np.zeros((N, 2, 2), dtype=complex)
            abcd[:, 0, 0] = 1
            abcd[:, 1, 1] = 1
            if mode == 'series':
                abcd[:, 0, 1] = z
            else:  # shunt
                with np.errstate(divide='ignore', invalid='ignore'):
                    abcd[:, 1, 0] = 1 / z
            return abcd

class SQUID(MicrowaveDevice):
    """
    Models a SQUID as a flux-tunable inductor. Its impedance is determined
    by the total external magnetic flux applied to its loop.
    """
    def __init__(self, L_j, L_s, phi_ext=0.0):
        self.L_j = L_j
        self.L_s = L_s
        self.phi_ext = phi_ext
        self.PHI_0 = 2.067e-15  # Flux quantum
        self.flux_lines = {}
        
    def add_flux_line(self, name, M, initial_current=0.0):
        line = FluxLine(name, M, initial_current)
        self.flux_lines[name] = line
        return line
        
    def set_current(self, name, new_I):
        if name in self.flux_lines:
            self.flux_lines[name].I = new_I
        else:
            raise KeyError(f"Flux line '{name}' does not exist on this SQUID.")
        
    def get_total_flux(self):
        total_flux = self.phi_ext
        for line in self.flux_lines.values():
            total_flux += line.get_flux()
        return total_flux
    
    def set_flux(self, phi):
        self.phi_ext = phi

    def get_L(self):
        phi = self.get_total_flux()
        cos_term = np.cos(np.pi * phi / self.PHI_0)
        abs_cos_term = np.abs(cos_term)

        if np.isscalar(abs_cos_term):
            L_j_tuned = float('inf') if abs_cos_term < 1e-9 else self.L_j / abs_cos_term
        else:
            L_j_tuned = np.where(abs_cos_term < 1e-9, float('inf'), self.L_j / abs_cos_term)

        return self.L_s + L_j_tuned
    def Z(self, f):
        omega = 2 * np.pi * f
        return 1j * omega * self.get_L()

class FluxLine:
    def __init__(self, name, M, initial_current=0.0):
        self.name = name
        self.M = M
        self.I = initial_current
        
    def get_flux(self):
        #print("Flux on",self.name,"=",self.M*self.I)
        return float(self.M)*self.I
class Resonator(MicrowaveDevice):
    def __init__(self, length, load, v_p=1.15e8, Qi=100000, Z0=50):
        self.length = length
        self.load = load
        self.v_p = v_p
        self.Qi = Qi
        self.Z0 = Z0
        
    def Z(self, f):
        if f == 0: return float('inf')
        
        beta = 2 * np.pi * f / self.v_p
        alpha = beta / (2 * self.Qi)
        gamma = alpha + 1j * beta
        
        zl_val = self.load.Z(f) 
        
        num = zl_val + self.Z0 * np.tanh(gamma * self.length)
        den = self.Z0 + zl_val * np.tanh(gamma * self.length)
        
        return self.Z0 * (num / den)

    
class Inductor(MicrowaveDevice):
    def __init__(self, L):
        self.L = L
        
    def Z(self, f):
        omega = 2 * np.pi * f
        return 1j * omega * self.L

class ScreenedInductor(MicrowaveDevice):
    def __init__(self, primary_inductor, screening_device, M):
        if not isinstance(primary_inductor, Inductor):
            raise TypeError("The primary_inductor must be an instance of the Inductor class.")
        self.primary_inductor = primary_inductor
        self.screening_device = screening_device
        self.M = M

    def Z(self, f):
        if f == 0:
            return self.primary_inductor.Z(f)

        omega = 2 * np.pi * f
        z_primary = self.primary_inductor.Z(f)
        z_screening = self.screening_device.Z(f)

        if np.isscalar(z_screening):
            if abs(z_screening) < 1e-12:
                return z_primary + complex(0, float('inf'))
            if abs(z_screening) == float('inf'):
                return z_primary
            z_reflected = (omega * self.M)**2 / z_screening
        else:
            z_reflected = np.zeros_like(z_screening, dtype=complex)
            
            valid_mask = (np.abs(z_screening) >= 1e-12) & (~np.isinf(z_screening))
            z_reflected[valid_mask] = (omega * self.M)**2 / z_screening[valid_mask]
            
            zero_mask = np.abs(z_screening) < 1e-12
            z_reflected[zero_mask] = complex(0, float('inf'))

        return z_primary + z_reflected

class Capacitor(MicrowaveDevice):
    def __init__(self, C):
        self.C = C

    def Z(self, f):
        if f == 0: return float('inf') 
        omega = 2 * np.pi * f
        return -1j / (omega * self.C)
    
class Terminator(MicrowaveDevice):
    def __init__(self, R):
        self.R = R

    def Z(self, f):
        return complex(self.R,0)

### SYSTEMS ###

class VNA_Simulator:
    def __init__(self, z0=50):
        self.z0 = z0
        self.chain = []

    def add(self, component, mode='series'):
        self.chain.append((component, mode))

    def get_s_matrix(self, f):
        if not np.isscalar(f):
            s_params = np.array([self.get_s_matrix(freq) for freq in f])
            print(s_params.shape)
            return np.moveaxis(s_params, 0, -1)

        abcd = np.identity(2, dtype=complex) 
        for comp, mode in self.chain:
            comp_abcd = comp.get_abcd(f, mode)
            abcd = abcd @ comp_abcd

        if abcd.ndim == 3:
            A, B = abcd[:, 0, 0], abcd[:, 0, 1]
            C, D = abcd[:, 1, 0], abcd[:, 1, 1]
        else:
            A, B, C, D = abcd.flatten()
            
        z0 = self.z0
        denom = A + B/z0 + C*z0 + D
        s11 = (A + B/z0 - C*z0 - D) / denom
        s12 = (2 * (A*D - B*C)) / denom
        s21 = 2 / denom
        s22 = (-A + B/z0 - C*z0 + D) / denom
        
        if abcd.ndim == 3:
            N = abcd.shape[0]
            s_matrix = np.zeros((N, 2, 2), dtype=complex)
            s_matrix[:, 0, 0] = s11
            s_matrix[:, 0, 1] = s12
            s_matrix[:, 1, 0] = s21
            s_matrix[:, 1, 1] = s22
            return s_matrix
        else:
            return np.array([[s11, s12], [s21, s22]])
    
    def draw_schematic(self, max_draw=4, use_color=True):
        num_channels = len(self.chain)
        
        if num_channels <= max_draw:
            draw_indices = list(range(num_channels))
            has_ellipsis = False
        else:
            draw_indices = [0, 1, 2, num_channels - 1]
            has_ellipsis = True

        x_start = 2
        x_spacing = 6.5 
        num_slots = len(draw_indices) + (1 if has_ellipsis else 0)
        total_x = x_start + (num_slots * x_spacing) + 2
        
        plot_width = max(10, total_x * 0.7)
        fig, ax = plt.subplots(figsize=(plot_width, 7), facecolor='white')
        ax.set_facecolor('white')

        y_feed = 8

        # --- COLOR PALETTE ---
        if use_color:
            c = {
                'feed': '#2c3e50',       
                'cap': '#27ae60',        
                'tline_e': '#2980b9',    
                'tline_f': '#ebf5fb',    
                'ind': '#d35400',        
                'squid': '#8e44ad',      
                'res': '#c0392b',        
                'src': '#16a085',        
                'gnd': '#7f8c8d',
                'flux': '#e67e22'        
            }
        else:
            c = {k: 'k' for k in ['feed', 'cap', 'tline_e', 'ind', 'squid', 'res', 'src', 'gnd', 'flux']}
            c['tline_f'] = 'white'

        # --- HELPER FUNCTIONS ---
        def draw_ground(x, y):
            ax.plot([x, x], [y, y-0.2], color=c['gnd'], lw=1.5)
            ax.plot([x-0.2, x+0.2], [y-0.2, y-0.2], color=c['gnd'], lw=1.5)
            ax.plot([x-0.12, x+0.12], [y-0.28, y-0.28], color=c['gnd'], lw=1.5)
            ax.plot([x-0.05, x+0.05], [y-0.36, y-0.36], color=c['gnd'], lw=1.5)

        def draw_resistor(x, y_top, length=1.0, res_val=None):
            y_bot = y_top - length
            y = np.linspace(y_top, y_bot, 11)
            x_res = x + 0.12 * np.array([0, 1, -1, 1, -1, 1, -1, 1, -1, 1, 0])
            ax.plot(x_res, y, color=c['res'], lw=1.5)
            if res_val is not None:
                ax.text(x + 0.3, y_top - length / 2, f'{res_val:.0f} Ω', ha='left', va='center', fontsize=9, color=c['res'])
            return y_bot

        def draw_cap(x, y_top, cap=None):
            ax.plot([x, x], [y_top, y_top-0.4], color=c['cap'], lw=1.5)
            ax.plot([x-0.2, x+0.2], [y_top-0.4, y_top-0.4], color=c['cap'], lw=1.5)
            ax.plot([x-0.2, x+0.2], [y_top-0.5, y_top-0.5], color=c['cap'], lw=1.5)
            ax.plot([x, x], [y_top-0.5, y_top-1.0], color=c['cap'], lw=1.5)
            if cap:
                val_str = f"{cap.C * 1e15:.1f} fF"
                ax.text(x + 0.3, y_top - 0.5, f"$C_c$={val_str}", ha='left', va='center', fontsize=9, color=c['cap'])
            return y_top - 1.0

        def draw_tline(x, y_top, length=3.5, tline=None):
            rect = patches.Rectangle((x-0.15, y_top-length), 0.3, length, 
                                     facecolor=c['tline_f'], edgecolor=c['tline_e'], lw=1.5, zorder=3)
            ax.add_patch(rect)
            ax.plot([x, x], [y_top, y_top-0.2], color=c['tline_e'], lw=1.5)
            ax.plot([x, x], [y_top-length+0.2, y_top-length], color=c['tline_e'], lw=1.5)
            if tline:
                len_str = f"{tline.length * 1e3:.1f} mm"
                ax.text(x + 0.3, y_top - length / 2, f"len={len_str}", ha='left', va='center', fontsize=9, color=c['tline_e'])
                z0_str = f"$Z_0$={tline.Z0:.0f}Ω"
                ax.text(x - 0.3, y_top - length / 2, z0_str, ha='right', va='center', fontsize=9, color=c['tline_e'])
            return y_top - length

        def draw_inductor(x, y_top, loops=4, ind=None):
            y_bot = y_top - 1.0
            y = np.linspace(y_top, y_bot, 100)
            x_ind = x + 0.12 * -np.abs(np.sin(loops * np.pi * (y_top - y) / (y_top - y_bot)))
            ax.plot(x_ind, y, color=c['ind'], lw=1.5)
            if ind:
                val_str = f"{ind.L * 1e12:.1f} pH"
                ax.text(x - 0.35, 0.3 + (y_top + y_bot) / 2, f"$L_c$={val_str}", ha='right', va='center', fontsize=10, color=c['ind'])
            else:
                ax.text(x - 0.35, 0.3 + (y_top + y_bot) / 2, '$L_c$', ha='right', va='center', fontsize=11, color=c['ind'])
            return y_bot

        def draw_squid(x_left, y_center, squid_params=None):
            sq_w, sq_h = 0.8, 1.0 
            x_sq = x_left + 0.4   
            y_sq_bot = y_center - sq_h/2
            
            # SQUID Loop
            rect = patches.Rectangle((x_sq, y_sq_bot), sq_w, sq_h, fill=False, edgecolor=c['squid'], lw=1.5)
            ax.add_patch(rect)
            
            # Josephson Junction Cross
            x_jj = x_sq + sq_w
            jj_size = 0.08 
            ax.plot([x_jj-jj_size, x_jj+jj_size], [y_center-jj_size, y_center+jj_size], color=c['squid'], lw=1.5)
            ax.plot([x_jj-jj_size, x_jj+jj_size], [y_center+jj_size, y_center-jj_size], color=c['squid'], lw=1.5)
            
            if squid_params:
                m_val_str = f"{squid_params['M'] * 1e12:.1f} pH"
                ls_val_str = f"{squid_params['L_s'] * 1e12:.1f} pH"
                lj_val_str = f"{squid_params['L_j'] * 1e12:.1f} pH"
                ax.text(x_left - 0.1, y_center, f'$M_c$={m_val_str}', ha='right', va='center', fontsize=9, color='k')
                ax.text(x_sq + sq_w / 2, y_sq_bot + 0.15, f'$L_S$={ls_val_str}', ha='center', fontsize=9, color=c['squid'])
                ax.text(x_jj + 0.15, y_center+0.35, f'$L_j$={lj_val_str}', ha='left', va='center', fontsize=9, color=c['squid'])

                if 'flux_lines' in squid_params and squid_params['flux_lines']:
                    flux_x = x_jj + 0.9  # <-- INCREASED: Starts further right from the SQUID
                    for name, line in squid_params['flux_lines'].items():
                        y_top = y_center + 0.4
                        y_bot = y_center - 0.4
                        
                        # Draw Inductive coil
                        y = np.linspace(y_top, y_bot, 50)
                        x_ind = flux_x + 0.08 * -np.abs(np.sin(3 * np.pi * (y_top - y) / (y_top - y_bot)))
                        ax.plot(x_ind, y, color=c['flux'], lw=1.5)
                        
                        # Draw leads
                        ax.plot([flux_x, flux_x], [y_top, y_top+0.3], color=c['flux'], lw=1.5)
                        ax.plot([flux_x, flux_x], [y_bot, y_bot-0.3], color=c['flux'], lw=1.5)

                        # Dashed line for mutual Inductance coupling
                        ax.plot([x_jj+0.25, flux_x-0.15], [y_center, y_center], ls=':', color='gray', lw=1.0)

                        # Labels
                        ax.text(flux_x, y_top + 0.4, name, ha='center', va='bottom', fontsize=9, color=c['flux'], fontweight='bold')
                        ax.text(flux_x + 0.15, y_center + 0.15, f"M={line.M*1e12:.0f} pH", ha='left', va='center', fontsize=8, color='k')
                        ax.text(flux_x + 0.15, y_center - 0.15, f"I={line.I*1e6:.1f} µA", ha='left', va='center', fontsize=8, color='k')

                        flux_x += 1.3 # <-- INCREASED: Larger gap before the next flux line

            else:
                ax.text(x_left -1, y_center, '$M_c$', ha='center', va='center', fontsize=11, color='k')
                ax.text(x_sq + sq_w / 2, y_sq_bot + 0.15, '$L_S$', ha='center', fontsize=11, color=c['squid'])
                ax.text(x_jj + 0.25, y_center, '$L_J$', ha='center', va='center', fontsize=11, color=c['squid'])

        # --- DRAWING THE CIRCUIT ---
        
        # 1. Main Feedline
        ax.plot([1, total_x-1], [y_feed, y_feed], color=c['feed'], lw=2.5)

        # 2. Input Port (VNA Source)
        ax.plot([1, 1], [y_feed, y_feed-1], color=c['feed'], lw=1.5)
        source_circ = patches.Circle((1, y_feed-1.5), 0.5, fill=False, ec=c['src'], lw=1.5)
        ax.add_patch(source_circ)
        t = np.linspace(0.7, 1.3, 50)
        ax.plot(t, y_feed-1.5 + 0.15*np.sin(2*np.pi*(t-0.7)/0.6), color=c['src'], lw=1.5)
        y_r_bot = draw_resistor(1, y_feed-2.0, res_val=self.z0)
        draw_ground(1, y_r_bot)

        # 3. Output Port (Terminated)
        ax.plot([total_x-1, total_x-1], [y_feed, y_feed-1], color=c['feed'], lw=1.5)
        y_r_out = draw_resistor(total_x-1, y_feed-1, res_val=self.z0)
        draw_ground(total_x-1, y_r_out)

        # 4. Draw Each Channel smartly
        current_slot = 0
        for i in range(num_channels):
            if i not in draw_indices:
                if has_ellipsis and i == 3: 
                    x_ell = x_start + (current_slot * x_spacing)
                    ax.text(x_ell, y_feed - 2.5, '. . .', ha='center', va='center', fontsize=35, fontweight='bold', color=c['feed'])
                    current_slot += 1
                continue
            
            comp, mode = self.chain[i]
            x = x_start + (current_slot * x_spacing)
            
            ax.plot([x, x], [y_feed, y_feed-0.5], color=c['feed'], lw=1.5)
            
            # --- Extract components and their parameters ---
            coupling_cap, tline_res, coupling_ind, squid_params = None, None, None, None

            if isinstance(comp, Channel):
                for sub_comp in comp.components:
                    if isinstance(sub_comp, Capacitor):
                        coupling_cap = sub_comp
                    elif isinstance(sub_comp, Resonator):
                        tline_res = sub_comp
                        if isinstance(tline_res.load, ScreenedInductor):
                            screened_ind = tline_res.load
                            if isinstance(screened_ind.primary_inductor, Inductor):
                                coupling_ind = screened_ind.primary_inductor
                            if isinstance(screened_ind.screening_device, SQUID):
                                squid_dev = screened_ind.screening_device
                                squid_params = {
                                    'M': screened_ind.M,
                                    'L_s': squid_dev.L_s,
                                    'L_j': squid_dev.L_j,
                                    'flux_lines': squid_dev.flux_lines 
                                }

            y_curr = y_feed - 0.5
            y_curr = draw_cap(x, y_curr, cap=coupling_cap)
            y_curr = draw_tline(x, y_curr, tline=tline_res)
            
            y_ind_top = y_curr
            y_curr = draw_inductor(x, y_curr, ind=coupling_ind)
            
            draw_squid(x, (y_ind_top + y_curr) / 2, squid_params=squid_params)
            
            draw_ground(x, y_curr)

            label = comp.name if hasattr(comp, 'name') else f"Channel {i+1}"
            ax.text(x, y_feed + 0.5, label, ha='center', fontweight='bold', fontsize=11)
            
            current_slot += 1

        ax.set_aspect('equal')
        ax.set_xlim(0, total_x)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

class Channel(MicrowaveDevice):
    def __init__(self, name="Channel"):
        self.name = name
        self.components = []

    def add(self, component):
        self.components.append(component)

    def Z(self, f):
        if not self.components: return float('inf') 
        return sum(comp.Z(f) for comp in self.components)
    
    def __repr__(self):
        return f"Channel({len(self.components)} comps)"

    
# HELPERS
def extract_physical_params(fit_params, load_device, v_p=1.15e8, Z0=50, Cc_in=None):
    fr, Qr, Qc = fit_params[2], fit_params[3], fit_params[4]
    omega = 2 * np.pi * fr

    Qi = 1 / (1/Qr - 1/Qc) if (Qr > 0 and Qc > 0 and (1/Qr - 1/Qc) > 1e-12) else 1e12 
    
    if Cc_in is not None:
        Cc = Cc_in
    else:
        Cc = (1 / (2 * omega * Z0)) * np.sqrt(np.pi / Qc)
    
    z_load = load_device.Z(fr)
    L_load = np.imag(z_load) / omega
    
    Xc = 1 / (omega * Cc)
    XL = omega * L_load
    
    tan_bl = Z0 * (Xc - XL) / (Z0**2 + Xc * XL)
    
    bl = np.arctan(tan_bl)
    if bl < 0: bl += np.pi 
        
    length = bl / (omega / v_p)
    return length, Cc, Qi

def generate_flux_ramp(t, f_ramp, n_phi0):
    T_ramp = 1 / f_ramp
    PHI_0 = 2.067e-15
    phi_amplitude = n_phi0 * PHI_0
    phi_ramp = (phi_amplitude / T_ramp) * (t % T_ramp)
    
    return phi_ramp

def savefig(filename):
    plt.savefig(filename+".pdf")

def find_all_peaks(f, s21,dist = 1.0e6 , showplot = False, filename = None):
    s21_flipped = s21.max() - np.abs(s21)

    # Temporary debug plot
    # plt.figure()
    # plt.plot(f, s21_flipped)
    # plt.title("Is this 'flipped' signal showing clear peaks?")
    # plt.show()

    excluded_frequencies = []

    peaks_full, _ = find_peaks(s21_flipped, prominence=0.05, distance=10)

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
    peaks = np.array(final_peaks_list, dtype=int)
    
    if(showplot):
        s21_db = 20*np.log10(np.abs(s21))
        plt.vlines(x=f[peaks], ymin=s21_db.min(), ymax=s21_db.max(), 
                color='red', linestyle='--', alpha=0.5, label='Peak Locations')

        plt.plot(f, 20*np.log10(np.abs(s21)))
        plt.ylabel("S21 (dB)")
        plt.xlabel("Freq (Hz)")
        
        if(filename != None):
            savefig(filename+"_peak_locations")
            
        plt.show()
        
    return peaks