import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.patches as patches

### ENTITIES ###
class MicrowaveDevice:
    def Z(self, f): raise NotImplementedError
    
    def get_abcd(self, f, mode='series'):
        z = self.Z(f)
        if mode == 'series':
            return np.array([[1, z], [0, 1]])
        else: 
            return np.array([[1, 0], [1/z, 1]])
        
    #SQUID Entity: Acts as a non-linear inductor, simply calculates this inductance based off of input flux (given by mutual Inductance) Componants: Mutual Inductance, Inductance. Should be flexible to include multiple inputs with different couplings (Flux Ramp, TES ...) Params: SQUID params, Inputs: Flux, Outputs: Inductance

class SQUID(MicrowaveDevice):
    """
    Models a SQUID as a flux-tunable inductor. Its impedance is determined
    by the total external magnetic flux applied to its loop.

    Args:
       I_c (float): The critical current of the Josephson junction (in Amps).
       L_loop (float): The geometric loop inductance of the SQUID (in Henrys).
       phi_ext (float, optional): The initial external flux bias (in Webers). Defaults to 0.0.
    """
    def __init__(self, I_c, L_loop, phi_ext=0.0):
        self.I_c = I_c
        self.L_loop = L_loop
        self.phi_ext = phi_ext
        self.PHI_0 = 2.067e-15  # Flux quantum
    
    def set_flux(self,phi):
        self.phi_ext = phi

    def get_L(self):
        cos_term = np.cos(np.pi * self.phi_ext / self.PHI_0)

        if abs(cos_term) < 1e-9:
            return float('inf')

        L_j = self.PHI_0 / (2 * np.pi * self.I_c * abs(cos_term))
        
        return self.L_loop + L_j
        
    def Z(self, f):
        omega = 2 * np.pi * f
        return 1j * omega * self.get_L()

#Resonator Entity: A resonant frequency function in a transmission. Resonant frequency changes based on inductive load. : Componants: Resonance Params, Will be coupled to an inductor componant, which is coupled to a SQUID. Params: Resonator Params, Inputs: Inductance, Outputs: Resonant Params, S21
    
class Resonator(MicrowaveDevice):
    def __init__(self, length, load, v_p=1.15e8, Qi=100000, Z0=50):
        self.length = length
        self.load = load
        self.v_p = v_p
        self.Qi = Qi
        self.Z0 = Z0
        
    def Z(self, f):
        if f == 0: return float('inf')
        
        # Pure transmission line physics (no phenomenological fr inputs here!)
        beta = 2 * np.pi * f / self.v_p
        alpha = beta / (2 * self.Qi)
        gamma = alpha + 1j * beta
        
        zl_val = self.load.Z(f) 
        
        # Terminated waveguide impedance
        num = zl_val + self.Z0 * np.tanh(gamma * self.length)
        den = self.Z0 + zl_val * np.tanh(gamma * self.length)
        
        return self.Z0 * (num / den)
    
def extract_physical_params(fit_params, load_device, v_p=1.15e8, Z0=50):
    fr, Qr, Qc = fit_params[2], fit_params[3], fit_params[4]
    omega = 2 * np.pi * fr

    Qi = 1 / (1/Qr - 1/Qc) if (Qr > 0 and Qc > 0 and (1/Qr - 1/Qc) > 1e-12) else 1e12 
    Cc = (1 / (2 * omega * Z0)) * np.sqrt(np.pi / Qc)
    
    z_load = load_device.Z(fr)
    L_load = np.imag(z_load) / omega
    
    Xc = 1 / (omega * Cc)
    XL = omega * L_load
    
    # Exact series resonance condition mathematically forces the dip to align
    tan_bl = Z0 * (Xc - XL) / (Z0**2 + Xc * XL)
    
    bl = np.arctan(tan_bl)
    if bl < 0: bl += np.pi 
        
    length = bl / (omega / v_p)
    return length, Cc, Qi
    
class Inductor(MicrowaveDevice):
    def __init__(self, L):
        self.L = L
        
    def Z(self, f):
        omega = 2 * np.pi * f
        return 1j * omega * self.L

#Entity: Screened Inductor (Takes input inductor and screens it with other device with a mutual inductance M)
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

        if abs(z_screening) < 1e-12: return complex(0, float('inf'))
        if abs(z_screening) == float('inf'): return z_primary

        z_reflected = (omega * self.M)**2 / z_screening
        return z_primary + z_reflected
        

class Capacitor(MicrowaveDevice):
    def __init__(self, C):
        self.C = C

    def Z(self, f):
        if f == 0:
            return float('inf') 
        omega = 2 * np.pi * f
        return -1j / (omega * self.C)
    
class Terminator(MicrowaveDevice):
    def __init__(self, R):
        self.R = R

    def Z(self, f):
        return complex(self.R,0)


### SYSTEMS ###

#VNA System
class VNA_Simulator:
    def __init__(self, z0=50):
        self.z0 = z0
        self.chain = []

    def add(self, component, mode='series'):
        self.chain.append((component, mode))

    def get_s_matrix(self, f):
        if not np.isscalar(f):
            s_params = np.array([self.get_s_matrix(freq) for freq in f])
            return s_params.transpose(1, 2, 0)

        abcd = np.identity(2, dtype=complex)
        for comp, mode in self.chain:
            abcd = abcd @ comp.get_abcd(f, mode)

        A, B, C, D = abcd.flatten()
        z0 = self.z0
        denom = A + B/z0 + C*z0 + D
        s11 = (A + B/z0 - C*z0 - D) / denom
        s12 = (2 * (A*D - B*C)) / denom
        s21 = 2 / denom
        s22 = (-A + B/z0 - C*z0 + D) / denom
        return np.array([[s11, s12], [s21, s22]])
    
    def draw_schematic(self, max_draw=4, use_color=True):
        """Draws a textbook-style circuit diagram, smartly collapsing large arrays and optionally coloring components."""
        
        num_channels = len(self.chain)
        
        if num_channels <= max_draw:
            draw_indices = list(range(num_channels))
            has_ellipsis = False
        else:
            draw_indices = [0, 1, 2, num_channels - 1]
            has_ellipsis = True

        x_start = 2
        x_spacing = 4
        num_slots = len(draw_indices) + (1 if has_ellipsis else 0)
        total_x = x_start + (num_slots * x_spacing) + 2
        
        plot_width = max(10, total_x * 0.7)
        fig, ax = plt.subplots(figsize=(plot_width, 7), facecolor='white')
        ax.set_facecolor('white')

        y_feed = 8

        # --- COLOR PALETTE ---
        if use_color:
            c = {
                'feed': '#2c3e50',       # Dark Slate
                'cap': '#27ae60',        # Green
                'tline_e': '#2980b9',    # Blue edge
                'tline_f': '#ebf5fb',    # Light blue face
                'ind': '#d35400',        # Rust Orange
                'squid': '#8e44ad',      # Purple
                'res': '#c0392b',        # Red
                'src': '#16a085',        # Teal
                'gnd': '#7f8c8d'         # Grey
            }
        else:
            c = {k: 'k' for k in ['feed', 'cap', 'tline_e', 'ind', 'squid', 'res', 'src', 'gnd']}
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
            
            rect = patches.Rectangle((x_sq, y_sq_bot), sq_w, sq_h, fill=False, edgecolor=c['squid'], lw=1.5)
            ax.add_patch(rect)
            
            x_jj = x_sq + sq_w
            jj_size = 0.08 
            ax.plot([x_jj-jj_size, x_jj+jj_size], [y_center-jj_size, y_center+jj_size], color=c['squid'], lw=1.5)
            ax.plot([x_jj-jj_size, x_jj+jj_size], [y_center+jj_size, y_center-jj_size], color=c['squid'], lw=1.5)
            
            if squid_params:
                m_val_str = f"{squid_params['M'] * 1e12:.1f} pH"
                ls_val_str = f"{squid_params['L_loop'] * 1e12:.1f} pH"
                ic_val_str = f"{squid_params['I_c'] * 1e6:.1f} µA"
                ax.text(x_left -1, y_center, f'$M_c$={m_val_str}', ha='center', va='center', fontsize=9, color='k')
                ax.text(x_sq + sq_w / 2, y_sq_bot + 0.15, f'$L_S$={ls_val_str}', ha='center', fontsize=9, color=c['squid'])
                ax.text(x_jj + 0.25, y_center+0.3, f'$I_c$={ic_val_str}', ha='center', va='center', fontsize=9, color=c['squid'])
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
            
            # Tap off feedline
            ax.plot([x, x], [y_feed, y_feed-0.5], color=c['feed'], lw=1.5)
            
            # --- Extract components and their parameters ---
            coupling_cap, tline_res, coupling_ind, squid_params = None, None, None, None

            if isinstance(comp, Channel):
                # This part makes assumptions about the structure of a 'Channel'
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
                                    'L_loop': squid_dev.L_loop,
                                    'I_c': squid_dev.I_c
                                }

            # Construct the resonator leg
            y_curr = y_feed - 0.5
            y_curr = draw_cap(x, y_curr, cap=coupling_cap)
            y_curr = draw_tline(x, y_curr, tline=tline_res)
            
            y_ind_top = y_curr
            y_curr = draw_inductor(x, y_curr, ind=coupling_ind)
            
            # Add SQUID
            draw_squid(x, (y_ind_top + y_curr) / 2, squid_params=squid_params)
            
            # Ground the resonator leg
            draw_ground(x, y_curr)

            # Label the channel
            label = comp.name if hasattr(comp, 'name') else f"Channel {i+1}"
            ax.text(x, y_feed + 0.5, label, ha='center', fontweight='bold', fontsize=11)
            
            current_slot += 1

        ax.set_aspect('equal')
        ax.set_xlim(0, total_x)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        

#Channel System: Can simply add in line a resonator, inductor, SQUID, and input current
class Channel(MicrowaveDevice):
    def __init__(self, name="Channel"):
        self.name = name
        self.components = []

    def add(self, component):
        self.components.append(component)

    def Z(self, f):
        if not self.components:
            return float('inf') 
        return sum(comp.Z(f) for comp in self.components)
    
    def __repr__(self):
        # Helps the VNA display identify the channel's contents
        return f"Channel({len(self.components)} comps)"

#Readout System: Can couple many channels, stack their response, demodulate. Probably the most intensive part.


### HELPERS ###

    #Flux Ramp Generator: At first a simple current function gnerator, but could couple in resonant frequency as tone tracking and such.

    #Pulse Generator

    #TES Coupler
    
    #Resonance Loader
