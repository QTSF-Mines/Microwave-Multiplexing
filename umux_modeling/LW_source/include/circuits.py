import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks
from scipy.optimize import root_scalar
from umux_modeling.LW_source.include.components import *

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
    
    def draw_schematic(self, max_draw=2, use_color=True):
        num_channels = len(self.chain)
        
        if num_channels <= max_draw:
            draw_indices = list(range(num_channels))
            has_ellipsis = False
        else:
            draw_indices = [0, 1, num_channels - 1]
            has_ellipsis = True

        x_start = 2
        x_spacing = 6.5 
        num_slots = len(draw_indices) + (1 if has_ellipsis else 0)
        total_x = x_start + (num_slots * x_spacing) + 2
        
        plot_width = max(10, total_x * 0.7)
        fig, ax = plt.subplots(figsize=(plot_width, 7), facecolor='white')
        ax.set_facecolor('white')

        y_feed = 8

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
                    ax.text(x_ell, y_feed - 2.5, '. . .', ha='center', va='center', fontsize=25, fontweight='bold', color=c['feed'])
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
    
    def get_resonance(self, f_guess=None, f_sweep=None):
        if f_sweep is None:
            if f_guess is None:
                raise ValueError("Must provide either an initial f_guess or an f_sweep array.")

            f_sweep = np.linspace(f_guess - 2.55e6, f_guess + 2.5e6, 1000)
            
        Z_mags = [np.abs(self.Z(f)) for f in f_sweep]

        
        return f_sweep[np.argmin(Z_mags)]
        
    
    def __repr__(self):
        return f"Channel({len(self.components)} comps)"