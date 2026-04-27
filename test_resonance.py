import numpy as np
import matplotlib.pyplot as plt
from umux_modeling.LW_uMUX_src.source.include.components import *
from umux_modeling.LW_uMUX_src.source.include.circuits import *

# Create a test SQUID and Channel
squid = SQUID(L_j=100e-12, L_s=20e-12)
squid.add_flux_line("flux", M=10e-12)
ind = Inductor(L=1e-9)
scr_ind = ScreenedInductor(ind, squid, M=100e-12)
res = Resonator(length=0.01, load=scr_ind, Z0=50)
cap = Capacitor(C=10e-15)

channel = Channel("Test")
channel.add(cap)
channel.add(res)

# Sweep flux
fluxes = np.linspace(-100e-6, 100e-6, 100)
squid.flux_lines["flux"].I = fluxes # This sets it to an array!

f_guess = 5e9
# Test original method logic
# (Wait, original method relies on minimize_scalar, but our vectorized method replaced it)
resonances = channel.get_resonance(f_guess=f_guess)

np.save("resonances_test.npy", resonances)
print(resonances)
