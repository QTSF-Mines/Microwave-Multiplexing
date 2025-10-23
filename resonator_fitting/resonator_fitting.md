---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: base
    language: python
    name: python3
---

```python

import numpy as np
import scipy as sp
import circle_fitting as cf
import resonance_fitting as rf
import all_resonances as ar
import time
import matplotlib.pyplot as plt
plt.ion()

filename = "../NIST_resonators_BenMates/frsurvey_umux300k_v6_w1_20220825.npz"

###########################################################################################
# When we take a flux-ramp survey we should save at least the following variables:
#   f_wide      : Frequencies in wide VNA sweep (Hz).                           (l)
#   s21_wide    : Complex S21 values from wide VNA sweep.                       (l)
#   fc          : Center frequencies for the flux-ramp survey (Hz).             (n)
#   ibias       : Bias current values applied to the flux-ramp coil (A).        (m)
#   fbias       : Frequency offsets in VNA sweeps for flux-ramp survey (Hz).    (p)
#   s21_fr      : Complex S21 values from flux-ramp survey sweeps.          (n x m x p)
data = np.load(filename)
f_wide = data['f_wide']
s21_wide = data['s21_wide']
fc = data['fc']
ibias = data['ibias']
fbias = data['fbias']
s21_fr = data['s21_fr']
tau = data['tau']
```

Narrow into one resonator

```python
f_min = 5.607e9
f_max = 5.6085e9

plt.plot(f_wide, 20*np.log10(s21_wide))
plt.xlim((f_min, f_max))
plt.ylabel("S21 (dB)")
plt.xlabel("Freq (GHz)")

mask = (f_wide >= f_min) & (f_wide <= f_max)
f_region = f_wide[mask]
s21_region = s21_wide[mask]




```

```python
print(f_max-f_min)
```

```python

filename = "NIST_resonators_test/"+str(int(f_min))+"_"+str(int(f_max))+"Hz"
a_fine, tau_fine, fr_fine, Qr_fine, Qc_fine, phi0_fine, chi2 = rf.refined_fit(f_region, s21_region, tau, plot_mode='condensed', filename = filename)


```

```python
f_start = 5.45e9
f_stop = 5.57e9

f_range = f_stop-f_start

# mask = (f_wide >= f_start) & (f_wide <= f_stop)
# f = f_wide[mask]
# s21 = s21_wide[mask]
f = f_wide
s21 = s21_wide


plt.plot(f*1e-9, 20*np.log10(np.abs(s21)))

plt.ylabel("S21 (dB)")
plt.xlabel("Freq (GHz)")


```

```python
ar.fit_all_resonators(f, s21, tau, showplot=True, filename="NIST_umux_all")
```
