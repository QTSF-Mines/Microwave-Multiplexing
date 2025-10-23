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
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
```

```python
df = pd.read_csv(
    "can_resonance.csv",
    skiprows=6,          
    encoding="latin1"
)

df["Freq(Hz)"] = pd.to_numeric(df["Freq(Hz)"], errors="coerce")
df["S11 Log Mag(dB)"] = pd.to_numeric(df["S11 Log Mag(dB)"], errors="coerce")
df["S11 Phase(°)"] = pd.to_numeric(df["S11 Phase(°)"], errors="coerce")  # or your renamed version

freq = df["Freq(Hz)"].to_numpy()
s11 = df["S11 Log Mag(dB)"].to_numpy()
phase = df["S11 Phase(°)"].to_numpy()



```

```python


plt.plot(freq/1e9, s11)
plt.plot()
plt.xlabel("Freq (GHz)")
plt.ylabel("S11 Log Mag (dB)")
plt.grid(True)
plt.show()

```

```python
for val in s11:
    
half_max = (np.min(s11!= 'nan')-np.max(s11))/2 + np.min(s11)
print(s11)
np.min(s11)
print(half_max)
freq1 = freq[np.where(s11 == half_max)[0]]
print(freq1)

```

```python

```
