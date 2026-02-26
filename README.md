# Microwave-Multiplexing
Houses all scripts related to the development of RF-SQUID multiplexers at Colorado School of Mines. Includes scripts for resonance fitting.


## Resonator Fitting
See [[resonator_fitting.ipynb]]. Based on the resonator fitting scheme in appendix E of Jiansong Gao's Cal Tech thesis. This first fits to the circle and the phase data to find the initial parameters then uses these as initial guesses for the full least-squares fitting. This follws the scheme in Ben Mates' scripts (NIST_resonators_BenMates), with added least squares fitting. 


The two relevent scripts are:

- circle_fitting.py
- resonance_fitting.py

## uMux Modeling

Modeling of the analytic current response of microwave multiplexer resonators. This extends to flux ramp modulation. Demodulation of pulses as well. 

