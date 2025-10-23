import numpy as np
import scipy.optimize as op

import matplotlib.pyplot as plt
plt.ion()

# NOTE: lambda is a reserved word in python, so we use lamb instead.

Phi0 = 2.068e-15    # Magnetic flux quantum
Z1 = 50.0           # 50 Ohm CPW resonator
Ls = 22.4e-12       # SQUID self-inductance (from FastHenry simulation)


def phi_of_phie(phie, lamb):
    """ Solves the self-consistent magnetic flux in the SQUID.

    The magnetic flux in an unshunted rf-SQUID obeys:

    Phi = Phi_external - lambda*sin(2*pi*Phi/Phi0)

    where the second term describes flux applied to the SQUID by its own circulating current.

    This is Kepler's Equation (https://en.wikipedia.org/wiki/Kepler%27s_equation), which is a
    transcendental equation without a closed-form inverse. We therefore solve it numerically,
    assuming lambda < 1. If lambda > 1 the inversion is multi-valued and is not solved with
    this code.

    Args:
        phie      : Externally applied flux in radians (phie = 2*pi*Phi_e/Phi0).
        lamb      : Lambda parameter defined as lambda = 2*pi*Ic*Ls/Phi0. (also known as beta_L)
    
    Returns:
        phi       : Total flux in the SQUID in radians (phi = 2*pi*Phi/Phi0).
    """
    phiguess = phie
    phi = op.fsolve(lambda phi : phi + lamb*np.sin(phi) - phie, phiguess)   # Numerically solve Kepler Eq.

    return phi


def f0_of_I(I,I0,Min,Mc,fb,lamb):
    """ Solves for the resonance frequency as a function of input parameters

    This model comes from Ben Mates' thesis (Eq. 2.16 and 2.50):

    f0(phi) - fb = (4 f0^2 / Z1) * (Mc^2 / Ls) * (lambda cos(phi)) / (1 + lambda cos(phi))

    Args:
        I         : Current applied to the SQUID input (or flux-ramp) coil (A).
        I0        : Offset flux in units of input current (A).
        Min       : Mutual inductance of input (or flux-ramp) coil into the SQUID (H).
        Mc        : Mutual inductance of the resonator into the SQUID (H).
        fb        : "Base" frequency of the resonator, not loaded by the rf-SQUID (Hz).
        lamb      : Lambda parameter defined as lambda = 2*pi*Ic*Ls/Phi0. (also known as beta_L)
    
    Returns:
        f0        : Resonance frequency loaded by the rf-SQUID (Hz).
    """
    phie = 2*np.pi*(I+I0)*Min/Phi0
    phi = phi_of_phie(phie,lamb)
    f0 = fb + (4*(fb**2)/Z1) * (Mc**2)/Ls * lamb*np.cos(phi)/(1 + lamb*np.cos(phi))

    return f0


def short_signal_periodicity(x):
    """ Attempts to guess the periodicity of a signal containing between 1 and 2 periods

    To guess the periodicity of a short signal, we stretch a window between 1/2 and all of
    the data, testing which window length minimizes the power in higher harmonics of the
    fundamental.

    Because we know that the SQUID response has some power not in the fundamental, we
    accept power in the first 3 Fourier bins.

    We also report the phase of the fundamental frequency for the identified period.

    Args:
        x       : Signal whose periodicity to guess.
    
    Returns:
        ppp     : Number of points per period.
        poff    : Offset points to the zero phase of the oscillation.
    """
    hhp = np.zeros(len(x)//2)
    phases = np.zeros(len(x)//2)

    for n in range(len(hhp)):
        m = len(x) - n                              # Shrink the window by n points
        X = np.fft.rfft(x[:m]) / m                  # Calculate Fourier spectrum
        hhp[n] = np.sum(np.abs(X[4:])**2)           # Add up the power in the higher bins
        phases[n] = np.angle(X[1])                  # Record phase of fundamental
    
    ppp = len(x) - np.argmin(hhp)                   # Identify window with least power in higher bins
    poff = ppp*phases[np.argmin(hhp)]/(2*np.pi)

    return ppp,poff


def guess_params(I,f0):
    """ Calculates good initial guesses for the fit.

    The main danger with the construction of these guesses is f0 data with either extreme
    noise or outlier values from failed resonance fits; in either case these guesses can
    be far off the true values and lead to poor lambda fitting.

    Args:
        I         : Current applied to the SQUID input (or flux-ramp) coil (A).
        f0        : Measured resonance frequency at the different input currents (Hz).
    
    Returns:
        I0guess   : Guess at offset flux in units of input current (A).
        Minguess  : Guess at Min (or Mfr) coupling (H).
        Mcguess   : Guess at Mc between the resonator and the SQUID (H).
        fbguess   : Guess at "base" frequency of the resonator, not loaded by the rf-SQUID (Hz).
        lambguess : Guess at lambda parameter.
    """
    lambguess = 0.33        # Typical design target
    fbguess = np.mean(f0)
    Mcguess = np.sqrt( (np.max(f0)-np.min(f0))*Z1*Ls*(1-lambguess**2) / (8*lambguess*fbguess**2) )

    ppp,poff = short_signal_periodicity(f0)     # Assumes between 1 and 2 Phi0 of data
    Minguess = Phi0 / (ppp*(I[1] - I[0]))
    I0guess = I[0] + poff*(I[1]-I[0])

    return I0guess,Minguess,Mcguess,fbguess,lambguess


def fit_lambda(I,f0,showplot=False):
    """ Fits resonance frequency data for lambda and other parameters.
    
    Fits the measured resonance frequency vs. current to a theoretical model to extract
    useful parameters like: lambda, Min, Mc, and fb.

    Args:
        I         : Current applied to the SQUID input (or flux-ramp) coil (A).
        f0        : Measured resonance frequency at the different input currents (Hz).
        showplot  : Show a plot of the fit for user inspection.
    
    Returns:
        I0fit     : Fit for offset flux in units of input current (A).
        Minfit    : Fit for Min (or Mfr) coupling (H).
        Mcfit     : Fit for Mc between the resonator and the SQUID (H).
        fbfit     : Fit for "base" frequency of the resonator, not loaded by the rf-SQUID (Hz).
        lambfit   : Fit for lambda parameter.
    """

    guessparams = guess_params(I,f0)
    popt, pcov = op.curve_fit(f0_of_I, I, f0, guessparams)
    
    I0fit,Minfit,Mcfit,fbfit,lambfit = popt
        
    if showplot:
        ti = np.linspace(np.min(I),np.max(I),100)
        f0fit = f0_of_I(ti,I0fit,Minfit,Mcfit,fbfit,lambfit)
        plt.figure(999)
        plt.subplot(2,1,1)
        plt.cla()
        plt.plot(I*1e6,(f0-fbfit)/1e3,'ob')
        plt.plot(ti*1e6,(f0fit-fbfit)/1e3,'--r')
        plt.xlabel('Current (uA)')
        plt.xlim(np.min(I)*1e6,np.max(I)*1e6)
        plt.ylabel('Frequency Shift (kHz)')
        plt.draw()
        plt.subplot(2,1,2)
        plt.cla()
        plt.plot(I*1e6,(f0_of_I(I,I0fit,Minfit,Mcfit,fbfit,lambfit)-f0fit)/1e3,'or')
        plt.xlabel('Current (uA)')
        plt.xlim(np.min(I)*1e6,np.max(I)*1e6)
        plt.ylabel('Residuals (kHz)')
        plt.draw()
    
    return I0fit,Minfit,Mcfit,fbfit,lambfit