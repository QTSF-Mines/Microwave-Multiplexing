import numpy as np
import scipy.optimize as op
import scipy.signal as sig

import matplotlib.pyplot as plt
plt.ion()

def circlefit(x,y,showplot=False):
    """ Algebraic circle fit by Taubin

    Args:
        x,y         : x- and y-coordinates of points to fit.
    
    Returns:
        xc,yc       : x- and y-coordinates of best-fit circle center.
        r           : Radius of best-fit circle.

    Fit algorithm based off:
        G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                    Space Curves Defined By Implicit Equations, With
                    Applications To Edge And Range Image Segmentation",
        IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
    
    NOTE: This is actually not a true fit in a least-squares sense, but it
    is a very good approximation.
    """

    # Force inputs into column vectors
    npts = len(x)
    x1 = np.mean(x)
    y1 = np.mean(y)
    xtemp = x.reshape((npts,1)) - x1
    ytemp = y.reshape((npts,1)) - y1

    z = xtemp**2 + ytemp**2
    zmean = np.mean(z)
    z1 = (z - zmean) / (2*np.sqrt(zmean))

    (U,S,V) = np.linalg.svd(np.hstack((z1,xtemp,ytemp)),0)

    A = V[2,:]
    A[0] = A[0] / (2*np.sqrt(zmean))
    A = np.append(A,-zmean*A[0])
    xc = -(A[1])/A[0]/2 + x1
    yc = -(A[2])/A[0]/2 + y1
    r = np.sqrt(A[1]*A[1] + A[2]*A[2] - 4*A[0]*A[3])/np.abs(A[0])/2

    if showplot:
        plt.figure()
        plt.plot(x,y,'.k')
        thetatemp = np.linspace(0,2*np.pi,1000)
        plt.plot(xc+r*np.cos(thetatemp),yc+r*np.sin(thetatemp),'--r')
        plt.axis('equal')

    return xc,yc,r


def fit_resonance(f,s21,showplot=False):
    """ Fit a rotated Lorentzian model to transmission data

    We follow the approach laid out in Appendix E of Jiansong Gao's thesis: first
    fitting the complex transmission to a circle, then fitting the angle on that
    circle to an arctangent vs. frequency. We actually omit the final fit to the
    full model of Eq. E.1 because we have found it unneccessary after the first
    two fits.

    Args:
        f           : Frequency values of data set.
        s21         : S21 complex transmission parameters (cable-delay compensated).
    
    Returns:
        f0fit       : Fit to resonance frequency.
        Qcfit       : Fit to coupling Q.
        Qifit       : Fit to internal Q.

    NOTE: It is critically important that the S21 data input to this function
    already have the cable delay removed, or else it will fail to fit accurately.
    """

    # Estimate resonance frequency (crude)
    f0gind = np.argmin(np.abs(s21))
    f0g = f[f0gind]

    # Estimate bandwidth (crude)
    tempz = s21*np.exp((-1j)*np.angle(s21[f0gind]))
    fhbw1gind = np.argmin(np.imag(tempz))
    fhbw2gind = np.argmax(np.imag(tempz))
    bwg = f[fhbw2gind] - f[fhbw1gind]
    if bwg <= 0:
        print('Failed to estimate bandwidth.')
        return (0,0,0)
    nhbw = int(bwg/(2*(f[1]-f[0])))   # Number of points per half-bandwidth

    # Fit circle to data points between half-bandwidth points
    x = np.real(s21[fhbw1gind:fhbw2gind])
    y = np.imag(s21[fhbw1gind:fhbw2gind])
    (xc,yc,r) = circlefit(x,y,showplot=False)

    # Normalize by off-resonance value
    offres = np.sqrt(xc**2 + yc**2) + r
    xc = xc / offres
    yc = yc / offres
    r = r / offres
    s21 = s21 / offres

    if showplot:
        plt.figure()
        plt.subplot(1,3,1)
        plt.plot(0,0,'+k')
        plt.plot(np.real(s21),np.imag(s21),'.k')
        plt.plot(np.real(s21[fhbw1gind]),np.imag(s21[fhbw1gind]),'*y')
        plt.plot(np.real(s21[fhbw2gind]),np.imag(s21[fhbw2gind]),'*y')
        plt.plot(np.real(s21[f0gind]),np.imag(s21[f0gind]),'*g')
        plt.plot(xc,yc,'+r')
        thetatemp = np.linspace(0,2*np.pi,1000)
        plt.plot(xc+r*np.cos(thetatemp),yc+r*np.sin(thetatemp),'--r')
        plt.axis('equal')
        plt.xlabel("Real(S21)")
        plt.ylabel("Imag(S21)")

    # Rotate and translate circle to center of complex plane
    zc = xc + 1j*yc
    zcentered = (s21 - zc) * np.exp(1j * (np.pi - np.angle(zc)))
    theta = np.unwrap(np.angle(zcentered))
    if np.mean(theta) < -np.pi:     # Correct for extreme Fano cases where initial angle is past pi
        theta = theta + 2*np.pi

    # Find point of fastest change around the circle
    dtheta = sig.savgol_filter(theta,window_length=2*int(nhbw/4)+1,polyorder=3,deriv=1)   # Differentiating filter
    f0gind = np.argmax(-dtheta)
    f0g = f[f0gind]                                 # New f0 guess at steepest slope of angle
    theta0g = theta[f0gind]                         # Guess at rotation angle
    Qg = -dtheta[f0gind] * f0g / (4*(f[1]-f[0]))    # Guess at Q-factor

    # Define fit model for angle dependence on frequency
    def arctanmodel(f, f0, theta0, Q):
        return (theta0 + 2*np.arctan(2*Q*(1 - f/f0)))
         
    # Fit angle on resonance circle to arctan to get more accurate f0 and Q
    try:
        popt, pcov = op.curve_fit(arctanmodel, f, theta, [f0g,theta0g,Qg])
    except RuntimeError:
        return (0,0,0)
    f0fit = popt[0]
    theta0fit = popt[1]
    Qfit = popt[2]

    theta0fit = (theta0fit+np.pi) % (2*np.pi) - np.pi # Reduce to -pi to pi range

    if showplot:
        plt.subplot(1,3,2)
        plt.plot(f,theta,'.k')
        plt.plot(f,arctanmodel(f,f0fit,theta0fit,Qfit),'--r')
        plt.xlabel("Frequency")
        plt.ylabel("Theta (rad)")

    # Calculations to extract Qc and Qi
    zf0 = zc + r*np.exp(1j*(np.pi + np.angle(zc) + theta0fit))
    zd = 2*(zf0 - zc)
    zinf = zf0 - zd
    Qcfit = np.abs(zinf)/(2*r) * Qfit
    Qifit = 1/(1/Qfit - 1/Qcfit)
    l = np.real(zf0*np.conj(zinf)) / np.abs(zinf)**2
    Qifit = Qfit/l      # Correction to Qi from above that accounts for model imperfection

    if showplot:
        plt.subplot(1,3,3)
        ftemp = np.linspace(f[0],f[-1],1000)
        s21fit = zinf * (1 - ((Qfit/Qcfit)*np.exp(1j*np.angle(-zd/zinf)))/(1 + 2j*Qfit*((ftemp-f0fit)/f0fit)))
        plt.plot(f,np.abs(s21),'.k')
        plt.plot(ftemp,np.abs(s21fit),'--r')
        plt.xlabel("Frequency")
        plt.ylabel("|S21|")

    return f0fit,Qcfit,Qifit