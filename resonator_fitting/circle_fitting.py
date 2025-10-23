import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plot_circle(xc,yc,r,n):
    theta = np.linspace(0,2*np.pi, n)
    xs = xc + r*np.cos(theta)
    ys = yc + r*np.sin(theta)
    plt.plot(xs,ys)

def fit_circle(x_s,y_s):
    
    scale = np.mean(np.sqrt(x_s**2 + y_s**2))

    if scale < 1e-9:
        scale = 1.0
    x = x_s / scale
    y = y_s / scale

    n = len(x)
    w = x**2 + y**2

    Mw = np.sum(w)
    Mx = np.sum(x)
    My = np.sum(y)
    Mww = np.sum(w**2)
    Mxw = np.sum(x * w)
    Myw = np.sum(y * w)
    Mxx = np.sum(x**2)
    Myy = np.sum(y**2)
    Mxy = np.sum(x * y)

    M = np.array([
        [Mww, Mxw, Myw, Mw],
        [Mxw, Mxx, Mxy, Mx],
        [Myw, Mxy, Myy, My],
        [Mw,  Mx,  My,  n]
    ])

    B = np.array([
        [0, 0, 0, -2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-2, 0, 0, 0]
    ])

    eigenvalues, eigenvectors = sp.linalg.eig(M, B)


    n_min = -1
    min_eig_val = np.inf

    for i, val in enumerate(eigenvalues):
        if val.real > 0 and val.real < min_eig_val:
            min_eig_val = val.real
            n_min = i

    if n_min == -1:
        raise ValueError("Circle fit failed. No positive eigenvalue found.")

    params = eigenvectors[:, n_min].real
    A, B_param, C_param, D_param = params

    xc = -B_param / (2 * A)
    yc = -C_param / (2 * A)

    r = np.sqrt(B_param**2 + C_param**2 - 4 * A * D_param) / (2 * np.abs(A))
    
    
    xc_s = xc * scale
    yc_s = yc * scale
    r_s = r * scale

    return xc_s, yc_s, r_s