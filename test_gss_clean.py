import numpy as np
import math

def vectorized_gss(func, f_guess, bounds_width=2.55e6, tol=1.0):
    f_center = np.asarray(f_guess, dtype=float)
    a = f_center - bounds_width
    b = f_center + bounds_width
    
    invphi = (math.sqrt(5) - 1) / 2
    invphi2 = (3 - math.sqrt(5)) / 2
    
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    
    fc = func(c)
    fd = func(d)
    
    iters = 0
    while np.max(np.abs(h)) > tol:
        iters += 1
        mask = fc < fd
        
        new_b = np.where(mask, d, b)
        new_a = np.where(~mask, c, a)
        
        new_d = np.where(mask, c, new_a + invphi * (new_b - new_a))
        new_c = np.where(mask, new_a + invphi2 * (new_b - new_a), d)
        
        eval_f = np.where(mask, new_c, new_d)
        f_eval_res = func(eval_f)
        
        fc_new = np.where(mask, f_eval_res, fd)
        fd_new = np.where(mask, fc, f_eval_res)
        
        fc = fc_new
        fd = fd_new
        
        a, b, c, d = new_a, new_b, new_c, new_d
        h = b - a

    print(f"Converged in {iters} iterations")
    return 0.5 * (a + b)

def my_func(x):
    # A simple parabola centered at [10, -5, 20]
    mins = np.array([10.0, -5.0, 20.0])
    return (x - mins)**2

guess = np.array([0.0, 0.0, 0.0])
res = vectorized_gss(my_func, guess, bounds_width=50.0, tol=1e-5)
print("Result:", res)
