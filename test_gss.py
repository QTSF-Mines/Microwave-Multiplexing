import numpy as np
import math

def vectorized_gss(func, f_guess, bounds_width=2.55e6, tol=1.0):
    a = f_guess - bounds_width
    b = f_guess + bounds_width
    
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
        
        new_fd = np.where(mask, fc, 0.0) 
        new_fc = np.where(~mask, fd, 0.0) 
        
        eval_f = np.where(mask, new_c, new_d)
        f_eval_res = func(eval_f)
        
        fc = np.where(mask, f_eval_res, new_fc)
        fd = np.where(~mask, new_fd, f_eval_res)
        
        a, b, c, d = new_a, new_b, new_c, new_d
        h = b - a

    print(f"Converged in {iters} iterations")
    return 0.5 * (a + b)

# Test with a quadratic function
true_mins = np.array([5.0, -2.0, 10.0, 0.0])
def test_func(x):
    return (x - true_mins)**2

guesses = np.array([0.0, 0.0, 0.0, 0.0])
res = vectorized_gss(test_func, guesses, bounds_width=20.0, tol=1e-5)
print("True mins:", true_mins)
print("Found mins:", res)
