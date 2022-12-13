import numpy as np
import math
from scipy import linalg

def get_At(t): # test function!
    E_0 = 1
    v_t = 0.5*np.exp(-(t-2.5)**2)
    
    sigma_x = np.array([[0 + 0j,1 + 0j], [1 + 0j,0 + 0j]])
    if isinstance(t, np.ndarray):
        # to make np broadcasting work when t is a 1D array
        sigma_x = np.broadcast_to(sigma_x, (t.shape[0],) + (2, 2))
        v_t = np.expand_dims(np.expand_dims(v_t, 1), 2)

    # print(v_t.shape)
    # print(sigma_x.shape)
    
    return E_0*np.eye(2, dtype = complex) + v_t*sigma_x

def euler_integrator(f, dt, t, t0 = 0):
    """Simple integrator taking f, dt, t, and optionally t0 = 0,
    works on f that map from a single value to a matrix too"""
    # Create np array of locations to evaluate f at:
    locations = np.linspace(t0, t, num = int((t - t0) / dt)) + dt / 2
    # Evaluate function at locations:
    f_vals = f(locations)
    # The same of f_vals can be either of:
    #   [N], or
    #   [N, w, h]
    # where N is the number of locations and w/h are the width/height of a matrix;
    # setting axis=0 sums only over the first dimension, i.e. N
    return np.sum(f_vals, axis=0) * dt

def magnus(get_At, t, k=1, integrator = euler_integrator, integrator_dt = 0.01):
    """Assume we have a system following  U'(t) = A(t) U(t);
    use the Magnus expansion approach to estimate U(t)"""
    U_0 = np.eye(2, dtype=complex)

    Omega_t = np.zeros((2,2), dtype=complex)

    for ki in range(1, k+1):
        if ki == 1:
            Omega_t += integrator(get_At, integrator_dt, t, 0)
        else:
            raise Exception("Magnus expansion for k>1 not yet implemented")

    # print(Omega_t)
    
    answer = scipy.linalg.expm(Omega_t) @ U_0
    return answer
