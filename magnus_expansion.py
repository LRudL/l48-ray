import numpy as np
import math
import scipy
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
    return (0+-1j)*(E_0*np.eye(2, dtype = complex) + v_t*sigma_x)

def euler_integrator(f, dt, t, t0 = 0):
    """Simple integrator taking f, dt, t, and optionally t0 = 0,
    works on f that map from a single value to a matrix too."""
    # Create np array of locations to evaluate f at:
    locations = np.linspace(t0, t, num = int((t - t0) / dt)) + dt / 2
    # Evaluate function at locations:
    f_vals = f(locations)
    # The shape of f_vals can be either of:
    #   [N], or
    #   [N, w, h]
    # where N is the number of locations and w/h are the width/height of a matrix;
    # setting axis=0 sums only over the first dimension, i.e. N
    return np.sum(f_vals, axis=0) * dt

def euler_integrator2(f, dt, t, t0=0):
    s = 0
    t_max = t
    t = t0 + dt / 2 
    while t < t_max:
        s += f(t) * dt
        t += dt / 2
    return s

def generate_bernoulli(n):
    if n == 0:
        return 1
    else:
        x = 0
        for k in range(0, n):
            answer -= math.comb(n, k) * bernoulli(k) / (n - k + 1)

def memoizer(fn):
    stored_vals = {}
    def memoized(*args):
        # note: args is tuple, so can hash, so can use as key
        if args in stored_vals.keys():
            return stored_vals[args]
        else:
            x = fn(*args)
            stored_vals[args] = x
            return x
    return memoized

bernoulli = memoizer(generate_bernoulli)

def sqbrackets(A, B):
    # what is written [A, B] in the Wikipedia page for Magnus expansion
    return A @ B - B @ A

def ad(k, Omega, A):
    if k == 0:
        return A
    else:
        return sqbrackets(Omega, ad(k - 1, Omega, A))

def magnus(get_At, t, k=1, integrator = euler_integrator2, integrator_dt = 0.01):
    """Assume we have a system following  U'(t) = A(t) U(t);
    use the Magnus expansion approach to estimate U(t)"""
    U_0 = np.eye(2, dtype=complex)

    Omega_t = np.zeros((2,2), dtype=complex)

    Omega_t_ks = []
    
    for ki in range(1, k+1):
        if ki == 1:
            Omega_t_ks.append(integrator(get_At, integrator_dt, t, 0))            
        elif ki == 2:
            Omega_t_ks.append(
                0.5 * integrator(lambda t1 : integrator(lambda t2 :
                                                        sqbrackets(get_At(t1), get_At(t2)),
                                                        integrator_dt, t1, 0),
                                 integrator_dt, t, 0))
        elif ki == 3:
            Omega_t_ks.append(
                1 / 6 * integrator(lambda t1 : integrator(lambda t2 : integrator(lambda t3 :
                                                                                 sqbrackets(
                                                                                     get_At(t1),
                                                                                     sqbrackets(
                                                                                         get_At(t2),
                                                                                         get_At(t3)))
                                                                                 + sqbrackets(
                                                                                     get_At(t3),
                                                                                     sqbrackets(
                                                                                         get_At(t2),
                                                                                         get_At(t1))),
                                                                                 integrator_dt, t2, 0),
                                                          integrator_dt, t1, 0),
                                   integrator_dt, t, 0))
                                                                                      
        else:
            raise Exception("Magnus not implemented for k > 3")

    Omega_t = np.sum(Omega_t_ks, axis=0)
    
    answer = scipy.linalg.expm(Omega_t) @ U_0
    return answer

# print("---")
# print(magnus(get_At, 2, k=1))