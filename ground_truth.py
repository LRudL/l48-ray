import math
import numpy as np

def get_Ht_test1(t):
    """A test function that implements a simple case of H_t"""
    E_0 = 1
    v_t = 0.5*math.exp(-(t-2.5)**2)
    
    sigma_x = np.matrix([[0 + 0j,1 + 0j], [1 + 0j,0 + 0j]])
    
    return E_0*np.eye(2, dtype = complex) + v_t*sigma_x


def get_U_tplusdt(get_Ht, t, dt, U_t):
    """Takes: a function for calculating H at a t, a t, a dt size,
    and a value of U_t"""
    I = np.eye(2, dtype = complex)
    i = complex(0, 1)
    return np.matmul(I - i*get_Ht(t + 0.5*dt)*dt, U_t)


def naive_simulation(get_Ht, T, dt):
    t = 0
    # U_0 is always I
    U_t = np.eye(2, dtype = complex)

    Uts = []
    
    while (t<T):
        U_t = get_U_tplusdt(get_Ht, t, dt, U_t)
        Uts.append(U_t)
        t += dt
        
    return Uts
