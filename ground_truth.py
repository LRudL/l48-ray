import math
import numpy as np
import hamiltonians


def get_Ht_test1(t, E_0, b):
    """A test function that implements a simple case of H_t"""
    E_0 = 1
    v_t = 0.5 * math.exp(-(t - 2.5) ** 2)

    sigma_x = np.matrix([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]])

    return E_0 * np.eye(2, dtype=complex) + v_t * sigma_x


def get_U_tplusdt(get_Ht, t, dt, U_t):
    """
    This function calculates the unitary U after time (t+dt) given U at time t
        @params:
        get_Ht: a function for calculating the Hermitian matrix H at time t
        t: the time until which we have simulated the value of U
        dt: a small interval over which we want to understand how U changes
        U_t: the unitary U at time t

        @returns:
        U(t+dt): the value of U after a time step
    """
    I_dim = U_t.shape[0]
    I = np.eye(I_dim, dtype=complex)
    return (I - 1j * get_Ht(t + 0.5 * dt) * dt) @ U_t


def naive_simulation(get_Ht, T, dt):
    '''
    This function returns the result of euler integration over time T
        @params
        get_Ht: the Hermitian function which changes U over time
        T: the total duration over which to run the simulation
        dt: the size of the timesteps
        @returns
        U(T): ground truth value of U at the end of the simulation
        U_ts: value of U throughout the simulation
    '''
    t = 0
    # U_0 is always I
    I_dim = get_Ht(0).shape[0]
    U_t = np.eye(I_dim, dtype=complex)

    U_ts = []

    while (t < T):
        U_t = get_U_tplusdt(get_Ht, t, dt, U_t)
        t += dt
        U_ts.append(U_t)

    return U_t, U_ts

def naive_simulator(get_Ht, t, t_start, dt):
    if isinstance(get_Ht, hermitian_functions.Hamiltonian):
        get_Ht = get_Ht.at_t
    get_Ht_ = get_Ht
    if t_start != 0:
        get_Ht_ = lambda t : get_Ht(t + t_start)
    return naive_simulation(get_Ht_, t, dt)[0]
