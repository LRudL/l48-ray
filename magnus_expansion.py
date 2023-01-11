import numpy as np
import math
import scipy
import hermitian_functions
from scipy import linalg
from scipy import special

import ground_truth
from utils import fidelity


def rbf(x, y):
    return np.exp(-(x - y) ** 2 / 2)


def get_rbf(s, A):
    return lambda x, y: A * np.exp(- 1 / (2 * s) * (x - y) ** 2)


def get_At(t):  # test function!
    E_0 = 1
    v_t = 0.5 * np.exp(-(t - 2.5) ** 2)

    sigma_x = np.array([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]])
    if isinstance(t, np.ndarray):
        # to make np broadcasting work when t is a 1D array
        sigma_x = np.broadcast_to(sigma_x, (t.shape[0],) + (2, 2))
        v_t = np.expand_dims(np.expand_dims(v_t, 1), 2)

    # print(v_t.shape)
    # print(sigma_x.shape)
    return (0 + -1j) * (E_0 * np.eye(2, dtype=complex) + v_t * sigma_x)


def euler_integrator(f, dt, t, t0=0):
    """Simple integrator taking f, dt, t, and optionally t0 = 0,
    works on f that map from a single value to a matrix too."""
    # Create np array of locations to evaluate f at:
    locations = np.linspace(t0, t, num=int((t - t0) / dt)) + dt / 2
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
        t += dt
    return s


def sqbrackets(A, B):
    # what is written [A, B] in the Wikipedia page for Magnus expansion
    return A @ B - B @ A


def segmented_handler(callback, t, segment_margin, sample, verbose):
    max_dt = math.pi / segment_margin / np.linalg.norm(sample, ord='fro')
    Uts = []
    t_start = 0
    while t_start < t:
        t_end = min(t_start + max_dt, t)
        Ut = callback(t_start, t_end)
        Uts.append(Ut)
        t_start += max_dt
    if verbose:
        # for i, Ut in enumerate(Uts):
        # print(f"Segment matrix {i}:")
        # print(Ut)
        print(f"Divided section into {len(Uts)} segments")
    prod = np.eye(Uts[0].shape[0])
    for Ut in Uts:
        prod = Ut @ prod
    return prod


def magnus(
        get_Ht_, t, k=1, integrator=euler_integrator2, integrator_dt=0.01, dt=None,
        t_start=0, segmented=False, segment_margin=3,
        verbose: bool = False
):
    """Assume we have a system following  U'(t) = A(t) U(t);
    use the Magnus expansion approach to estimate U(t)"""

    print(
        f"Calling magnus with k={k}, integrator_dt={integrator_dt}, dt={dt}, segmented={segmented}, range={t_start} to {t}")

    if isinstance(get_Ht_, hermitian_functions.ConstantMatrixHermitian):
        get_Ht_ = get_Ht_.at_t
    if dt is not None:
        integrator_dt = dt
    get_Ht = get_Ht_
    if t_start != 0:
        get_Ht = lambda t: get_Ht_(t + t_start)
        t = t - t_start
    # from this point onwards, it is as if the range is [0, t],
    # even if it originally was [t_start, t]

    if segmented:
        sample = -(0 + 1j) * get_Ht(t / 2)

        callback = lambda t_start, t: magnus(
            get_Ht, t, k, integrator, integrator_dt, integrator_dt,
            t_start, False, segment_margin,
            verbose
        )

        return segmented_handler(callback, t, segment_margin, sample, verbose=True)

    n = get_Ht(0).shape[0]

    U_0 = np.eye(n, dtype=complex)

    get_At = lambda t: (0 - 1j) * get_Ht(t)

    Omega_t_ks = []

    for ki in range(1, k + 1):
        if ki == 1:
            Omega_t_ks.append(integrator(get_At, integrator_dt, t, 0))
        elif ki == 2:
            Omega_t_ks.append(
                0.5 * integrator(lambda t1: integrator(lambda t2:
                                                       sqbrackets(get_At(t1), get_At(t2)),
                                                       integrator_dt, t1, 0),
                                 integrator_dt, t, 0))
        elif ki == 3:
            Omega_t_ks.append(
                1 / 6 * integrator(lambda t1: integrator(lambda t2: integrator(lambda t3:
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
            print("The final value of Omega3:")
            print(Omega_t_ks[-1])
            print("------------------------------------------")

        else:
            raise Exception("Magnus not implemented for k > 3")

    if verbose:
        print("Omega matrices:")
        for k, omega_k in enumerate(Omega_t_ks):
            print(f'Omega {k + 1}:\n{omega_k}')

    # print(Omega_t_ks)
    Omega_t = np.sum(Omega_t_ks, axis=0)
    # print(Omega_t)

    answer = scipy.linalg.expm(Omega_t) @ U_0
    return answer


def analytic_magnus(
        components, t, rbf_scale=1, rbf_C=1, k=1, tstar_dt=0.01, dt=None,
        segmented=False, segment_margin=3,
        t_start=0,
        verbose=False
):
    """Assume we have a system following  U'(t) = -i H(t) U(t).
    Assume:
    - the function is some sort of pulse that can be
    approximated as a Gaussian wave
    - H(t) = H_0 + get_vt(t) V, and components is of the form
      (H_0, get_vt, V), i.e. the same format that
      .get_components on ConstantMatrixHermitian returns
    - we are modelling v(t) as a GP
    - the GP has an RBF covariance function, where the RBF is of form
      rbf_C * np.exp(- 1 / (2 * s) * (x1 - x2)^2)
    - in modelling the GP, we sample get_vt at intervals of tstar_dt
    - we are integrating up to term k of Magnus
    
    IF MODIFYING SIGNATURE: note that for segmentation to work properly, you must
    also pass additional things into the definition of the callback variable below"""
    if dt is not None:
        tstar_dt = dt

    if isinstance(components, hermitian_functions.ConstantMatrixHermitian):
        components = components.get_components()

    H_0, get_vt_, V = components
    get_vt = get_vt_
    if t_start != 0:
        get_vt = lambda t: get_vt_(t + t_start)
        t = t - t_start
    # from this point onwards, it is as if the range is [0, t],
    # even if it originally was [t_start, t]

    get_K = get_rbf(rbf_scale, rbf_C)

    if segmented:
        sample = -(0 + 1j) * (H_0 + get_vt(t / 2) * V)

        callback = lambda t_start, t: analytic_magnus(
            components, t, rbf_scale, rbf_C, k, tstar_dt, tstar_dt, False, segment_margin,
            t_start=t_start
        )

        return segmented_handler(callback, t, segment_margin, sample, verbose=verbose)

    A = rbf_C
    s = rbf_scale

    n = H_0.shape[0]
    U_0 = np.eye(n, dtype=complex)

    Omega_t = np.zeros((n, n), dtype=complex)

    Omega_t_ks = []

    tstar = np.linspace(0, t, int(t / tstar_dt))  # TODO check if better with t / star_dt + 1
    K_ts_ts = get_K(tstar[:, None], tstar[None, :])  # hack to evaluate over a 2d grid
    v_ts = get_vt(tstar)
    eta = np.linalg.solve(K_ts_ts, v_ts)
    erf_ts = scipy.special.erf(tstar / math.sqrt(2 * s))
    erf_ts_t = scipy.special.erf((tstar - t) / math.sqrt(2 * s))
    sspi2 = math.sqrt(s * math.pi / 2)
    for ki in range(1, k + 1):
        if ki == 1:
            sum = np.sum(eta * (erf_ts - erf_ts_t))
            res = (0 - 1j) * H_0 * t + (0 - 1j) * V * A * sspi2 * sum
            Omega_t_ks.append(res)
        elif ki == 2:
            edelta = np.exp(-tstar ** 2 / 2 / s) - np.exp(-(tstar - t) ** 2 / 2 / s)
            sum = np.sum(eta * (2 * s * edelta + sspi2 * (2 * tstar - t) * (erf_ts - erf_ts_t)))
            res = A * sqbrackets(H_0, V) * sum
            Omega_t_ks.append(res)
        else:
            raise Exception(f"Analytic Magnus not implemented for k={k}")

    Omega_t = np.sum(Omega_t_ks, axis=0)

    answer = scipy.linalg.expm(Omega_t) @ U_0
    return answer


def arg_broadcast_wrapper(fn, argname):
    def wrapped(*args, **kwargs):
        if isinstance(kwargs[argname], list):
            new_kwargs = {**kwargs}
            results = {}
            for argval in new_kwargs[argname]:
                new_kwargs[argname] = argval
                results[argname + "=" + str(argval)] = fn(*args, **new_kwargs)
            return results
        return fn(**kwargs)

    return wrapped


analytic_magnus_k = arg_broadcast_wrapper(analytic_magnus, "k")
naive_magnus_k = arg_broadcast_wrapper(magnus, "k")

if __name__ == "__main__":
    # print("---Magnus, non-segmented:")
    # print(magnus(
    #     hermitian_functions.two_spin_qubit_system.at_t,
    #     t=4, k=1, integrator_dt=0.004))
    # print("---Magnus, segmented:")
    # print(magnus(
    #     hermitian_functions.two_spin_qubit_system.at_t,
    #     t=4, k=1, integrator_dt=0.004, segmented=True))
    # print("---Analytic Magnus:")
    # print(analytic_magnus(
    #     hermitian_functions.two_spin_qubit_system.get_components(),
    #     t=4, k=2, tstar_dt=0.004,
    #     segmented=False, verbose=True))
    # print("---Analytic Magnus, segmented:")
    # print(analytic_magnus(
    #     hermitian_functions.two_spin_qubit_system.get_components(),
    #     t=4, k=2, tstar_dt=0.004,
    #     segmented=True, verbose=True))

    t_f = 5

    print("\n\n---Naive, ground truth::")
    gt = ground_truth.naive_simulator(hermitian_functions.two_spin_qubit_system, t_start=0, t=t_f, dt=1e-4)
    print(gt)
    print("\n\n---Magnus, non-segmented, k=2:")
    m_ns_2 = magnus(hermitian_functions.two_spin_qubit_system.at_t, t=t_f, k=2, integrator_dt=0.04)
    print(m_ns_2)
    print("\n\n---Magnus, non-segmented, k=3:")
    m_ns_3 = magnus(hermitian_functions.two_spin_qubit_system.at_t, t=t_f, k=3, integrator_dt=0.04)
    print(m_ns_3)
    print("\n\n---Magnus, segmented, k=2:")
    m_s_2 = magnus(hermitian_functions.two_spin_qubit_system.at_t, t=t_f, k=2, integrator_dt=0.04, segmented=True)
    print(m_s_2)
    print("\n\n---Magnus, segmented, k=3:")
    m_s_3 = magnus(hermitian_functions.two_spin_qubit_system.at_t, t=t_f, k=3, integrator_dt=0.04, segmented=True)
    print(m_s_3)

    print("---Analytic Magnus, non-segmented, k=2:")
    am_ns_1 = analytic_magnus(hermitian_functions.two_spin_qubit_system, t=t_f, k=1)
    print(am_ns_1)
    print("---Analytic Magnus, non-segmented, k=3:")
    am_ns_2 = analytic_magnus(hermitian_functions.two_spin_qubit_system, t=t_f, k=2)
    print(am_ns_2)
    print("---Analytic Magnus, segmented, k=1:")
    am_s_1 = analytic_magnus(hermitian_functions.two_spin_qubit_system, t=t_f, k=1, segmented=True)
    print(am_s_1)
    print("---Analytic Magnus, segmented, k=2:")
    am_s_2 = analytic_magnus(hermitian_functions.two_spin_qubit_system, t=t_f, k=2, segmented=True)
    print(am_s_2)

    print()
