import math
import numpy as np

def walker(map_fn, struct, prev_keys=[]):
    """Walks over a (potentially nested) data structure
    struct made of lists and dicts, and applies map_fn to
    the "leaves" of the structure, where a "leaf" is something
    that isn't a dict or a list"""
    if isinstance(struct, list):
        for i, val in enumerate(struct):
            if isinstance(val, list) or isinstance(val, dict):
                struct[i] = walker(map_fn, struct[i], prev_keys + [i])
            else:
                struct[i] = map_fn(struct[i], prev_keys)
    elif isinstance(struct, dict):
        for i, val in struct.items():
            if isinstance(val, list) or isinstance(val, dict):
                struct[i] = walker(map_fn, struct[i], prev_keys + [i])
            else:
                struct[i] = map_fn(struct[i], prev_keys)
    return struct


def fidelity(U_tilde, U, n):
    '''
        Used to compare the intelligent evaluation result (Magnus expansion) U_tilde, to the ground truth from
        Euler integration of the Schrodinger equation U
    '''
    return abs(1/n * np.matrix.trace(np.matrix.getH(U_tilde) @ U))**2


def get_singular_A(A):
    M = np.matrix.conj(A).T @ A
    eigenvalues, _ = np.linalg.eig(M)
    max_eig = np.max(eigenvalues)
    return np.sqrt(max_eig)


def magnus_convergence(get_Ht, T):
    ds = 0.0001
    integral = 0
    s = 0
    while s < T:
        integral += get_singular_A(get_Ht(s))*ds
        s += ds
    print(integral)
    return integral < math.pi


def hermitian(H):
    '''
        Tests whether matrix H is Hermitian
    '''
    H_conj_trans = np.matrix.conj(H).T
    return np.array_equal(H, H_conj_trans)


def unitary(U):
    '''
        Tests whether matrix U is unitary
    '''
    U_conj_trans = np.matrix.conj(U).T
    product = U @ U_conj_trans
    return np.isclose(product, np.eye(len(product), dtype = complex)).all()