import math
import numpy as np


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