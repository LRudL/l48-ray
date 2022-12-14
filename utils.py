import math
import numpy as np


def fidelity(U_tilde, U, n):
    '''
        Used to compare the intelligent evaluation result (Magnus expansion) U_tilde, to the ground truth from
        Euler integration of the Schrodinger equation U
    '''
    return math.abs(1/n * np.matrix.trace(np.matrix.getH(U_tilde) @ U))**2


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