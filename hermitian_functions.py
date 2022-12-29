import math
import numpy as np


def b_t(t):
    return 0.5*math.exp(-(t-2.5)**2)


# 𝐻 = 𝐸_0 I_2 + 𝑏(𝑡)sigma_x
def single_spin_qubit(t, E_0=1, b=b_t):
    sigma_x = np.matrix([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]])
    return E_0 * np.eye(2, dtype=complex) + b(t) * sigma_x


def two_spin_qubits(t, D=1, b = b_t):
    sigma_x = np.matrix([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]])
    sigma_z = np.matrix([[0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j]])
    constant_term = np.tensordot(D*sigma_z, sigma_z)
    varying_term = b(t)*(np.tensordot(sigma_x, np.eye(2, dtype=complex))+ np.tensordot(np.eye(2, dtype=complex), sigma_x))
    return constant_term + varying_term

