import math
from typing import Tuple

import numpy as np


# Class for systems of the form: H(t) = H_0 + v(t)V
class ConstantMatrixHermitian:
    def __init__(self, name, H_0, get_vt, V):
        self.name = name
        self.H_0 = H_0
        self.get_vt = get_vt
        self.V = V

    def get_components(self) -> Tuple:
        return self.H_0, self.get_vt, self.V

    def at_t(self, t, *args):
        # NOTE: this throws away all other args
        # So passing in E_0, etc. in e.g. naive_simulation will not
        # change things anymore. Create a new ConstantMatrixHermitian object
        # with different constants to change parameters like that,
        # or make this class more expressive.
        return self.H_0 + self.get_vt(t) * self.V


# HELPER FUNCTIONS / DEFINITIONS:

def b_t(t):
    return 0.5 * np.exp(-(t - 2.5) ** 2)


def sin_t(t):
    return np.sin(t)


sigma_x = np.array([
    [0 + 0j, 1 + 0j],
    [1 + 0j, 0 + 0j]])

sigma_y = np.array([
    [0 + 0j, 0 - 1j],
    [0 + 1j, 0 + 0j]])

sigma_z = np.array([
    [1 + 0j, 0 + 0j],
    [0 + 0j, -1 + 0j]])

I_2 = np.eye(2, dtype=complex)

single_spin_qubit_system = ConstantMatrixHermitian("single spin qubit", I_2, b_t, sigma_x)

alt_ssq_system = ConstantMatrixHermitian("alt single spin qubit", sigma_x, b_t, sigma_z)

alt_sin_ssq_system = ConstantMatrixHermitian("alt sin single spin qubit", sigma_x, sin_t, sigma_z)

two_spin_qubit_system = ConstantMatrixHermitian("two spin qubit", np.kron(sigma_z, sigma_z), b_t,
                                                np.kron(sigma_x, I_2) + np.kron(I_2, sigma_x))

single_spin_qubit = single_spin_qubit_system.at_t
two_spin_qubits = two_spin_qubit_system.at_t
alt_ssq = alt_ssq_system.at_t
alt_sin_ssq = alt_sin_ssq_system.at_t
