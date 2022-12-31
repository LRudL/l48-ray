import math
import numpy as np


# Class for systems of the form: H(t) = H_0 + v(t)V
class ConstantMatrixHermitian:
    def __init__(self, H_0, get_vt, V):
        self.H_0 = H_0
        self.get_vt = get_vt
        self.V = V

    def get_components(self):
        return (self.H_0, self.get_vt, self.V)

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
    return math.sin(t)


sigma_x = np.matrix([
    [0 + 0j, 1 + 0j],
    [1 + 0j, 0 + 0j]])

sigma_y = np.matrix([
    [0 + 0j, 0 - 1j],
    [0 + 1j, 0 + 0j]])

sigma_z = np.matrix([
    [1 + 0j, 0 + 0j],
    [0 + 0j, -1 + 0j]])

I_2 = np.eye(2, dtype=complex)

single_spin_qubit_system = ConstantMatrixHermitian(I_2, b_t, sigma_x)

alt_ssq_system = ConstantMatrixHermitian(sigma_x, b_t, sigma_z)

alt_sin_ssq_system = ConstantMatrixHermitian(sigma_x, sin_t, sigma_z)

two_spin_qubit_system = ConstantMatrixHermitian(np.kron(sigma_z, sigma_z), b_t,
                                                np.kron(sigma_x, I_2) + np.kron(I_2, sigma_x))

single_spin_qubit = single_spin_qubit_system.at_t
two_spin_qubits = two_spin_qubit_system.at_t
alt_ssq = alt_ssq_system.at_t
alt_sin_ssq = alt_sin_ssq_system.at_t
