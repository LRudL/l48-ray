import magnus_expansion
from ground_truth import naive_simulation
import hermitian_functions
from magnus_expansion import magnus
import math
from utils import fidelity, unitary
import matplotlib.pyplot as plt
import numpy as np


def distance_error(euler, magnus):
    ground_distance = 0
    for i in range(len(euler)):
        for j in range(len(euler[0])):
            ground_distance += euler[i][j]**2
    ground_distance = math.sqrt(ground_distance)
    magnus_distance = 0
    for i in range(len(magnus)):
        for j in range(len(magnus[0])):
            magnus_distance += magnus[i][j]**2
    magnus_distance = math.sqrt(ground_distance)
    return magnus_distance-ground_distance


def fidelity_over_n(U_tilde, U):
    fidelities = []
    for n in range(1,1000):
        fidelities.append(fidelity(U_tilde, U, n).item(0,0))
    plt.plot(np.arange(1,1000), fidelities)
    plt.show()


def fidelity_over_dts(k, ground_truth):
    dts = [0.03, 0.01, 0.003, 0.001]
    fidelities = []
    for dt in dts:
        print(dt)
        magnus_ssq_Ut = magnus(hermitian_functions.two_spin_qubits, 5, k=k, integrator_dt=dt)
        print(type(ground_truth), type(magnus_ssq_Ut))
        fidelities.append(fidelity(magnus_ssq_Ut, ground_truth, len(magnus_ssq_Ut)).item(0,0))
    plt.plot(dts, fidelities)
    plt.xlabel("dt")
    plt.ylabel("Fidelity")
    plt.xscale("log")
    plt.title("Fidelity of naive magnus to naive euler over varying dt at omega order = " + str(k))
    plt.show()


if __name__ == '__main__':
    ground_truth_ssq = np.matrix([[0.17945022+0.60663369j, 0.74265185-0.21968603j],
                            [0.74265185-0.21968603j, 0.17945022+0.60663369j]])
    ground_truth_tsq,_ = np.matrix([[ 0.70404166+0.45647453j, -0.29595834+0.45647453j],
                            [-0.29595834+0.45647453j,  0.70404166+0.45647453j]])
    tsq, tsqs = naive_simulation(hermitian_functions.two_spin_qubits, 5, 0.00001, I = 4)
    product = tsq @ np.matrix.conj(tsq).T
    print(tsq)
    print(product)
    # magnus_ssq_Ut = magnus(hermitian_functions.single_spin_qubit, 5, k=2, integrator_dt = 0.001)
    # print("magnus \n", magnus_ssq_Ut)
    # print(fidelity(magnus_ssq_Ut, euler_ssq_Ut, len(euler_ssq_Ut)))
    # #fidelity_over_n(magnus_ssq_Ut, euler_ssq_Ut)
    fidelity_over_dts(1, tsq)
