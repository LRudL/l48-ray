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


def fidelity_over_dts(k, t, ground_truth, hermitian):
    dts = [0.001]
    fidelities = []
    for dt in dts:
        print(dt)
        magnus_Ut = magnus(hermitian, t, k=k, integrator_dt=dt)
        print(magnus_Ut)
        fid = fidelity(magnus_Ut, ground_truth, len(magnus_Ut)).item(0,0)
        print(fid)
        fidelities.append(fid)
    # plt.plot(dts, fidelities)
    # plt.xlabel("dt")
    # plt.ylabel("Fidelity")
    # plt.xscale("log")
    # plt.title("Fidelity of naive magnus to naive euler over varying dt at omega order = " + str(k))
    # plt.show()


if __name__ == '__main__':
    ground_truth_ssq = np.matrix([[0.17945022+0.60663369j, 0.74265185-0.21968603j],
                            [0.74265185-0.21968603j, 0.17945022+0.60663369j]])
    ground_truth_tsq = np.matrix([[ 5.39394615e-01+0.21393098j, -6.29490754e-12-0.14641604j,
  -6.29494504e-12-0.14641604j,  2.55731720e-01-0.7449957j ],
 [ 6.29485797e-12-0.14641604j,  5.39394615e-01-0.21393098j,
   2.55731720e-01+0.7449957j,   6.29494854e-12-0.14641604j],
 [ 6.29485797e-12-0.14641604j,  2.55731720e-01+0.7449957j,
   5.39394615e-01-0.21393098j,  6.29494854e-12-0.14641604j],
 [ 2.55731720e-01-0.7449957j,  -6.29497726e-12-0.14641604j,
  -6.29494761e-12-0.14641604j,  5.39394615e-01+0.21393098j]])
    ground_truth_alt_sin, _ = naive_simulation(hermitian_functions.alt_sin_ssq, 2*math.pi, 0.000001)
    print(ground_truth_alt_sin)
    # tsq, tsqs = naive_simulation(hermitian_functions.two_spin_qubits, 5, 0.000001)
    # product = tsq @ np.matrix.conj(tsq).T
    # print(tsq)
    # print(product)
    # magnus_ssq_Ut = magnus(hermitian_functions.single_spin_qubit, 5, k=2, integrator_dt = 0.001)
    # print("magnus \n", magnus_ssq_Ut)
    # print(fidelity(magnus_ssq_Ut, euler_ssq_Ut, len(euler_ssq_Ut)))
    # #fidelity_over_n(magnus_ssq_Ut, euler_ssq_Ut)
    fidelity_over_dts(1, 2*math.pi, ground_truth_alt_sin, hermitian_functions.alt_sin_ssq)
    fidelity_over_dts(2, 2*math.pi, ground_truth_alt_sin, hermitian_functions.alt_sin_ssq)
    fidelity_over_dts(3, 2*math.pi, ground_truth_alt_sin, hermitian_functions.alt_sin_ssq)

