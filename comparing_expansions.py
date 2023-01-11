import math

import matplotlib.pyplot as plt
import hermitian_functions
from ground_truth import naive_simulation
from magnus_expansion import magnus
from utils import fidelity, magnus_convergence


def plot_fidelities(dts, fidelities, system):
    omega = 1
    for f in fidelities:
        plt.plot(dts, f, label = "Max omega: "+ str(omega))
        omega += 1
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("dt")
    plt.title("Fidelities of naive magnus for " + system)
    plt.show()


def fidelity_over_dts(max_omega, max_t, ground_truth, hermitian, dts: list, verbose: bool = True, segmented = False):
    fidelities = []
    for dt in dts:
        if verbose:
            print(f'Simulating dt={dt:.1e}')

        magnus_propagator = magnus(hermitian, max_t, k=max_omega, integrator_dt=dt, verbose=False, segmented=segmented)
        fid = fidelity(magnus_propagator, ground_truth, len(magnus_propagator))
        fidelities.append(fid)

        if verbose:
            print(f'Resulting propagator:\n{magnus_propagator}')
            print(f'Fidelity relative to ground truth: {fid:.3f} \n')
    return fidelities


if __name__ == '__main__':
    max_t = 2.
    system = hermitian_functions.two_spin_qubits
    segmentation_needed = not(magnus_convergence(system, max_t))
    print("Series convergence for max_t: ", not(segmentation_needed))
    # ground_truth_alt_sin, _ = naive_simulation(hermitian_functions.alt_sin_ssq, max_t, 0.0001)
    # ground_truth_alt_sin = np.array([[-0.02379359 - 1.29625832e-07j, -0.18683116 - 9.82108654e-01j],
    #                                  [0.18683116 - 9.82108654e-01j, -0.02379359 + 1.29625851e-07j]])
    ground_truth, _ = naive_simulation(system, max_t, 0.0001)

    dts = [5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 5e-3]
    f1 = fidelity_over_dts(1, max_t, ground_truth, system, dts, segmented=segmentation_needed)
    f2 = fidelity_over_dts(2, max_t, ground_truth, system, dts, segmented=segmentation_needed)
    f3 = fidelity_over_dts(3, max_t, ground_truth, system, dts, segmented=segmentation_needed)

    fidelities_over_omegas = [f1, f2, f3]
    plot_fidelities(dts, fidelities_over_omegas, "Two spin qubit")