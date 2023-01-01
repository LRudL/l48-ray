import math

import hermitian_functions
from ground_truth import naive_simulation
from magnus_expansion import magnus
from utils import fidelity


def fidelity_over_dts(max_omega, max_t, ground_truth, hermitian, dts: list, verbose: bool = True):
    fidelities = []
    for dt in dts:
        if verbose:
            print(f'Simulating dt={dt:.1e}')

        magnus_propagator = magnus(hermitian, max_t, k=max_omega, integrator_dt=dt, verbose=verbose)
        fid = fidelity(magnus_propagator, ground_truth, len(magnus_propagator))
        fidelities.append(fid)

        if verbose:
            print(f'Resulting propagator:\n{magnus_propagator}')
            print(f'Fidelity relative to ground truth: {fid:.3f}')


if __name__ == '__main__':
    max_t = 2.
    ground_truth_alt_sin, _ = naive_simulation(hermitian_functions.alt_sin_ssq, max_t, 0.0001)
    # ground_truth_alt_sin = np.array([[-0.02379359 - 1.29625832e-07j, -0.18683116 - 9.82108654e-01j],
    #                                  [0.18683116 - 9.82108654e-01j, -0.02379359 + 1.29625851e-07j]])
    print(ground_truth_alt_sin)

    dts = [5e-2]
    fidelity_over_dts(1, max_t, ground_truth_alt_sin, hermitian_functions.alt_sin_ssq, dts)
    fidelity_over_dts(2, max_t, ground_truth_alt_sin, hermitian_functions.alt_sin_ssq, dts)
    fidelity_over_dts(3, max_t, ground_truth_alt_sin, hermitian_functions.alt_sin_ssq, dts)
