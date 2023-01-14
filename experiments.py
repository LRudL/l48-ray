# %%
from typing import Callable, Union, Any, List
import numpy as np
import matplotlib.pyplot as plt

import hamiltonians
import utils
from ground_truth import naive_simulator
from magnus_expansion import naive_magnus_k, analytic_magnus_k


class Simulator:
    def __init__(self, name, function, **kwargs):
        self.name = name
        self.kwargs = kwargs
        # requirements on `function``:
        # - it can take in a hamiltonians.Hamiltonian 
        #   as a first argument
        # - 
        self.function = function

    def run(self, system, **additional_kwargs):
        full_kwargs = {**self.kwargs, **additional_kwargs}
        # print(full_kwargs)
        return self.function(system, **full_kwargs)


DEFAULT_SAVE_PATH = "experiment_data"


class Experiment:
    def __init__(
            self, name: str,
            systems: List[hamiltonians.Hamiltonian],
            # different systems to plot
            simulators,
            # independent variable for graph x axis:
            indep_var: str,  # "dt" or "segments"
            indep_var_range,
            const_vars=None,
    ):
        self.name = name
        self.systems = systems
        self.simulators = simulators
        self.indep_var = indep_var
        self.indep_var_range = indep_var_range
        self.const_vars = const_vars

    def results(self):
        results = {}
        for i_sim, sim in enumerate(self.simulators):
            results[sim.name] = {}
            for i_sys, system in enumerate(self.systems):
                results[sim.name][system.name] = {}
                result_arr = []
                for i_x, x in enumerate(self.indep_var_range):
                    # Create dictionary of all variables to pass in:
                    sim_vars = {**{self.indep_var: x}, **self.const_vars}
                    # System is first arg (this is required of simulators);
                    # rest args from unpacking the kwarg dictionary sim_vars
                    sim_result = sim.run(system, **sim_vars)

                    result_arr.append(sim_result)
                results[sim.name][system.name] = result_arr
        # Output format:
        # dict[simulator_name ->
        #   dict[system_name ->
        #       array[indep_variable_index -> OBJ]]]
        #  where, if simulator is a magnus into which k was
        #  passed as a list, OBJ will be a dictionary from
        #  "k=n" where n is the value of k to a value,
        #  and otherwise OBJ will be a value
        return results

    def fidelities(self, ground_truths):
        results = self.results()

        def find_ground_truth_and_get_fidelity(result_matrix, prev_keys):
            correct_ground_truth = None
            for key in prev_keys:
                # key loops through all of the indices, list index and dictionary key,
                # above the matrix in results
                if key in ground_truths.keys():
                    correct_ground_truth = ground_truths[key]
            if correct_ground_truth is None:
                raise Exception(f"ground_truths passed into fidelities does not contain an entry"
                                f" for any of {prev_keys}")
            return utils.fidelity(result_matrix, correct_ground_truth)

        return utils.walker(find_ground_truth_and_get_fidelity, results)

    def run(self, ground_truth, save=False, save_path=DEFAULT_SAVE_PATH):
        results = self.results()
        fidelities = self.fidelities(ground_truth)
        if save:
            np.save(save_path + "/" + self.name + "_results", fidelities)

            ## TODO: also save graphs of all the experiments

        return results


experiments = [
    Experiment(
        name="truncation omega and dt",
        systems=[
            hamiltonians.ssq_shifted_system,
            hamiltonians.alt_sin_ssq_system,
            hamiltonians.tsq_shifted_system
        ],
        simulators=[
            Simulator(
                name="magnus", function=naive_magnus_k,
                k=[1, 2, 3], segmented=False
            ),
            Simulator(
                name="analytic magnus", function=analytic_magnus_k,
                k=[1, 2], segmented=False
            )
        ],
        indep_var="dt",
        indep_var_range=[1, .9, .8, .7, .6, .5, .4, .3, .2, .1],
        # Pass any variables you want to be passed to all simulators here:
        const_vars={"t_start": 0, "t": 2}
    ),
    Experiment(
        name="truncation segmentation",
        systems=[
            hamiltonians.ssq_shifted_system,
            hamiltonians.alt_sin_ssq_system,
            hamiltonians.tsq_shifted_system
        ],
        simulators=[
            Simulator(
                name="magnus",
                function=naive_magnus_k,
                # Pass any arguments specific to the simulator within
                # the simulator.
                k=[1, 2, 3]
            ),
            Simulator(
                name="analytic magnus",
                function=analytic_magnus_k,
                # Pass any arguments specific to the simulator within
                # the simulator.
                k=[1, 2]
            )
        ],
        indep_var="segmented",
        indep_var_range=[1, 2, 3, 4, 5],
        const_vars={"t_start": 0, "t": 2}
    ),
    Experiment(
        name="pulse mismatch",
        systems=[
            hamiltonians.Hamiltonian(f'two spin qubit-{pulse_name}', hamiltonians.two_spin_qubit_system.H_0, pulse,
                                     hamiltonians.two_spin_qubit_system.V) for pulse_name, pulse in
            (('Gaussian', hamiltonians.b_t_shifted),
             ('Hann', hamiltonians.hann_pulse),
             ('Blackman', hamiltonians.blackman_pulse),
             ('Double Gaussian', hamiltonians.double_gaussian_pulse))
        ],
        simulators=[
            Simulator(
                name="analytic_magnus",
                function=analytic_magnus_k,
                k=[2], segmented=False
            )
        ],
        indep_var="dt",
        indep_var_range=[1, .9, .8, .7, .6, .5, .4, .3, .2, .1],
        # Pass any variables you want to be passed to all simulators here:
        const_vars={"t_start": 0, "t": 2}
    ),
    Experiment(
        name="convergence",
        systems=[
            hamiltonians.two_spin_qubit_system
        ],
        simulators=[
            Simulator(
                name="naive", function=naive_simulator
            ),
            Simulator(
                name="magnus",
                function=naive_magnus_k,
                # Pass any arguments specific to the simulator within
                # the simulator.
                k=[1, 2, 3], segmented=False
            ),
            Simulator(
                name="analytic_magnus",
                function=analytic_magnus_k,
                k=[1, 2], segmented=False
            )
        ],
        indep_var="dt",
        indep_var_range=[1e-1, 3e-2, 1e-2],
        # Pass any variables you want to be passed to all simulators here:
        const_vars={"t_start": 0, "t": 2}
    )
]


def plot_truncation(experiment, ground_truths, indep_var="dt"):
    results = experiment.fidelities(ground_truths)
    indep_var_vals = experiment.indep_var_range
    for simulator in results:
        simulator_results = results.get(simulator)
        for system in simulator_results:
            system_results = simulator_results.get(system)
            k_values = len(system_results[0])
            markers = ['*', 'o', 'X', 's', '>', 'd', 'H', '<', '^', '1', '2', '3', 'p']
            for max_omega in range(1, k_values + 1):
                key = "k=" + str(max_omega)
                fidelities = []
                for dt_res in system_results:
                    fidelities.append(dt_res.get(key))
                plt.plot(indep_var_vals, fidelities, label=key, marker=markers[max_omega - 1])
            plt.legend()
            plt.xlabel(indep_var)
            plt.ylabel("fidelity")
            if indep_var == "dt":
                plt.title(
                    "Truncation error over different maximum omega values for system: " + system + ", " + simulator)
                plt.savefig("omega truncation error " + system + ", " + simulator + ".png", bbox_inches="tight")
            else:
                plt.title("Truncation error over different number of segments: " + system + ", " + simulator)
                plt.savefig("segmentation truncation error " + system + ", " + simulator + ".png", bbox_inches="tight")


def plot_convergence(experiment, ground_truths, fontsize: float = 13):
    results = experiment.fidelities(ground_truths)
    dts = experiment.indep_var_range
    for simulator in results:
        simulator_results = results.get(simulator)
        for system in simulator_results:
            system_results = simulator_results.get(system)
            k_values = len(system_results)
            if simulator == "naive":
                fidelities = system_results
                plt.plot(dts, fidelities, label=simulator)
            else:
                for max_omega in range(1, k_values + 1):
                    key = "k=" + str(max_omega)
                    fidelities = []
                    for dt_res in system_results:
                        fidelities.append(dt_res.get(key))
                    plt.plot(dts, fidelities, label=simulator + " " + key)
    plt.xlabel(r"$\Delta$t", fontsize=fontsize)
    plt.ylabel("Fidelity", fontsize=fontsize)
    plt.legend()
    plt.title("Convergence to the ground truth of a system over varying dt")
    plt.savefig("convergence experiment t=0-2", bbox_inches="tight")


# TODO - prettify plot
def plot_pulses_mismatch(experiment: Experiment, ground_truths, fontsize: float = 13):
    results = experiment.fidelities(ground_truths)
    dts = experiment.indep_var_range
    for simulator in results:
        simulator_results = results.get(simulator)
        for system in simulator_results:
            pulse_name = system[system.find('-') + 1:]
            system_results = simulator_results.get(system)
            fid_per_dt = [list(res.values())[0] for res in system_results]

            plt.plot(dts, fid_per_dt, label=pulse_name)

    plt.xlabel(r"$\Delta$t", fontsize=fontsize)
    plt.ylabel("Fidelity", fontsize=fontsize)
    plt.legend()


ground_truths = {"single spin qubit": [[-0.40680456 - 0.88888417j, -0.19159047 + 0.0876828j],
                                       [-0.19159047 + 0.0876828j, -0.40680456 - 0.88888417j]],
                 "alt sin single spin qubit": [[-0.73582847 - 0.46178757j, -0.30887546 - 0.38717933j],
                                               [0.30887546 - 0.38717933j, -0.73582847 + 0.46178757j]],
                 "two spin qubit": [[-0.42536777 - 0.87350815j, 0.15624935 - 0.05410286j, 0.15624935 - 0.05410286j,
                                     -0.00921961 + 0.03578977j],
                                    [-0.15624935 - 0.05410286j, -0.42536777 + 0.87350815j, -0.00921961 - 0.03578977j,
                                     -0.15624935 - 0.05410286j],
                                    [-0.15624935 - 0.05410286j, -0.00921961 - 0.03578977j, -0.42536777 + 0.87350815j,
                                     -0.15624935 - 0.05410286j],
                                    [-0.00921961 + 0.03578977j, 0.15624935 - 0.05410286j, 0.15624935 - 0.05410286j,
                                     -0.42536777 - 0.87350815j]]}

ground_truths_shifted = {"single spin qubit": [[-0.3290767 - 0.71904284j, -0.55659709 + 0.25473096j],
                                               [-0.55659709 + 0.25473096j, -0.3290767 - 0.71904284j]],
                         "alt sin single spin qubit": [[-0.73582847 - 0.46178757j, -0.30887546 - 0.38717933j],
                                                       [0.30887546 - 0.38717933j, -0.73582847 + 0.46178757j]],
                         "two spin qubit": [
                             [-0.55430206 - 0.67417283j, 0.16989756 - 0.23037152j, 0.16989756 - 0.23037152j,
                              -0.1381539 + 0.23512509j],
                             [-0.16989756 - 0.23037152j, -0.55430206 + 0.67417283j, -0.1381539 - 0.23512509j,
                              -0.16989756 - 0.23037152j],
                             [-0.16989756 - 0.23037152j, -0.1381539 - 0.23512509j, -0.55430206 + 0.67417283j,
                              -0.16989756 - 0.23037152j],
                             [-0.1381539 + 0.23512509j, 0.16989756 - 0.23037152j, 0.16989756 - 0.23037152j,
                              -0.55430206 - 0.67417283j]]}

if __name__ == "__main__":
    # TODO - properly generate gt for these systems
    pulse_mismatch_gt = {system.name: ground_truths_shifted['two spin qubit'] for system in experiments[2].systems}
    plot_pulses_mismatch(experiments[2], pulse_mismatch_gt)
    # plot_truncation(experiments[1], ground_truths_shifted, indep_var="segmented")

    plt.show()
