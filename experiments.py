# %%
from typing import Callable, Union, Any
import numpy as np
import matplotlib.pyplot as plt

import hermitian_functions
import utils
from ground_truth import naive_simulator
from magnus_expansion import naive_magnus_k, analytic_magnus_k


class Simulator:
    def __init__(self, name, function, **kwargs):
        self.name = name
        self.kwargs = kwargs
        # requirements on `function``:
        # - it can take in a hermitian_functions.ConstantHermitianFunction 
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
            systems,
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


systems = [
    hermitian_functions.single_spin_qubit_system,
    hermitian_functions.alt_ssq_system,
    hermitian_functions.alt_sin_ssq_system,
    hermitian_functions.two_spin_qubit_system
]

experiments = [
    Experiment(
        name="truncation omega and dt",
        systems=[
            hermitian_functions.single_spin_qubit_system,
            hermitian_functions.alt_sin_ssq_system,
            hermitian_functions.two_spin_qubit_system
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
        indep_var_range=[1, 9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1, 2e-1, 1e-1],
        # Pass any variables you want to be passed to all simulators here:
        const_vars={"t_start": 0, "t": 2}
    ),
    Experiment(
        name="convergence",
        systems=[
            hermitian_functions.two_spin_qubit_system
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


def plot_convergence(experiment, ground_truths):
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
    plt.xlabel("dt")
    plt.ylabel("fidelity")
    plt.legend()
    plt.title("Convergence to the ground truth of a system over varying dt")
    plt.savefig("convergence experiment t=0-2", bbox_inches="tight")
    plt.show()


def plot_truncation_omegas(experiment, ground_truths):
    results = experiment.fidelities(ground_truths)
    print(results)
    dts = experiment.indep_var_range
    for simulator in results:
        simulator_results = results.get(simulator)
        for system in simulator_results:
            system_results = simulator_results.get(system)
            k_values = len(system_results[0])
            for max_omega in range(1, k_values + 1):
                key = "k=" + str(max_omega)
                fidelities = []
                for dt_res in system_results:
                    fidelities.append(dt_res.get(key))
                print(simulator + key + str(fidelities))
                plt.plot(dts, fidelities, label=key)
            plt.legend()
            plt.xlabel("dt")
            plt.ylabel("fidelity")
            plt.title("Truncation error over different maximum omega values for system: " + system + ", " + simulator)
            plt.savefig("omega truncation error " + system + ", " + simulator + ".png", bbox_inches="tight")
            plt.show()


# ground_truths = {"single spin qubit": {(0,2):[[-0.40680456-0.88888417j, -0.19159047+0.0876828j ],
#                                         [-0.19159047+0.0876828j,  -0.40680456-0.88888417j]]},
#                  "alt sin single spin qubit": {(0,2): [[ 0.44764106-2.75461214e-06j, -0.27463933+8.51025179e-01j],
#                                         [ 0.27463933+8.51025179e-01j,  0.44764106+2.75461214e-06j]]},
#                  "two spin qubit": {(0,2):  [[ 5.39394615e-01+0.21393098j, -6.29490754e-12-0.14641604j,
#                                                 -6.29494504e-12-0.14641604j,  2.55731720e-01-0.7449957j ],
#                                                 [ 6.29485797e-12-0.14641604j,  5.39394615e-01-0.21393098j,
#                                                 2.55731720e-01+0.7449957j,   6.29494854e-12-0.14641604j],
#                                                 [ 6.29485797e-12-0.14641604j,  2.55731720e-01+0.7449957j,
#                                                 5.39394615e-01-0.21393098j,  6.29494854e-12-0.14641604j],
#                                                 [ 2.55731720e-01-0.7449957j,  -6.29497726e-12-0.14641604j,
#                                                  -6.29494761e-12-0.14641604j,  5.39394615e-01+0.21393098j]]}}

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

if __name__ == "__main__":
    plot_truncation_omegas(experiments[0], ground_truths)
# %%
