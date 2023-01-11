# %%
from typing import Callable, Union, Any
import numpy as np

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
        self, name : str,
        systems,
        # different systems to plot
        simulators,
        # independent variable for graph x axis:
        indep_var : str, # "dt" or "segments"
        indep_var_range ,
        const_vars = None,
    ):
        self.name = name
        self.systems = systems
        self.simulators = simulators
        self.indep_var = indep_var
        self.indep_var_range = indep_var_range
        self.const_vars = const_vars
    
    def results(self):
        # results = np.zeros((len(self.simulators), len(self.systems), len(self.indep_var_range)))
        results = {}
        for i_sim, sim in enumerate(self.simulators):
            results[sim.name] = {}
            for i_sys, system in enumerate(self.systems):
                results[sim.name][system.name] = {}
                result_arr = []
                for i_x, x in enumerate(self.indep_var_range):
                    # Create dictionary of all variables to pass in:
                    sim_vars = {**{self.indep_var: x}, **self.const_vars}
                    # print(sim_vars)
                    # System is first arg (this is required of simulators);
                    # rest args from unpacking the kwarg dictionary sim_vars
                    sim_result = sim.run(system, **sim_vars)
                    # print("----")
                    # print(sim_result)
                    # print(ground_truth)
                    # print(len(ground_truth))
                    
                    # if isinstance(sim_result, dict):
                    #     result = {key : utils.fidelity(val, ground_truth, len(ground_truth)) for key, val in sim_result.items()}
                    # else:
                    #     result = utils.fidelity(sim_result, ground_truth, len(ground_truth))

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
                raise Exception(f"ground_truths passed into fidelities does not contain an entry for any of {prev_keys}")
            return utils.fidelity(result_matrix, correct_ground_truth, len(correct_ground_truth))
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
        name="convergence",
        systems=[
            hermitian_functions.single_spin_qubit_system#,
            # hermitian_functions.alt_ssq_system,
            # hermitian_functions.alt_sin_ssq_system,
            # hermitian_functions.two_spin_qubit_system
        ],
        simulators=[
            Simulator(
                name = "naive", function = naive_simulator
            ),
            Simulator(
                name = "magnus",
                function = naive_magnus_k,
                # Pass any arguments specific to the simulator within
                # the simulator.
                k=[1, 2], segmented = False
            ),
            Simulator(
                name = "analytic_magnus",
                function = analytic_magnus_k,
                k=[1, 2], segmented=False
            )
        ],
        indep_var="dt",
        indep_var_range=[2e-1, 2e-2],
        # Pass any variables you want to be passed to all simulators here:
        const_vars={"t_start": 0, "t": 2}
    )
]
# %%

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

ground_truths = {"single spin qubit": [[-0.40680456-0.88888417j, -0.19159047+0.0876828j ],
                                        [-0.19159047+0.0876828j,  -0.40680456-0.88888417j]],
                 "alt sin single spin qubit": [[ 0.44764106-2.75461214e-06j, -0.27463933+8.51025179e-01j],
                                        [ 0.27463933+8.51025179e-01j,  0.44764106+2.75461214e-06j]],
                 "two spin qubit": [[ 5.39394615e-01+0.21393098j, -6.29490754e-12-0.14641604j,
                                                -6.29494504e-12-0.14641604j,  2.55731720e-01-0.7449957j ],
                                                [ 6.29485797e-12-0.14641604j,  5.39394615e-01-0.21393098j,
                                                2.55731720e-01+0.7449957j,   6.29494854e-12-0.14641604j],
                                                [ 6.29485797e-12-0.14641604j,  2.55731720e-01+0.7449957j,
                                                5.39394615e-01-0.21393098j,  6.29494854e-12-0.14641604j],
                                                [ 2.55731720e-01-0.7449957j,  -6.29497726e-12-0.14641604j,
                                                 -6.29494761e-12-0.14641604j,  5.39394615e-01+0.21393098j]]}

if __name__ == "__main__":
    experiments[0].fidelities(ground_truths)
# %%
