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
        systems : list[int],
        # different systems to plot
        simulators: list[Simulator],
        # independent variable for graph x axis:
        indep_var : str, # "dt" or "segments"
        indep_var_range : list[Union[int, float]],
        const_vars : dict[str, Any] = {}
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
    
    def fidelities(self, ground_truth):
        results = self.results()
        return utils.walker(
            lambda mat : utils.fidelity(mat, ground_truth, len(ground_truth)),
            results
        )


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

ground_truth_1 = np.array(
    [[0.17945022+0.60663369j, 0.74265185-0.21968603j],
    [0.74265185-0.21968603j, 0.17945022+0.60663369j]]
 )

experiments[0].fidelities(ground_truth_1)
# %%
