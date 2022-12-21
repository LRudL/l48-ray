import magnus_expansion
from ground_truth import naive_simulation
import hermitian_functions
from magnus_expansion import magnus
from utils import fidelity, unitary
import numpy as np

if __name__ == '__main__':
    euler_ssq_Ut, euler_ssq_Uts = naive_simulation(hermitian_functions.single_spin_qubit, 5, 0.01)
    print("euler \n", euler_ssq_Ut)
    magnus_ssq_Ut = magnus(magnus_expansion.get_At, 2, k=1)
    print("magnus \n", magnus_ssq_Ut)
