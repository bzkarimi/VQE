# Variational Quantum Eigensolver (VQE)

Simple quantum computing codes using VQE 

## **General comments regarding the codes**:

1. vqe_state_vector.py: Given a random probability vector, it determines a possible parameterization for our single qubit variational form that is close to the given vector using the U3 gate.  

2. vqe_LiH.py: It calculates ground state LiH as a function of r using VQE and Statevector Simulator.

3. vqe_H2.py: Running VQE on a Noisy Simulator with error mitigation.  
**Note**: You need to have an IBMQ token and account to be able to run this code.




