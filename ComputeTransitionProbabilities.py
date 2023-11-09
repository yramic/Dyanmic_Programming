"""
 ComputeTransitionProbabilities.py

 Python function template to compute the transition probability matrix.

 Dynamic Programming and Optimal Control
 Fall 2023
 Programming Exercise
 
 Contact: Antonio Terpin aterpin@ethz.ch
 
 Authors: Abhiram Shenoi, Philip Pawlowsky, Antonio Terpin

 --
 ETH Zurich
 Institute for Dynamic Systems and Control
 --
"""

import numpy as np


def compute_transition_probabilities(Constants):
    """Computes the transition probability matrix P.

    It is of size (K,K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - P[i,j,l] corresponds to the probability of transitioning
            from the state i to the state j when input l is applied.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L).
    """
    K = Constants.T * Constants.D * Constants.N * Constants.M
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)

    P = np.zeros((K, K, L))
    # state_space = np.array(list(itertools.product(t, z, y, x)))

    # itertools.product constructs the state space by forming all combinations of the given  sets.
    # It does so by permuting the elements starting from the last set.
    # So the first element will be (t0,z0,y0,x0), followed by (t0,z0,y0,x1) up to (t0,z0,y0,x_M-1)
    # This is followed by (t0,z0,y1,x0) and so on. Consequently, moving forward one step in each direction is equal to:
    # x: 1
    # y: M
    # z: N*M
    # t: D*N*M
    # So, to move forward one step in time, one level in the z direction, and one step up (in y), the corresponding state would be
    # i + D*N*M + N*M + M
    # Care has to be taken for x, because it wraps around

    for i_t in range(Constants.T - 1):
        for i_z in range(Constants.Z):
            for i_y in range(Constants.M):
                for i_x in range(Constants.N):
                    pass

    # TODO fill the transition probability matrix P here

    return P
