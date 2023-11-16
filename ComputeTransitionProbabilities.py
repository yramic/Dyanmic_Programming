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

    # itertools.product constructs the state space by forming all combinations of the given sets.
    # It does so by permuting the elements starting from the last set.
    # So the first element will be (t0,z0,y0,x0), followed by (t0,z0,y0,x1) up to (t0,z0,y0,x_M-1)
    # This is followed by (t0,z0,y1,x0) and so on. Consequently, moving forward one step in each direction is equal to:
    # x: 1
    # y: M
    # z: N*M
    # t: D*N*M
    # So, to move forward one step in time, one level in the z direction, and one step up (in y), the corresponding state would be
    # i + D*N*M + N*M + M
    # Care has to be taken for x and t, because they wrap around
    x_step = 1
    y_step = Constants.M
    z_step = y_step * Constants.N
    t_step = z_step * Constants.D
    t_step_wrap = -(Constants.T - 1) * t_step

    # for u = STAY
    for i_t in range(Constants.T):
        for i_z in range(Constants.Z):
            for i_y in range(Constants.M):
                for i_x in range(Constants.N):
                    i = i_x + i_y * y_step + i_z * z_step + i_t * t_step

                    # if i_t doesn't need to wrap
                    if i_t != Constants.T - 1:
                        P[i][i + t_step][Constants.V_STAY] = (
                            Constants.Alpha
                            * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                        )  # Stay where you are

                        if i_x == 0:
                            P[i][i + t_step + Constants.M - 1][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                            )  # Go left at x=0
                        else:
                            P[i][i + t_step - x_step][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                            )  # Go left at x!=0

                        if i_x == Constants.M - 1:
                            P[i][i + t_step + 1 - Constants.M][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                            )  # Go right at x=M-1
                        else:
                            P[i][i + t_step + x_step][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                            )  # Go right at x!=M-1

                        if i_y != Constants.N - 1:
                            P[i][i + t_step + y_step][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                            )  # Go north at y!=N-1

                        if i_y != 0:
                            P[i][i + t_step - y_step][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                            )  # Go south at y!=0
                    # i_t wraps around
                    else:
                        P[i][i + t_step_wrap][Constants.V_STAY] = (
                            Constants.Alpha
                            * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                        )  # Stay where you are

                        if i_x == 0:
                            P[i][i + t_step_wrap + Constants.M - 1][
                                Constants.V_STAY
                            ] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                            )  # Go left at x=0
                        else:
                            P[i][i + t_step_wrap - x_step][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                            )  # Go left at x!=0

                        if i_x == Constants.M - 1:
                            P[i][i + t_step_wrap + 1 - Constants.M][
                                Constants.V_STAY
                            ] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                            )  # Go right at x=M-1
                        else:
                            P[i][i + t_step_wrap + x_step][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                            )  # Go right at x!=M-1

                        if i_y != Constants.N - 1:
                            P[i][i + t_step_wrap + y_step][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                            )  # Go north at y!=N-1

                        if i_y != 0:
                            P[i][i + t_step_wrap - y_step][Constants.V_STAY] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                            )  # Go south at y!=0

    # for u = UP
    for i_t in range(Constants.T):
        for i_z in range(Constants.Z):
            for i_y in range(Constants.M):
                for i_x in range(Constants.N):
                    i = i_x + i_y * y_step + i_z * z_step + i_t * t_step

                    if t_step != Constants.T - 1:
                        # Move only horizontally despite trying to go up, not at top level
                        # TODO: Transition probabilities at the top with input UP are undefined (should they be zero because the behaviour is not allowed, or should they be calculated as normal)
                        if i_z != Constants.D - 1:
                            P[i][i + t_step][Constants.V_UP] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                                * Constants.P_V_TRANSITION[0]
                            )  # Stay where you are despite trying to go up (no wind)

                            if i_x == 0:
                                P[i][i + t_step + Constants.M - 1][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go left at x=0
                            else:
                                P[i][i + t_step - x_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go left at x!=0

                            if i_x == Constants.M - 1:
                                P[i][i + t_step + 1 - Constants.M][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go right at x=M-1
                            else:
                                P[i][i + t_step + x_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go right at x!=M-1

                            if i_y != Constants.N - 1:
                                P[i][i + t_step + y_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go north at y!=N-1

                            if i_y != 0:
                                P[i][i + t_step - y_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go south at y!=0

                            # Go up
                            P[i][i + t_step + z_step][Constants.V_UP] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                                * Constants.P_V_TRANSITION[0]
                            )  # Go up only (no wind)

                            if i_x == 0:
                                P[i][i + t_step + z_step + Constants.M - 1][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go left at x=0
                            else:
                                P[i][i + t_step + z_step - x_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go left at x!=0

                            if i_x == Constants.M - 1:
                                P[i][i + t_step + z_step + 1 - Constants.M][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go right at x=M-1
                            else:
                                P[i][i + t_step + z_step + x_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go right at x!=M-1

                            if i_y != Constants.N - 1:
                                P[i][i + t_step + z_step + y_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go north at y!=N-1

                            if i_y != 0:
                                P[i][i + t_step + z_step - y_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go south at y!=0
                    # i_t wraps around
                    else:
                        if i_z != Constants.D - 1:
                            P[i][i + t_step_wrap][Constants.V_UP] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                                * Constants.P_V_TRANSITION[0]
                            )  # Stay where you are despite trying to go up (no wind)

                            if i_x == 0:
                                P[i][i + t_step_wrap + Constants.M - 1][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go left at x=0
                            else:
                                P[i][i + t_step_wrap - x_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go left at x!=0

                            if i_x == Constants.M - 1:
                                P[i][i + t_step_wrap + 1 - Constants.M][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go right at x=M-1
                            else:
                                P[i][i + t_step_wrap + x_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go right at x!=M-1

                            if i_y != Constants.N - 1:
                                P[i][i + t_step_wrap + y_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go north at y!=N-1

                            if i_y != 0:
                                P[i][i + t_step_wrap - y_step][Constants.V_UP] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go south at y!=0

                            # Go up
                            P[i][i + t_step_wrap + z_step][Constants.V_UP] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                                * Constants.P_V_TRANSITION[0]
                            )  # Go up only (no wind)

                            if i_x == 0:
                                P[i][i + t_step_wrap + z_step + Constants.M - 1][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go left at x=0
                            else:
                                P[i][i + t_step_wrap + z_step - x_step][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go left at x!=0

                            if i_x == Constants.M - 1:
                                P[i][i + t_step_wrap + z_step + 1 - Constants.M][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go right at x=M-1
                            else:
                                P[i][i + t_step_wrap + z_step + x_step][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go right at x!=M-1

                            if i_y != Constants.N - 1:
                                P[i][i + t_step_wrap + z_step + y_step][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go north at y!=N-1

                            if i_y != 0:
                                P[i][i + t_step_wrap + z_step - y_step][
                                    Constants.V_UP
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go south at y!=0

    # for u = DOWN
    for i_t in range(Constants.T):
        for i_z in range(Constants.Z):
            for i_y in range(Constants.M):
                for i_x in range(Constants.N):
                    i = i_x + i_y * y_step + i_z * z_step + i_t * t_step

                    if t_step != Constants.T - 1:
                        # Move only horizontally despite trying to go down, not at bottom level
                        if i_z != 0:
                            P[i][i + t_step][Constants.V_DOWN] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                                * Constants.P_V_TRANSITION[0]
                            )  # Stay where you are despite trying to go down (no wind)

                            if i_x == 0:
                                P[i][i + t_step + Constants.M - 1][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go left at x=0
                            else:
                                P[i][i + t_step - x_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go left at x!=0

                            if i_x == Constants.M - 1:
                                P[i][i + t_step + 1 - Constants.M][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go right at x=M-1
                            else:
                                P[i][i + t_step + x_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go right at x!=M-1

                            if i_y != Constants.N - 1:
                                P[i][i + t_step + y_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go north at y!=N-1

                            if i_y != 0:
                                P[i][i + t_step - y_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go south at y!=0

                            # Go down
                            P[i][i + t_step - z_step][Constants.V_DOWN] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                                * Constants.P_V_TRANSITION[0]
                            )  # Go down only (no wind)

                            if i_x == 0:
                                P[i][i + t_step - z_step + Constants.M - 1][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go left at x=0
                            else:
                                P[i][i + t_step - z_step - x_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go left at x!=0

                            if i_x == Constants.M - 1:
                                P[i][i + t_step - z_step + 1 - Constants.M][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go right at x=M-1
                            else:
                                P[i][i + t_step - z_step + x_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go right at x!=M-1

                            if i_y != Constants.N - 1:
                                P[i][i + t_step - z_step + y_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go north at y!=N-1

                            if i_y != 0:
                                P[i][i + t_step - z_step - y_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go south at y!=0

                    # i_t wraps around
                    else:
                        # Move only horizontally despite trying to go down, not at bottom level
                        if i_z != 0:
                            P[i][i + t_step_wrap][Constants.V_DOWN] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                                * Constants.P_V_TRANSITION[0]
                            )  # Stay where you are despite trying to go down (no wind)

                            if i_x == 0:
                                P[i][i + t_step_wrap + Constants.M - 1][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go left at x=0
                            else:
                                P[i][i + t_step_wrap - x_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go left at x!=0

                            if i_x == Constants.M - 1:
                                P[i][i + t_step_wrap + 1 - Constants.M][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go right at x=M-1
                            else:
                                P[i][i + t_step_wrap + x_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go right at x!=M-1

                            if i_y != Constants.N - 1:
                                P[i][i + t_step_wrap + y_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go north at y!=N-1

                            if i_y != 0:
                                P[i][i + t_step_wrap - y_step][Constants.V_DOWN] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                                    * Constants.P_V_TRANSITION[0]
                                )  # Go south at y!=0

                            # Go down
                            P[i][i + t_step_wrap - z_step][Constants.V_DOWN] = (
                                Constants.Alpha
                                * Constants.P_H_TRANSITION[i_z][Constants.H_STAY]
                                * Constants.P_V_TRANSITION[0]
                            )  # Go down only (no wind)

                            if i_x == 0:
                                P[i][i + t_step_wrap - z_step + Constants.M - 1][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go left at x=0
                            else:
                                P[i][i + t_step_wrap - z_step - x_step][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_WEST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go left at x!=0

                            if i_x == Constants.M - 1:
                                P[i][i + t_step_wrap - z_step + 1 - Constants.M][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go right at x=M-1
                            else:
                                P[i][i + t_step_wrap - z_step + x_step][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_EAST]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go right at x!=M-1

                            if i_y != Constants.N - 1:
                                P[i][i + t_step_wrap - z_step + y_step][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_NORTH]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go north at y!=N-1

                            if i_y != 0:
                                P[i][i + t_step_wrap - z_step - y_step][
                                    Constants.V_DOWN
                                ] = (
                                    Constants.Alpha
                                    * Constants.P_H_TRANSITION[i_z][Constants.H_SOUTH]
                                    * Constants.P_V_TRANSITION[1]
                                )  # Go south at y!=0

    # TODO fill the transition probability matrix P here

    return P
