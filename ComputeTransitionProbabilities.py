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
    import numpy as np

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
    g_x_step = 1
    g_x_step_wrap = -(Constants.M - 1)
    y_step = Constants.M
    g_z_step = y_step * Constants.N
    g_t_step = g_z_step * Constants.D
    g_t_step_wrap = -(Constants.T - 1) * g_t_step

    # for u = STAY
    for u in input_space:
        for i_t in range(Constants.T):
            for i_z in range(Constants.D):
                for i_y in range(Constants.N):
                    for i_x in range(Constants.M):
                        i = i_x + i_y * y_step + i_z * g_z_step + i_t * g_t_step
                        t_step = g_t_step if i_t != Constants.T - 1 else g_t_step_wrap
                        x_right = g_x_step if i_x != Constants.M - 1 else g_x_step_wrap
                        x_left = -g_x_step if i_x != 0 else -g_x_step_wrap
                        P_V_stay = (
                            1 if u == Constants.V_STAY else Constants.P_V_TRANSITION[0]
                        )
                        # P_V_move = (
                        #     0 if u == Constants.V_STAY else Constants.P_V_TRANSITION[1]
                        # )
                        z_step = g_z_step if u == Constants.V_UP else -g_z_step

                        if (u == Constants.V_UP and i_z == Constants.D - 1) or (
                            u == Constants.V_DOWN and i_z == 0
                        ):
                            continue

                        # ----------------no vertical displacement ---------------------------
                        P[i][i + t_step + x_left][u] = (
                            Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_WEST]
                        ) * P_V_stay  # Go left at x!=0

                        P[i][i + t_step + x_right][u] = (
                            Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_EAST]
                        ) * P_V_stay  # Go right at x!=M-1

                        # if we are at the top
                        if i_y == Constants.N - 1:
                            P[i][i + t_step][u] = (
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_STAY]
                                + Constants.P_H_TRANSITION[i_z].P_WIND[
                                    Constants.H_NORTH
                                ]
                            ) * P_V_stay  # Stay where you are if no wind or pushed north
                            P[i][i + t_step - y_step][u] = (
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_SOUTH]
                            ) * P_V_stay  # Go south at y!=0
                        # if we are at the bottom
                        elif i_y == 0:
                            P[i][i + t_step][u] = (
                                (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_STAY
                                    ]
                                    + Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_SOUTH
                                    ]
                                )
                            ) * P_V_stay  # Stay where you are if no wind or pushed south
                            P[i][i + t_step + y_step][u] = (
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_NORTH]
                            ) * P_V_stay  # Go north at y!=N-1
                        # not at extremum in y
                        else:
                            P[i][i + t_step][u] = (
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_STAY]
                            ) * P_V_stay  # Stay where you are if no wind
                            P[i][i + t_step + y_step][u] = (
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_NORTH]
                            ) * P_V_stay  # Go north at y!=N-1
                            P[i][i + t_step - y_step][u] = (
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_SOUTH]
                            ) * P_V_stay  # Go south at y!=0
                        # -----------------------------------------------------

                        # ------------vertical displacement---------------------
                        if u == Constants.V_UP or u == Constants.V_DOWN:
                            P[i][i + t_step + z_step + x_left][u] = (
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_WEST]
                                * Constants.P_V_TRANSITION[1]
                            )  # Go left at x=0

                            P[i][i + t_step + z_step + x_right][u] = (
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_EAST]
                                * Constants.P_V_TRANSITION[1]
                            )  # Go right at x=M-1

                            # if we are at the top
                            if i_y == Constants.N - 1:
                                P[i][i + t_step + z_step][u] = (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_STAY
                                    ]
                                    + Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_NORTH
                                    ]
                                ) * Constants.P_V_TRANSITION[
                                    1
                                ]  # Stay where you are if no wind or pushed north
                                P[i][i + t_step + z_step - y_step][u] = (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_SOUTH
                                    ]
                                ) * Constants.P_V_TRANSITION[
                                    1
                                ]  # Go south at y!=0
                            # if we are at the bottom
                            elif i_y == 0:
                                P[i][i + t_step + z_step][u] = (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_STAY
                                    ]
                                    + Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_SOUTH
                                    ]
                                ) * Constants.P_V_TRANSITION[
                                    1
                                ]  # Stay where you are if no wind or pushed south
                                P[i][i + t_step + z_step + y_step][u] = (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_NORTH
                                    ]
                                ) * Constants.P_V_TRANSITION[
                                    1
                                ]  # Go north at y!=N-1
                            # not at extremum in y
                            else:
                                P[i][i + t_step + z_step][u] = (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_STAY
                                    ]
                                ) * Constants.P_V_TRANSITION[
                                    1
                                ]  # Stay where you are if no wind
                                P[i][i + t_step + z_step + y_step][u] = (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_NORTH
                                    ]
                                ) * Constants.P_V_TRANSITION[
                                    1
                                ]  # Go north at y!=N-1

                                P[i][i + t_step + z_step - y_step][u] = (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_SOUTH
                                    ]
                                ) * Constants.P_V_TRANSITION[
                                    1
                                ]  # Go south at y!=0

                        # check PDF
                        non_zero = P[i, P[i, :, u] != 0, u]
                        if not np.isclose(np.sum(non_zero), 1):
                            print(non_zero)
                            print(np.sum(non_zero))
                            raise ValueError("Not a valid PDF")
    return P


def compute_transition_probabilities_sparse(Constants):
    """Computes the probability transition matrix P in a sparse format.

    Because scipy sparse only natively supports 2D sparse matrices,
    and because the input space is discrete and small (L=3), the function returns
    a list of length L of K,K sparse matrices
    """

    from scipy.sparse import csr_array
    from scipy.sparse import coo_array

    K = Constants.T * Constants.D * Constants.N * Constants.M
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)
    P = []

    g_x_step = 1
    g_x_step_wrap = -(Constants.M - 1)
    y_step = Constants.M
    g_z_step = y_step * Constants.N
    g_t_step = g_z_step * Constants.D
    g_t_step_wrap = -(Constants.T - 1) * g_t_step

    for u in input_space:
        row_idx = []
        col_idx = []
        values = []
        for i_t in range(Constants.T):
            for i_z in range(Constants.D):
                for i_y in range(Constants.N):
                    for i_x in range(Constants.M):
                        i = i_x + i_y * y_step + i_z * g_z_step + i_t * g_t_step
                        t_step = g_t_step if i_t != Constants.T - 1 else g_t_step_wrap
                        x_right = g_x_step if i_x != Constants.M - 1 else g_x_step_wrap
                        x_left = -g_x_step if i_x != 0 else -g_x_step_wrap
                        P_V_stay = (
                            1 if u == Constants.V_STAY else Constants.P_V_TRANSITION[0]
                        )
                        # P_V_move = (
                        #     0 if u == Constants.V_STAY else Constants.P_V_TRANSITION[1]
                        # )
                        z_step = g_z_step if u == Constants.V_UP else -g_z_step

                        if (u == Constants.V_UP and i_z == Constants.D - 1) or (
                            u == Constants.V_DOWN and i_z == 0
                        ):
                            continue

                        # ----------------no vertical displacement ---------------------------

                        row_idx.append(i)
                        col_idx.append(i + t_step + x_left)
                        values.append(
                            Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_WEST]
                            * P_V_stay
                        )

                        row_idx.append(i)
                        col_idx.append(i + t_step + x_right)
                        values.append(
                            Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_EAST]
                            * P_V_stay
                        )

                        # if we are at the top
                        if i_y == Constants.N - 1:
                            row_idx.append(i)
                            col_idx.append(i + t_step)
                            values.append(
                                (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_STAY
                                    ]
                                    + Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_NORTH
                                    ]
                                )
                                * P_V_stay
                            )

                            row_idx.append(i)
                            col_idx.append(i + t_step - y_step)
                            values.append(
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_SOUTH]
                                * P_V_stay
                            )
                        # if we are at the bottom
                        elif i_y == 0:
                            row_idx.append(i)
                            col_idx.append(i + t_step)
                            values.append(
                                (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_STAY
                                    ]
                                    + Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_SOUTH
                                    ]
                                )
                                * P_V_stay
                            )

                            row_idx.append(i)
                            col_idx.append(i + t_step + y_step)
                            values.append(
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_NORTH]
                                * P_V_stay
                            )
                        # not at extremum in y
                        else:
                            row_idx.append(i)
                            col_idx.append(i + t_step)
                            values.append(
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_STAY]
                                * P_V_stay
                            )

                            row_idx.append(i)
                            col_idx.append(i + t_step + y_step)
                            values.append(
                                (
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_NORTH
                                    ]
                                )
                                * P_V_stay
                            )

                            row_idx.append(i)
                            col_idx.append(i + t_step - y_step)
                            values.append(
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_SOUTH]
                                * P_V_stay
                            )
                        # -----------------------------------------------------

                        # ------------vertical displacement---------------------
                        if u == Constants.V_UP or u == Constants.V_DOWN:
                            row_idx.append(i)
                            col_idx.append(i + t_step + z_step + x_left)
                            values.append(
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_WEST]
                                * Constants.P_V_TRANSITION[1]
                            )

                            row_idx.append(i)
                            col_idx.append(i + t_step + z_step + x_right)
                            values.append(
                                Constants.P_H_TRANSITION[i_z].P_WIND[Constants.H_EAST]
                                * Constants.P_V_TRANSITION[1]
                            )

                            # if we are at the top
                            if i_y == Constants.N - 1:
                                row_idx.append(i)
                                col_idx.append(i + t_step + z_step)
                                values.append(
                                    (
                                        Constants.P_H_TRANSITION[i_z].P_WIND[
                                            Constants.H_STAY
                                        ]
                                        + Constants.P_H_TRANSITION[i_z].P_WIND[
                                            Constants.H_NORTH
                                        ]
                                    )
                                    * Constants.P_V_TRANSITION[1]
                                )

                                row_idx.append(i)
                                col_idx.append(i + t_step + z_step - y_step)
                                values.append(
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_SOUTH
                                    ]
                                    * Constants.P_V_TRANSITION[1]
                                )
                            # if we are at the bottom
                            elif i_y == 0:
                                row_idx.append(i)
                                col_idx.append(i + t_step + z_step)
                                values.append(
                                    (
                                        Constants.P_H_TRANSITION[i_z].P_WIND[
                                            Constants.H_STAY
                                        ]
                                        + Constants.P_H_TRANSITION[i_z].P_WIND[
                                            Constants.H_SOUTH
                                        ]
                                    )
                                    * Constants.P_V_TRANSITION[1]
                                )

                                row_idx.append(i)
                                col_idx.append(i + t_step + z_step + y_step)
                                values.append(
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_NORTH
                                    ]
                                    * Constants.P_V_TRANSITION[1]
                                )
                            # not at extremum in y
                            else:
                                row_idx.append(i)
                                col_idx.append(i + t_step + z_step)
                                values.append(
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_STAY
                                    ]
                                    * Constants.P_V_TRANSITION[1]
                                )

                                row_idx.append(i)
                                col_idx.append(i + t_step + z_step + y_step)
                                values.append(
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_NORTH
                                    ]
                                    * Constants.P_V_TRANSITION[1]
                                )

                                row_idx.append(i)
                                col_idx.append(i + t_step + z_step - y_step)
                                values.append(
                                    Constants.P_H_TRANSITION[i_z].P_WIND[
                                        Constants.H_SOUTH
                                    ]
                                    * Constants.P_V_TRANSITION[1]
                                )

        P.append(coo_array((values, (row_idx, col_idx)), shape=(K, K)).tocsr())
    return P
