"""
 ComputeStageCosts.py

 Python function template to compute the stage cost matrix.

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


def compute_stage_cost(Constants):
    """Computes the stage cost matrix for the given problem.

    It is of size (K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - G[i,l] corresponds to the cost incurred when using input l
            at state i.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Stage cost matrix G of shape (K,L)
    """
    K = Constants.T * Constants.D * Constants.N * Constants.M
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)
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
    # Care has to be taken for x and t, because they wrap around

    def compute_horizontal_cities_cost():
        cities_cost = np.empty((Constants.N, Constants.M))
        for i_y in range(Constants.N):
            for i_x in range(Constants.M):
                cost = 0
                # city location has form [y,x]
                for city in Constants.CITIES_LOCATIONS:
                    cost += np.sqrt(
                        np.min(
                            [
                                (i_x - Constants.M - city[1]) ** 2,
                                (i_x - city[1]) ** 2,
                                (i_x + Constants.M - city[1]) ** 2,
                            ]
                        )
                        + pow(i_y - city[0], 2)
                    )

                cities_cost[i_y][i_x] = cost
        return cities_cost

    def compute_solar_cost():
        solar_cost = np.empty((Constants.T, Constants.M))
        for i_t in range(Constants.T):
            x_sun = np.floor(
                (Constants.M - 1) * (Constants.T - 1 - i_t) / (Constants.T - 1)
            )
            for i_x in range(Constants.M):
                solar_cost[i_t][i_x] = np.min(
                    [
                        (i_x - Constants.M - x_sun) ** 2,
                        (i_x - x_sun) ** 2,
                        (i_x + Constants.M - x_sun) ** 2,
                    ]
                )
        return solar_cost

    discount_factor = 1 / Constants.ALPHA
    x_step = 1
    y_step = Constants.M
    z_step = y_step * Constants.N
    t_step = z_step * Constants.D
    cities_cost = compute_horizontal_cities_cost()
    solar_cost = compute_solar_cost()
    G = np.ones((K, L)) * np.inf

    for i_t in range(Constants.T):
        for i_z in range(Constants.D):
            for i_y in range(Constants.N):
                for i_x in range(Constants.M):
                    i = i_x + i_y * y_step + i_z * z_step + i_t * t_step

                    # not allowed to go up when at the top
                    if i_z == Constants.D - 1:
                        G[i][Constants.V_STAY] = G[i][
                            Constants.V_DOWN
                        ] = discount_factor * (
                            cities_cost[i_y][i_x]
                            + Constants.LAMBDA_LEVEL * i_z
                            + Constants.LAMBDA_TIMEZONE * solar_cost[i_t][i_x]
                        )

                    # not allowed to go down when at the bottom
                    elif i_z == 0:
                        G[i][Constants.V_STAY] = G[i][
                            Constants.V_UP
                        ] = discount_factor * (
                            cities_cost[i_y][i_x]
                            # + Constants.LAMBDA_LEVEL * i_z # can remove because i_z is zero
                            + Constants.LAMBDA_TIMEZONE * solar_cost[i_t][i_x]
                        )

                    # can choose any input when in a middle layer
                    else:
                        G[i][:] = discount_factor * (
                            cities_cost[i_y][i_x]
                            + Constants.LAMBDA_LEVEL * i_z
                            + Constants.LAMBDA_TIMEZONE * solar_cost[i_t][i_x]
                        )

    return G
