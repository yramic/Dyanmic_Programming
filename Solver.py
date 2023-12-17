"""
 Solver.py

 Python function template to solve the discounted stochastic
 shortest path problem.

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


def solution(P, G, alpha):
    """Computes the optimal cost and the optimal control input for each
    state of the state space solving the discounted stochastic shortest
    path problem by:
            - Value Iteration;
            - Policy Iteration;
            - Linear Programming;
            - or a combination of these.

    Args:
        P  (np.array): A (K x K x L)-matrix containing the transition probabilities
                       between all states in the state space for all control inputs.
                       The entry P(i, j, l) represents the transition probability
                       from state i to state j if control input l is applied
        G  (np.array): A (K x L)-matrix containing the stage costs of all states in
                       the state space for all control inputs. The entry G(i, l)
                       represents the cost if we are in state i and apply control
                       input l
        alpha (float): The discount factor for the problem

    Returns:
        np.array: The optimal cost to go for the discounted stochastic SPP
        np.array: The optimal control policy for the discounted stochastic SPP

    """
    # TODO: Don't know if this is allowed
    from Constants import Constants

    K = G.shape[0]

    J_opt = np.ones(K)
    J_opt_prev = np.ones(K)
    u_opt = np.zeros(K)
    input_space = [Constants.V_DOWN, Constants.V_STAY, Constants.V_UP]

    x_step = 1
    y_step = Constants.M
    z_step = y_step * Constants.N
    t_step = z_step * Constants.D
    t_step_wrap = -(Constants.T - 1) * t_step
    x_step_wrap = -(Constants.M - 1)

    err = 1e-100
    verbose = True

    # solvers:
    # 0 = Standard Value Iteration (passing)
    # 1 = Gauss Seidel value iteration (passing)
    # 2 = Linear Programming (passing)
    # 3 = Policy Iteration (passing)
    solver = 3

    def get_possible_next_states(i_t, i_z, i_y, i_x, i, u):
        j = []

        # the state corresponding to one time step and no spatial steps
        j_stay = i + t_step_wrap if i_t == Constants.T - 1 else i + t_step

        # regardless of the input applied, the balloon can be in the same place or one to the east or west in the next time step
        j.append(j_stay)
        j.append(j_stay + x_step_wrap if i_x == Constants.M - 1 else j_stay + x_step)
        j.append(j_stay - x_step_wrap if i_x == 0 else j_stay - x_step)

        # north west movement in the current z-plane, depending on bounds
        if i_y != Constants.N - 1:
            j.append(j_stay + y_step)
        if i_y != 0:
            j.append(j_stay - y_step)

        # add states accounting for positive vertical movement if input is UP and not at top
        if u == Constants.V_UP and i_z != Constants.D - 1:
            j_up = j_stay + z_step
            j.append(j_up)
            j.append(j_up + x_step_wrap if i_x == Constants.M - 1 else j_stay + x_step)
            j.append(j_up - x_step_wrap if i_x == 0 else j_stay - x_step)
            if i_y != Constants.N - 1:
                j.append(j_up + y_step)
            if i_y != 0:
                j.append(j_up - y_step)

        # add states accounting for negative vertical movement if input is DOWN and not at bottom
        if u == Constants.V_DOWN and i_z != 0:
            j_down = j_stay - z_step
            j.append(j_down)
            j.append(
                j_down + x_step_wrap if i_x == Constants.M - 1 else j_stay + x_step
            )
            j.append(j_down - x_step_wrap if i_x == 0 else j_stay - x_step)
            if i_y != Constants.N - 1:
                j.append(j_down + y_step)
            if i_y != 0:
                j.append(j_down - y_step)

        return j

    def expected_cost(G, P, i, i_t, i_z, i_y, i_x):
        cost = np.empty(3)
        for u in input_space:
            cost[u] = G[i, u]
            for j in get_possible_next_states(
                i_t=i_t, i_z=i_z, i_y=i_y, i_x=i_x, i=i, u=u
            ):
                cost += P[i, j, u] * J_opt[j]

        return cost

    # TODO implement Value Iteration, Policy Iteration,
    #      Linear Programming or a combination of these

    # ------------Gauss-Seidel VI --------------------------------
    # (Value Iteration with in place cost updates) (working)
    if solver == 1:
        iter = 0
        while iter < 300:
            iter += 1
            for i_t in range(Constants.T):
                for i_z in range(Constants.D):
                    for i_y in range(Constants.N):
                        for i_x in range(Constants.M):
                            i = i_x + i_y * y_step + i_z * z_step + i_t * t_step
                            # cost = expected_cost(
                            #     G, P, i=i, i_t=i_t, i_z=i_z, i_y=i_y, i_x=i_x
                            # )
                            cost = np.empty(len(input_space))
                            for u in input_space:
                                # print(P[i, P[i, :, u] != 0, u])
                                cost[u] = G[i, u] + alpha * np.sum(P[i, :, u] * J_opt)
                            J_opt[i] = np.min(cost)
                            u_opt[i] = np.argmin(cost)

            current_error = np.max(np.abs(J_opt_prev - J_opt)) / np.max(np.abs(J_opt))
            # changed rtol from 1e-4
            if np.allclose(J_opt, J_opt_prev, rtol=1e-06, atol=1e-07):
                # if current_error < err:
                break
            else:
                J_opt_prev = np.copy(J_opt)
                if verbose:
                    print("Iteration {} with error {:.4f}".format(iter, current_error))
        return J_opt, u_opt
    # ----------------------------------------------------------

    # ---------------Value Iteration-----------------------------
    if solver == 0:
        iter = 0
        while iter < 200:
            iter += 1
            for i_t in range(Constants.T):
                for i_z in range(Constants.D):
                    for i_y in range(Constants.N):
                        for i_x in range(Constants.M):
                            i = i_x + i_y * y_step + i_z * z_step + i_t * t_step
                            # cost = expected_cost(
                            #     G, P, i=i, i_t=i_t, i_z=i_z, i_y=i_y, i_x=i_x
                            # )
                            cost = np.empty(len(input_space))
                            for u in input_space:
                                # print(P[i, P[i, :, u] != 0, u])
                                cost[u] = G[i, u] + alpha * np.sum(
                                    P[i, :, u] * J_opt_prev
                                )

                            J_opt[i] = np.min(cost)
                            u_opt[i] = np.argmin(cost)

            current_error = np.max(np.abs(J_opt_prev - J_opt)) / np.max(np.abs(J_opt))
            if np.allclose(J_opt, J_opt_prev, rtol=1e-04, atol=1e-07):
                # if current_error < err:
                break
            else:
                # test = np.allclose(J_opt, J_opt_prev, rtol=1e-04, atol=1e-07)
                J_opt_prev = J_opt.copy()

                if verbose:
                    # print(
                    #     "Iteration {} with error {:.4f} and alternate convergence {}".format(
                    #         iter, current_error, test
                    #     )
                    # )
                    print("Iteration {} with error {:.4f}".format(iter, current_error))

                    # print("Iteration {} ".format(iter))

        print("Number of iterations: {}".format(iter))
        # print(J_opt)
        # print(J_opt_prev)
        # print(u_opt)
        return J_opt, u_opt

    # ------------------------------------------------------------------

    # ----------------------Linear Programming------------------------------
    if solver == 2:
        from scipy.optimize import linprog

        c = -np.ones(K)
        L = len(input_space)
        A_ub = np.zeros((K * L, K))
        b_ub = np.zeros(K * L)

        for i in range(L):
            A_ub[K * i : K * (i + 1) if i != L - 1 else None, :] = (
                np.eye(K) - alpha * P[:, :, i]
            )
            b_ub[K * i : K * (i + 1) if i != L - 1 else None] = G[:, i]

        valid_indices = ~np.isinf(b_ub)
        b_ub = b_ub[valid_indices]
        A_ub = A_ub[valid_indices, :]
        # b_ub = -b_ub

        result = linprog(
            c=c, A_ub=A_ub, b_ub=b_ub, method="highs", bounds=[(None, None)]
        )
        if result.status != 0:
            raise ValueError("Linear program failed")

        J_opt = result.x
        # derive optimal policy from optimal cost
        for i_t in range(Constants.T):
            for i_z in range(Constants.D):
                for i_y in range(Constants.N):
                    for i_x in range(Constants.M):
                        i = i_x + i_y * y_step + i_z * z_step + i_t * t_step
                        cost = np.empty(len(input_space))
                        for u in input_space:
                            cost[u] = G[i, u] + alpha * np.sum(P[i, :, u] * J_opt)
                        u_opt[i] = np.argmin(cost)

        return J_opt, u_opt

    # --------------------------------------------------------------------------------

    # --------------------Policy Iteration--------------------------------------------
    if solver == 3:
        from scipy.optimize import linprog

        u_opt = np.ones(K)  # V_STAY at all starting states is a proper policy
        A = np.eye(K) - alpha * P[:, :, 1]  # I-alpha*P(mu_0(i))
        b = G[:, 1]  # constraints are initially the costs associated with mu_0(i)=STAY
        c = -np.ones(
            K
        )  # objective function is the negative sum of V(i) over all states
        result = linprog(c=c, A_eq=A, b_eq=b)
        J_opt = result.x
        iter = 0
        P_mu = np.empty((K, K))
        G_mu = np.empty(K)
        while iter < 10:
            iter += 1
            for i_t in range(Constants.T):
                for i_z in range(Constants.D):
                    for i_y in range(Constants.N):
                        for i_x in range(Constants.M):
                            i = i_x + i_y * y_step + i_z * z_step + i_t * t_step
                            cost = np.empty(len(input_space))
                            for u in input_space:
                                cost[u] = G[i, u] + alpha * np.sum(P[i, :, u] * J_opt)
                            u_opt[i] = np.argmin(cost)
                            P_mu[i, :] = P[i, :, int(u_opt[i])]
                            G_mu[i] = G[i, int(u_opt[i])]
            A = np.eye(K) - alpha * P_mu
            result = linprog(c=c, A_eq=A, b_eq=G_mu)
            J_opt = result.x

            current_error = np.max(np.abs(J_opt_prev - J_opt)) / np.max(np.abs(J_opt))
            # changed rtol from 1e-4
            if np.allclose(J_opt, J_opt_prev, rtol=1e-05, atol=1e-07):
                # if current_error < err:
                break
            else:
                # test = np.allclose(J_opt, J_opt_prev, rtol=1e-04, atol=1e-07)
                J_opt_prev = J_opt.copy()

                if verbose:
                    # print(
                    #     "Iteration {} with error {:.4f} and alternate convergence {}".format(
                    #         iter, current_error, test
                    #     )
                    # )
                    print("Iteration {} with error {:.4f}".format(iter, current_error))

                    # print("Iteration {} ".format(iter))

        return J_opt, u_opt
    # ---------------------------------------------------------------------------------


def freestyle_solution(Constants):
    """Computes the optimal cost and the optimal control input for each
    state of the state space solving the discounted stochastic shortest
    path problem with a 200 MiB memory cap.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the discounted stochastic SPP
        np.array: The optimal control policy for the discounted stochastic SPP
    """
    K = Constants.T * Constants.D * Constants.N * Constants.M

    J_opt = np.zeros(K)
    u_opt = np.zeros(K)

    # TODO implement a solution that not necessarily adheres to
    #      the solution template. You are free to use
    #      compute_transition_probabilities and
    #      compute_stage_cost, but you are also free to introduce
    #      optimizations.

    return J_opt, u_opt
