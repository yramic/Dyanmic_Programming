"""Solution."""
from matplotlib.dates import SA
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
import sklearn.gaussian_process as gp
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        # GP for logP function f
        self.f_kernel = gp.kernels.Matern(nu=2.5, length_scale = 0.5, length_scale_bounds="fixed") + \
                        gp.kernels.WhiteKernel(noise_level=0.15 ** 2)
        self.f_model = gp.GaussianProcessRegressor(kernel = self.f_kernel, n_restarts_optimizer=10)

        # GP for SA function v
        self.v_kernel = gp.kernels.ConstantKernel(4) + \
                        gp.kernels.DotProduct() + \
                        gp.kernels.DotProduct() + \
                        gp.kernels.Matern(nu=2.5, length_scale=0.5, length_scale_bounds="fixed") + \
                        gp.kernels.WhiteKernel(noise_level=0.0001 ** 2)
        self.v_model = gp.GaussianProcessRegressor(kernel = self.v_kernel, n_restarts_optimizer=10)
        
        # store data points
        self.x_data = []
        self.v_data = []
        self.f_data = []
        
        # other parameters
        self.N = 0
        self.kappa = 0.001 # exploration / exploitation tradeoff parameter for expected improvement
        self._lambda = 5 # weighting for constraint expected improvement

        # Variables for normalizing the Data!
        self.x_mean = None
        self.x_std = None
        self.x_normalized = []
        self.v_data_tot = []
        self.f_data_tot = []


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        
        x_next = self.optimize_acquisition_function()
        # Clip the recommendation to the specified domain [0, 10]
        x_next = np.array(np.clip(x_next, *DOMAIN[0]))

        return x_next
    
    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        f_mean, f_sigma = self.f_model.predict(x, return_std= True)
        v_mean, v_sigma = self.v_model.predict(x, return_std= True)
        
        best_f = max(self.f_data)
        best_v = max(self.v_data)

        # compute expected improvement for f
        zf = (f_mean - (best_f+ self.kappa)) / f_sigma
        eif = (f_mean - best_f) * norm.cdf(zf) + f_sigma * norm.pdf(zf)

        # compute expected improvement for v
        zv = (v_mean - (best_v + self.kappa)) / v_sigma
        eiv = (v_mean - best_v) * norm.cdf(zv) + v_sigma * norm.pdf(zv)

        # compute joint expected improvement adjusted with lambda
        return eif - eiv * self._lambda
    
    #################### ADDED #####################

    def normalize_input(self):
        # if self.x_mean is None or self.x_std is None:
        self.x_mean = np.mean(self.x_data)
        self.x_std = np.std(self.x_data)

        # Normalization of the input data:
        self.x_normalized = (self.x_data - self.x_mean) / self.x_std
    
    def store_data(self, x: float, f: float, v: float):
        self.x_data.append(x)
        self.f_data.append(f)  
        self.v_data.append(v)

        x = np.atleast_2d(x)
        f = np.atleast_2d(f)
        v = np.atleast_2d(v)

        self.N += 1
    ################################################

        
    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.

        self.x_data.append(x)
        self.f_data.append(f)  
        self.v_data.append(v)
        
        x = np.atleast_2d(x)
        f = np.atleast_2d(f)
        v = np.atleast_2d(v)

        self.N += 1
        
        # fit GPs for f and v
        self.f_model.fit(np.array(self.x_data).reshape(-1, 1), np.array(self.f_data).reshape(-1, 1))
        self.v_model.fit(np.array(self.x_data).reshape(-1, 1), np.array(self.v_data).reshape(-1, 1))

    ########################### ADDED ################################
    def GP_fit(self):
        for i in range(self.N):
            self.f_model.fit(np.array(self.x_normalized[i]).reshape(-1, 1), np.array(self.f_data_tot[i]).reshape(-1, 1))
            self.v_model.fit(np.array(self.x_normalized[i]).reshape(-1, 1), np.array(self.v_data_tot[i]).reshape(-1, 1))
    ##################################################################

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        
        # checks if the corresponding SA (v_data) is below the safety threshold 
        # and if the objective value (f_data) is greater than the current maximum
        
        max_f = -math.inf
        index = -1

        for i in range(self.N):
            if self.v_data[i] < SAFETY_THRESHOLD and self.f_data[i] > max_f:
                max_f = self.f_data[i]
                index = i
        return self.x_data[index]
        
    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        x = np.arange(0, 11, 0.1)
        plt.plot(x, [f(x_i) for x_i in x], color='red', label='Bioavailability f')
        plt.plot(x, self.f_model.predict(x.reshape(-1,1)), color='black', label='f GP estimation')
        plt.plot(x, self.v_model.predict(x.reshape(-1,1)), color='blue', label='v GP estimation')
        plt.scatter(self.x_data, [f(x_i) for x_i in self.x_data], color='green', label='Sample points at f')
        plt.show()


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    ##################### ADDED #####################
    for j in range(20): # Initially: 20
        # Get next recommendation
        x = agent.next_recommendation()

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.store_data(x, obj_val, cost_val)

    agent.normalize_input()

    agent.GP_fit()
    #################################################

    # # Loop until budget is exhausted
    # for j in range(20): # Initially: 20
    #     # Get next recommendation
    #     x = agent.next_recommendation()

    #     # Obtain objective and constraint observation
    #     obj_val = f(x) + np.random.randn()
    #     cost_val = v(x) + np.random.randn()
    #     agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')

    agent.plot()

if __name__ == "__main__":
    main()
