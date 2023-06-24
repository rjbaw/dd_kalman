import numpy as np
from kalman import test

means = np.random.uniform(-1, 1, size=1000000)
widths = np.random.uniform(0.1, 0.3, size=1000000)
print(test.gaussians(0.4, means, widths))
print(test.monte_carlo_pi_parallel(100))

# (A,H) are known
n, m = 1, 1
var_state = 0.1
covar_state = 0.05
var_obs = 0.05
nsim = 20
T = [100, 200, 500]
M = [10, 20, 30]

def simple_mass_spring_damp():
    return
