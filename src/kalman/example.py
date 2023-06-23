#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import numba
from numba import jit

## Test functions
@jit(nopython=True, parallel=True)
def gaussians(x, means, widths):
    '''Return the value of gaussian kernels.
    
    x - location of evaluation
    means - array of kernel means
    widths - array of kernel widths
    '''
    n = means.shape[0]
    result = np.exp( -0.5 * ((x - means) / widths)**2 ) / widths
    return result / np.sqrt(2 * np.pi) / n

@jit(nopython=True, parallel=True)
def monte_carlo_pi_parallel(nsamples):
    acc = 0
    # Only change is here
    for _ in numba.prange(nsamples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

@jit(nopython=True, parallel=True)
def mse(nsamples):
    return

means = np.random.uniform(-1, 1, size=1000000)
widths = np.random.uniform(0.1, 0.3, size=1000000)
print(gaussians(0.4, means, widths))
print(monte_carlo_pi_parallel(100))
