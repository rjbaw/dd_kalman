#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
from kalman import test

def simple_mass_spring_damp(T, Q, R, P, Ts=0.1):
    '''
    m x''(t) + c x'(t) + k x(t) = 0
    m, mass
    c, damping coefficient
    k, spring constant

    E[ξ(t)ξ(t)^T] = Q
    E[ω(t)ω(t)^T] = R

    x(t+1) = A x(t) + ξ(t)
    y(t) = H x(t) + ω(t)
    {ξ}_t∈Z, uncorrelated zero-mean process
    {ω}_t∈Z, measurement noise vectors
    '''
    # (A,H) are known

    n, m = 2, 1
    H = np.array([1.0, 0.0])
    c = 4 # Damping constant
    k = 1 # Stiffness of the spring
    m = 20 # Mass
    F = 1 # Force
    Ts = 0.1
    N = int(T/Ts) # Simulation length
    x = np.zeros((2,N))
    y = np.zeros(N)
    ξ = np.random.normal(0, Q, size=N)
    ω = np.random.normal(0, R, size=N)
    ξ = np.zeros(N)
    x[:,0] = [0,0] # Initial Position

    A = np.array([[1, Ts], [Ts*(-k/m), 1+Ts*(-c/m)]])
    B = np.array([0, Ts*(1/m)])
    for k in range(N-1):
        x[:,k+1] = A @ x[:,k] + B * F + ξ[k]
        y[k+1] = H @ x[:,k] + ω[k]
    return y, A, H

if __name__ == "__main__":

    nsim = 20
    Q = 0.1
    R = 0.1
    P = 0.05
    Ts = 0.1
    N_array = [100, 200, 500]
    batch_sizes = [10, 20, 30]

    for N in N_array:
        print('Horizon:', N)
        x = torch.arange(N)
        y, A, H = simple_mass_spring_damp(N, Q, R, P)
        t = np.arange(0, N, Ts)
        model = test.Model(y, A, H, P)
        y = torch.Tensor(y)
        losses, preds = test.sgd(model, y, int(N/Ts))
        plt.figure()
        plt.plot(t, y)
        plt.plot(t, preds, color='r')
        plt.title('Simulation of Mass-Spring-Damper System')
        plt.xlabel('t [s]')
        plt.ylabel('x(t)')
        plt.grid()
        plt.legend(["x"])
        plt.show()
        plt.figure()
        plt.plot(losses)
        plt.show()
