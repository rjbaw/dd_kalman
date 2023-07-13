#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from kalman import test

def simple_mass_spring_damp(T, Q, R, P, Ts=0.1):
    '''
    m x''(t) + c x'(t) + k x(t) = 0
    m, mass
    c, damping coefficient
    k, spring constant
    F, force
    Ts, time step
    N, simulation length

    E[ξ(t)ξ(t)^T] = Q
    E[ω(t)ω(t)^T] = R

    x(t+1) = A x(t) + ξ(t)
    y(t) = H x(t) + ω(t)
    {ξ}_t∈Z, uncorrelated zero-mean process
    {ω}_t∈Z, measurement noise vectors
    x(t) ∈ R^n, state of the system
    y(t) ∈ R^m
    '''
    # (A,H) are known
    H = np.array([[1.0, 0.0]])
    C = 4
    K = 2
    M = 20
    F = 1 
    N = int(T/Ts)
    A = np.array([[1, Ts], [Ts*(-K/M), 1+Ts*(-C/M)]])
    B = np.array([0, Ts*(1/M)])
    u = F

    n = A.shape[0]
    m = H.shape[0]
    x = np.zeros((n,N+1))
    y = np.zeros((m,N))
    ξ = np.random.normal(0, np.sqrt(Q),size=(N+1,n)).T * np.sqrt(Ts) 
    ω = np.random.normal(0, R, size=(N,m)).T * 1/np.sqrt(Ts)
    m_0 = np.random.normal(0, P)
    x[:,0] = [m_0,0]
    for k in range(N):
        x[:,k+1] = A @ x[:,k] + B * u + ξ[:,k]
        y[:,k] = H @ x[:,k] + ω[:,k]
    return x[:,:-1], y, A, H

if __name__ == "__main__":

    #nsim = 20
    #batch_sizes = [10, 20, 30]

    torch.manual_seed(100)
    np.random.seed(100)
    n, m = 2, 1
    Q = np.ones(n) * 0.1
    R = 0.1
    P = 0.05
    Ts = 0.1
    T = 100
    steps = 5000
    N = int(T/Ts)

    print('Horizon:', T)
    x, y, A, H = simple_mass_spring_damp(T, Q, R, P)
    t = np.arange(0, T, Ts)
    model = test.Model(A, H)
    losses, preds, weights = test.sgd(model, y, N, steps)
    print("Kalman Gain L: ", weights[-1])
    
    # P = APA^T - APH^T (HPH^T + R)^(-1) HPA^T + Q 
    Q = np.ones((n,n)) * 0.01
    P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
    L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    print("P: ", P)
    print("L: ", L)

    for i in range(y.shape[0]):

        plt.figure()
        plt.plot(t, x[i])
        plt.plot(t, preds[i], color='r')
        plt.title('Simulation of Mass-Spring-Damper System')
        plt.xlabel('t [s]')
        plt.ylabel('x(t)')
        plt.grid()
        plt.legend(["x", "filtered x"])
        plt.show()

        plt.figure()
        plt.plot(t, y[i])
        plt.plot(t, preds[i], color='r')
        plt.title('Simulation of Mass-Spring-Damper System')
        plt.xlabel('t [s]')
        plt.ylabel('y(t)')
        plt.grid()
        plt.legend(["y", "filtered y"])
        plt.show()

        plt.figure()
        plt.plot(losses)
        plt.title('Loss Function')
        plt.xlabel('n [steps]')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(weights)
        plt.axhline(L[i], linestyle='--')
        plt.title('Kalman Gain')
        plt.xlabel('n [steps]')
        plt.legend(["Gain", "Optimal Gain"])
        plt.grid()
        plt.show()
