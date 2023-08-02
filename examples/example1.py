#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from kalman import test

def series_mass_spring_damp(T, q, r, P_0, Ts=0.1, n_series=1):
    '''
    m x''(t) + c x'(t) + k x(t) = 0
    mass (kg)
    cdamp, damping coefficient (N/m^2)
    kspring, spring constant (N/m^2)
    force (N)
    Ts, time step (s)
    N, simulation length (simulation time x time steps)

    E[ξ(t)ξ(t)^T] = Q
    E[ω(t)ω(t)^T] = R

    x(t+1) = A x(t) + ξ(t)
    y(t) = H x(t) + ω(t)
    {ξ}_t∈Z, uncorrelated zero-mean process
    {ω}_t∈Z, measurement noise vectors
    x(t) ∈ R^n, state of the system
    y(t) ∈ R^m
    '''
    m = 1
    cdamp = 4
    kspring = 2
    mass = 20
    force = 10 
    N = int(T/Ts)

    M = np.eye(n_series) * mass
    K = np.zeros((n_series,n_series))
    C = np.zeros((n_series,n_series))
    K[0,0] = -kspring
    C[0,0] = -cdamp
    for idx in range(n_series-1):
        K[idx:idx+2,idx:idx+2] += np.array([[-kspring, kspring],
                                            [kspring,-kspring]])
        C[idx:idx+2,idx:idx+2] += np.array([[-cdamp, cdamp],
                                            [cdamp,-cdamp]])
    
    #K = np.diag(np.ones(n_series-1)*kspring, -1) \
    #    + np.diag(np.ones(n_series)*(-n_series*kspring), 0) \
    #    + np.diag(np.ones(n_series-1)*kspring, 1)
    #C = np.diag(np.ones(n_series-1)*cdamp, -1) \
    #    + np.diag(np.ones(n_series)*(-n_series*cdamp), 0) \
    #    + np.diag(np.ones(n_series-1)*cdamp, 1)

    A = Ts * ( np.block([[np.zeros((n_series,n_series)), np.eye(n_series)],
                         [np.linalg.inv(M) @ K, np.linalg.inv(M) @ C]]) )
    A += np.eye(A.shape[0])

    n = A.shape[0]
    H = np.zeros((m,n))
    H[0][-2] = 1 #choose output
    B = Ts * ( 1/mass * np.append(np.zeros(n-1), 1) )
    Q = np.eye(n) * q
    R = np.eye(m) * r
    u = force

    x = np.zeros((n,N+1))
    y = np.zeros((m,N))
    m_0 = np.random.normal(0, P_0, size=n)
    ξ = np.random.multivariate_normal(np.zeros(n), Q, N+1).T 
    ω = np.random.multivariate_normal(np.zeros(m), R, N).T
    x[:,0] = m_0 + B * u
    for k in range(N):
        x[:,k+1] = A @ x[:,k] + B * u + ξ[:,k]
        y[:,k] = H @ x[:,k] + ω[:,k]
        #x[:,k+1] = A @ x[:,k] + 1/mass * ξ[:,k]
        #y[:,k] = H @ x[:,k] + 1/mass * ω[:,k]
        #x[:,k+1] = A @ x[:,k]
        #y[:,k] = H @ x[:,k]

    print('|-------------------------------------|')
    print('  K:\n', K)
    print('  C:\n', C)
    print('  B:\n', B)
    print('|-------------------------------------|')

    return x[:,:-1], y, A, H, Q, R

def simple_mass_spring_damp(T, Q, R, P_0, Ts=0.1):
    '''
    m x''(t) + c x'(t) + k x(t) = 0
    mass (kg)
    cdamp, damping coefficient (N/m^2)
    kspring, spring constant (N/m^2)
    force (N)
    Ts, time step (s)
    N, simulation length (simulation time x time steps)

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
    n, m = 2, 1
    cdamp = 4
    kspring = 2
    mass = 20
    force = 1 
    u = force
    N = int(T/Ts)
    A = np.eye(2) + Ts * np.array([[0, 1], [-kspring/mass, -cdamp/mass]])
    B = Ts * np.array([0, 1/mass])
    H = np.array([[1.0, 0.0]])

    x = np.zeros((n,N+1))
    y = np.zeros((m,N))
    m_0 = np.random.normal(0, P_0, size=n)
    #ξ = np.random.multivariate_normal(np.zeros(n), Q, N+1).T * np.sqrt(Ts) 
    #ω = np.random.multivariate_normal(np.zeros(m), R, N).T * 1/np.sqrt(Ts)
    ξ = np.random.multivariate_normal(np.zeros(n), Q, N+1).T 
    ω = np.random.multivariate_normal(np.zeros(m), R, N).T
    x[:,0] = m_0 + B * u

    for k in range(N):
        x[:,k+1] = A @ x[:,k] + B * u + ξ[:,k]
        y[:,k] = H @ x[:,k] + ω[:,k]
        #x[:,k+1] = A @ x[:,k]
        #y[:,k] = H @ x[:,k]

    return x[:,:-1], y, A, H

if __name__ == "__main__":

    #nsim = 20
    #batch_sizes = [10, 20, 30]
    #torch.manual_seed(100)
    #np.random.seed(100)

    Ts = 0.1
    P_0 = 0.05
    T = 100
    steps = 1000
    q = 0.1
    r = 0.1
    n_series = 3
    N = int(T/Ts)

    #x, y, A, H = simple_mass_spring_damp(T, q, r, P_0)
    x, y, A, H, Q, R = series_mass_spring_damp(T, q, r, P_0, n_series=n_series)
    t = np.arange(0, T, Ts)
    model = test.Model(A, H, gpu=False)
    losses, preds, weights = test.sgd(model, y, N, steps, lr=10e-3, gpu=False)
    
    # APA^T - APH^T (HPH^T + R)^(-1) HPA^T + Q - P = 0
    P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
    L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    n, m = A.shape[0], H.shape[0]

    print('|-------------------------------------|')
    print('  Horizon:', T)
    print('  n, m: ', n, m)
    print("  Kalman Gain L: \n", weights[:,:,-1])
    print("  L dimensions:", weights.shape)
    print("  P: \n", P)
    print("  L: \n", L)
    print('  A: \n', A)
    print('|-------------------------------------|')

    # Validation Results
    #x, y, A, H = simple_mass_spring_damp(T, Q, R, P_0) 
    x, y, A, H, Q, R = series_mass_spring_damp(T, q, r, P_0, n_series=n_series)
    preds = model(y,N).cpu().detach().numpy()

    for i in range(m):

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

        for j in range(n):
            plt.figure()
            plt.plot(weights[j][i])
            plt.axhline(L[j], linestyle='--')
            plt.title('Kalman Gain')
            plt.xlabel('n [steps]')
            plt.legend(["Gain", "Optimal Gain"])
            plt.grid()
            plt.show()

            plt.figure()
            plt.semilogy((L[j] - weights[j][i])**2)
            plt.title('MSE Kalman Gain')
            plt.xlabel('n [steps]')
            plt.grid()
            plt.show()
