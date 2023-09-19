#!/usr/bin/env python
# coding: utf-8

import numpy as np
from dataclasses import dataclass

@dataclass
class system:
    x: np.array
    y: np.array
    A: np.array
    H: np.array
    Q: np.array
    R: np.array
    B: np.array
    K: np.array
    C: np.array

def series_mass_spring_damp(T, Ts, conf):
    '''
    m x''(t) + c x'(t) + k x(t) = 0
    mass (kg)
    conf.cdamp, damping coefficient (N/m^2)
    kspring, spring constant (Ns/m)
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

    cdamp = conf.cdamp
    kspring = conf.kspring
    mass = conf.mass
    force = conf.force 
    P_0 = conf.P_0
    r = conf.r
    q = conf.q
    n_series = conf.n_series
    
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

    A = Ts * ( np.block([[np.zeros((n_series,n_series)), np.eye(n_series)],
                         [np.linalg.inv(M) @ K, np.linalg.inv(M) @ C]]) )
    A += np.eye(A.shape[0])
    n = A.shape[0]
    m = n
    H = np.eye(n)
    B = Ts * ( 1/mass * np.append(np.zeros(n-1), 1) )
    Q = np.eye(n) * q
    R = np.eye(m) * r
    u = force

    x = np.zeros((n,N+1))
    y = np.zeros((m,N))
    m_0 = np.random.normal(0, P_0, size=n)
    ξ = np.random.multivariate_normal(np.zeros(n), Q, N+1).T 
    ω = np.random.multivariate_normal(np.zeros(m), R, N).T
    if conf.initial:
        x[:,0] += m_0
    if conf.forcing:
        x[:,0] += B * u
    for k in range(N):
        x[:,k+1] = A @ x[:,k] + ξ[:,k]
        if conf.c_forcing:
            x[:,k+1] += B * u
        y[:,k] = H @ x[:,k] + ω[:,k]
    sys = system(
        x = x[:,:-1], y = y, A = A, H = H,
        Q = Q, R = R, K = K, C = C, B = B,
    )
    return sys 


def series_mass_spring_damp_pendulum(T, Ts, conf):
    '''
    m x''(t) + c x'(t) + k x(t) = 0
    mass (kg)
    conf.cdamp, damping coefficient (N/m^2)
    kspring, spring constant (Ns/m)
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

    cdamp = conf.cdamp
    kspring = conf.kspring
    mass = conf.mass
    force = conf.force 
    P_0 = conf.P_0
    r = conf.r
    q = conf.q
    n_series = conf.n_series
    
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

    A = Ts * ( np.block([[np.zeros((n_series,n_series)), np.eye(n_series)],
                         [np.linalg.inv(M) @ K, np.linalg.inv(M) @ C]]) )
    A += np.eye(A.shape[0])
    n = A.shape[0]
    m = n
    H = np.eye(n)
    B = Ts * ( 1/mass * np.append(np.zeros(n-1), 1) )
    Q = np.eye(n) * q
    R = np.eye(m) * r
    u = force

    x = np.zeros((n,N+1))
    y = np.zeros((m,N))
    m_0 = np.random.normal(0, P_0, size=n)
    ξ = np.random.multivariate_normal(np.zeros(n), Q, N+1).T 
    ω = np.random.multivariate_normal(np.zeros(m), R, N).T
    if conf.initial:
        x[:,0] += m_0
    if conf.forcing:
        x[:,0] += B * u
    for k in range(N):
        x[:,k+1] = A @ x[:,k] + ξ[:,k]
        if conf.c_forcing:
            x[:,k+1] += B * u
        y[:,k] = H @ x[:,k] + ω[:,k]
    sys = system(
        x = x[:,:-1], y = y, A = A, H = H,
        Q = Q, R = R, K = K, C = C, B = B,
    )
    return sys 
