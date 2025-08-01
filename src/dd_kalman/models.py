#!/usr/bin/env python
# coding: utf-8

import numpy as np
from dataclasses import dataclass
from scipy.integrate import RK45

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

def rk4(x,y,T,A,B,H,u,ω,ξ,Ts,forcing):
    def model(t,x_0):
        x_next = A @ x_0
        if forcing:
            x_next += np.squeeze(B * u)
        return x_next
        
    N = int(T/Ts)
    for k in range(N):
        #sol = RK45(model,0,x_0,T,1,vectorized=True)
        #k1 = Ts*( f(x_0) )
        #k2 = Ts*( f(x_0 + 1/2 * k1) )
        #k3 = Ts*( f(x_0 + 1/2 * k2) )
        #k4 = Ts*( f(x_0 + 1/2 * k3) )
        #x[:,k+1] = x_0 + (k1+2*k2+2*k3+k4)/6
        x_0 = x[:,k]
        sol = RK45(model,k*Ts,x_0,T)
        sol.step()
        x[:,k+1] = sol.y + Ts*ξ[:,k]
        y[:,k] = H @ x[:,k] + ω[:,k]

    return x,y

def audio(conf):

    import soundfile as sf

    audio_in = conf.input
    y, sr = sf.read(audio_in, dtype='float32')
    y = y.T
    print('Audio Length:', y.shape[1])
    print('Audio Channels:', y.shape[0])
    print('Sampling Rate[Hz]:', sr)
    print('Audio Length[s]:', y.shape[1]/sr)
    
    sys = system(
        x = sr, y = y, A = 0, H = 1,
        Q = 0, R = 0, K = 0, C = 0, B = 0,
    )
    return sys 

def pendulum(conf):
    '''
    # m x''(t) + c x'(t) + k x(t) = 0
    mass (kg)
    cdamp, damping coefficient (N/m^2)
    kspring, spring constant (Ns/m)
    force (N)
    Ts, time step (s)
    N, simulation length (simulation time x time steps)

    x(t+1) = A x(t) + ξ(t)
    y(t) = H x(t) + ω(t)
    {ξ}_t∈Z, uncorrelated zero-mean process
    {ω}_t∈Z, measurement noise vectors
    x(t) ∈ R^n, state of the system
    y(t) ∈ R^m
    '''

    g = 9.81
    T = conf.T
    Ts = conf.Ts
    mass = conf.mass
    force = conf.force 
    P_0 = conf.P_0
    r = conf.r
    q = conf.q
    length = conf.length
    mass = conf.mass
    μ = conf.mu

    N = int(T/Ts)
    A = np.array([[0, 1], [-g/length, -μ/mass]])
    n = A.shape[0]
    m = n
    H = np.eye(n)
    B = 1/(mass*length) * np.array([[1.0, 0.0]]).T 
    Q = np.eye(n) * q
    R = np.eye(m) * r
    u = force

    x = np.zeros((n,N+1))
    y = np.zeros((m,N))
    m_0 = 0
    x_0 = np.random.normal(m_0, P_0, size=n)
    ξ = np.random.multivariate_normal(np.zeros(n), Q, N+1).T 
    ω = np.random.multivariate_normal(np.zeros(m), R, N).T
    if conf.initial:
        x[:,0] += x_0
    if conf.forcing:
        x[:,0] += np.squeeze(B * u)

    x, y = rk4(x,y,T,A,B,H,u,ω,ξ,Ts,conf.c_forcing)

    sys = system(
        x = x[:,:-1], y = y, A = A, H = H,
        Q = Q, R = R, K = 0, C = 0, B = B,
    )

    return sys 

def series_mass_spring_damp(conf):
    '''
    m x''(t) + c x'(t) + k x(t) = 0
    mass (kg)
    cdamp, damping coefficient (N/m^2)
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

    T = conf.T
    Ts = conf.Ts
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

    A = np.block([[np.zeros((n_series,n_series)), np.eye(n_series)],
                         [np.linalg.inv(M) @ K, np.linalg.inv(M) @ C]])

    n = A.shape[0]
    m = n
    H = np.eye(n)
    B = 1/mass * np.append(np.zeros(n-1), 1)
    Q = np.eye(n) * q
    R = np.eye(m) * r
    u = force

    x = np.zeros((n,N+1))
    y = np.zeros((m,N))
    m_0 = 0
    x_0 = np.random.normal(m_0, P_0, size=n)
    ξ = np.random.multivariate_normal(np.zeros(n), Q, N+1).T 
    ω = np.random.multivariate_normal(np.zeros(m), R, N).T
    if conf.initial:
        x[:,0] += x_0
    if conf.forcing:
        x[:,0] += B * u

    x, y = rk4(x,y,T,A,B,H,u,ω,ξ,Ts,conf.c_forcing)

    sys = system(
        x = x[:,:-1], y = y, A = A, H = H,
        Q = Q, R = R, K = K, C = C, B = B,
    )
    return sys 

def series_mass_spring_damp_pendulum(conf):
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

    g = 9.81
    T = conf.T
    Ts = conf.Ts
    cdamp = conf.cdamp
    kspring = conf.kspring
    smass = conf.smass
    bmass = conf.bmass
    force = conf.force 
    P_0 = conf.P_0
    r = conf.r
    q = conf.q
    n_series = conf.n_series
    length = conf.length
    
    N = int(T/Ts)
    K = np.zeros((n_series,n_series))
    C = np.zeros((n_series,n_series))

    K[0,0] = -kspring
    C[0,0] = -cdamp

    mg_block = smass * g * np.eye(n_series)
    zero_block = np.zeros((n_series,n_series))
    M = np.block(
        [
            [np.eye(2*n_series), np.zeros((2*n_series,2*n_series))],
            [zero_block, zero_block, bmass * np.eye(n_series), zero_block],
            [zero_block, zero_block, smass * np.eye(n_series), smass * length * np.eye(n_series)],
        ]
        
    )

    for idx in range(n_series-1):
        K[idx:idx+2,idx:idx+2] += np.array([[-kspring, kspring],
                                            [kspring,-kspring]])
        C[idx:idx+2,idx:idx+2] += np.array([[-cdamp, cdamp],
                                            [cdamp,-cdamp]])

    A = np.block(
        [
            [np.zeros((2*n_series,2*n_series)), np.eye(2*n_series)],
            [K, mg_block, C, zero_block],
            [zero_block, -mg_block, zero_block, zero_block],
        ]
    )
    A = np.linalg.inv(M) @ A

    n = A.shape[0]
    m = n
    H = np.eye(n)

    B = np.zeros(n)
    B[n - n_series] += Ts * 1/bmass * 1

    Q = np.eye(n) * q
    R = np.eye(m) * r
    u = force

    x = np.zeros((n,N+1))
    y = np.zeros((m,N))
    m_0 = 0
    x_0 = np.random.normal(m_0, P_0, size=n)
    ξ = np.random.multivariate_normal(np.zeros(n), Q, N+1).T 
    ω = np.random.multivariate_normal(np.zeros(m), R, N).T
    if conf.initial:
        x[:,0] += x_0
    if conf.forcing:
        x[:,0] += B * u

    x, y = rk4(x,y,T,A,B,H,u,ω,ξ,Ts,conf.c_forcing)

    sys = system(
        x = x[:,:-1], y = y, A = A, H = H,
        Q = Q, R = R, K = K, C = C, B = B,
    )
    return sys 
