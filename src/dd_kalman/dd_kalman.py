#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm


def log_regret(Tinit, β, λ, y, prev=None):
    """
    R_N = ∑_{k=1}^{N} ||y_k - ̃y_k||^2 - ∑_{k=1}^{N} ||y_k - ̃y_k||^2 
    ̂y_{k+1} = arg min E 
    ̂y, kalman filter prediction
    ̃y, online learning algorithm

    G_P = [ C(A - KC)^{p-1}K ... CK ]
    """
    def Zstack(y, k, p):
        #Z = y[:,(k-p):(k)].T
        Z = np.reshape(y[:,(k-p):(k)].T, (p*y.shape[0],1))
        return Z
    time_start = 0
    if prev is not None:
        time_start = prev.shape[1]
        y = np.concatenate([prev, y], axis=1)

    m, time_length = y.shape
    ytilde = np.zeros((m,time_length))
    losses = np.zeros(time_length)
    T = Tinit
    i = 1
    pbar = tqdm(total=time_length, unit='epochs')
    while True:
        T = 2**(i-1) * Tinit
        p = max(1, int(np.ceil(β * np.log(T))))
        if T <= p or (T+p) > y.shape[1]:
            break

        V_prev = λ * np.eye(m*p)
        partial_sum = np.zeros((m,m*p))
        for t in range(p,T):
            Z = Zstack(y,t,p)
            V_prev += Z @ Z.T
            partial_sum += y[:,t:t+1] @ Z.T
        V_inv = np.linalg.inv(V_prev)
        G_prev = partial_sum @ V_inv

        for k in range(T, min(2*T, y.shape[1])):
            Z = Zstack(y,k,p)
            ytilde[:,k] = np.squeeze(G_prev @ Z)
            losses[k] = np.sum((y[:,k] - ytilde[:,k])**2, axis=0)
            error = y[:,k] - ytilde[:,k]

            V_inv_Z = V_inv @ Z
            V_inv = V_inv - (V_inv_Z @ V_inv_Z.T) / float(1 + Z.T @ V_inv_Z)

            G = G_prev + np.expand_dims(error, axis=1) @ (Z.T @ V_inv)
            
            G_prev = G
            pbar.update(1)
        i += 1
    pbar.close()

    return ytilde[:,time_start:], losses[time_start:]
