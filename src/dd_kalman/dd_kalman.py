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
        p = round(β * np.log(T))
        if (T+p) > y.shape[1]:
            break

        V_prev = λ * np.eye(m*p)
        partial_sum = np.zeros((m,m*p))
        for t in range(T):
            Z = Zstack(y,t+p,p)
            V_prev += Z @ Z.T
            partial_sum += np.expand_dims(y[:,t+p], axis=0).T @ Z.T
        G_prev = partial_sum @ np.linalg.inv(V_prev)

        for j in range(2*T):
            k = T+j
            if k >= y.shape[1]:
                break
            Z = Zstack(y,k,p)
            ytilde[:,k] = np.squeeze(G_prev @ Z)
            losses[k] = np.sum((y[:,k] - ytilde[:,k])**2, axis=0)

            V = V_prev + Z @ Z.T
            #G = G_prev + np.expand_dims((y[:,k] - ytilde[:,k]),axis=0) @ Z.T @ np.linalg.inv( V )
            G = G_prev + np.expand_dims((y[:,k] - ytilde[:,k]),axis=1) @ Z.T @ np.linalg.inv( V )
            G_prev = G
            V_prev = V
            pbar.update(1)
        i += 1
    pbar.close()

    return ytilde[:,time_start:], losses
