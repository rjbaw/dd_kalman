#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dd_kalman as ddk
from conf import *

def env_init(n_series):
    sys_conf = msd_sys_config(n_series=n_series)
    sys = ddk.series_mass_spring_damp(sys_conf)
    #sys_conf = msdp_sys_config(n_series=n_series)
    #sys = ddk.series_mass_spring_damp_pendulum(sys_conf)
    #sys_conf = pendulum_sys_config(c_forcing=True, initial=False, r=0, q=0, force=1)
    #sys = ddk.pendulum(sys_conf) # Validation
    return sys_conf, sys

def main_range():
    n_series_range = 100
    device = ddk.get_device(gpu=True)
    plt.figure()
    for n_series in range(n_series_range):
        sys_conf, sys = env_init(n_series+1)
        T = sys_conf.T
        Ts = sys_conf.Ts
        N = int(T/Ts)
        A, H, R, Q = sys.A, sys.H, sys.R, sys.Q
        P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
        L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        P_init = scipy.linalg.solve_discrete_are(A.T,H.T,Q*1,R*1)
        L_init = A @ P_init @ H.T @ np.linalg.inv(H @ P_init @ H.T + R)
        train_conf = dspo_config(initial = L_init, device=device, lr=3e-3 * n_series/2)
        y = torch.Tensor(sys.y).to(device)
        model = ddk.dspo_model(A, H, train_conf)
        losses, preds, weights = ddk.sgd(model, y, N, train_conf)
        # Validation
        sys_conf, sys = env_init(n_series+1)
        y = torch.Tensor(sys.y).to(device)
        preds = model(y,N).cpu().detach().numpy()
        mse = (np.expand_dims(L,axis=-1) - weights)**2
        plt.semilogy( np.sum(mse, axis=(0,1)),
                      label='Num Series {}'.format(n_series+1) )
        ret = dspo_result(
            T = T, Ts = Ts, n_series = n_series, L = L, P = P,
            losses = losses, preds = preds, weights = weights,
        )

        ddk.print_info(sys,ret)
        #plot(sys,ret)

        np.save("data/numpy/y-{}.npy".format(n_series+1), sys.y)
        np.save("data/numpy/losses-{}.npy".format(n_series+1), losses)
        np.save("data/numpy/preds-{}.npy".format(n_series+1), preds)
        np.save("data/numpy/weights-{}.npy".format(n_series+1), weights)
        np.save("data/numpy/L-{}.npy".format(n_series+1), L)
        np.save("data/numpy/P-{}.npy".format(n_series+1), P)

    plt.title('MSE Kalman Gain')
    plt.xlabel('n [steps]')
    plt.legend()
    plt.grid()
    plt.savefig("img/unsorted/mse_nseries.png")
    plt.show()
    plot_data(n_series_range)

def plot_data(n_series_range):
    mse = np.zeros(n_series_range)
    for i,n_series in enumerate(range(n_series_range)):
        y = np.load("data/numpy/y-{}.npy".format(n_series+1))
        losses = np.load("data/numpy/losses-{}.npy".format(n_series+1))
        preds = np.load("data/numpy/preds-{}.npy".format(n_series+1))
        weights = np.load("data/numpy/weights-{}.npy".format(n_series+1))
        L = np.load("data/numpy/L-{}.npy".format(n_series+1))
        P = np.load("data/numpy/P-{}.npy".format(n_series+1))
        mse[i] = np.sum((np.expand_dims(L,axis=-1) - weights)**2 , axis=(0,1))[-1]
    plt.figure()
    plt.title('Final MSE Kalman Gain')
    plt.xlabel('n series dimension')
    plt.grid()
    plt.plot(np.arange(n_series_range)+1, mse)
    plt.show()
    
if __name__ == "__main__":
    main_range()
    #n_series_range=10
    #plot_data(n_series_range)
