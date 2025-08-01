#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dd_kalman as ddk
from conf import *

def plot(sys,ret):
    t = np.arange(0, ret.T, ret.Ts)
    L, weights, preds, losses = ret.L, ret.weights, ret.preds, ret.losses
    n, m = sys.A.shape[0], sys.H.shape[0]
    for i in range(m):
        plt.figure()
        plt.plot(t, sys.y[i])
        plt.plot(t, preds[i], color='r')
        plt.title('Simulation of Mass-Spring-Damper System')
        plt.xlabel('t [s]')
        plt.ylabel('y(t)')
        plt.grid()
        plt.legend(["y", "filtered y"])
        plt.savefig("img/unsorted/y_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
        plt.show()

        plt.figure()
        plt.plot(losses)
        plt.title('Loss Function')
        plt.xlabel('n [steps]')
        plt.grid()
        plt.savefig("img/unsorted/trainloss_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
        plt.show()

        for j in range(n):
            plt.figure()
            plt.plot(weights[j][i])
            plt.axhline(L[j][i], linestyle='--')
            plt.title('Kalman Gain')
            plt.xlabel('n [steps]')
            plt.legend(["Optimal Gain", "DARE Gain"])
            plt.grid()
            plt.savefig("img/unsorted/gain_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
            plt.show()

            plt.figure()
            plt.semilogy((L[j][i] - weights[j][i])**2)
            plt.title('MSE Kalman Gain')
            plt.xlabel('n [steps]')
            plt.grid()
            plt.savefig("img/unsorted/msegain_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
            plt.show()

def env_init(n_series):
    #sys_conf = msd_sys_config(n_series=n_series)
    #sys = ddk.series_mass_spring_damp(sys_conf)
    #sys_conf = msdp_sys_config(n_series=n_series, r=0.001, q=0.001)
    #sys = ddk.series_mass_spring_damp_pendulum(sys_conf)
    sys_conf = pendulum_sys_config(forcing=True, c_forcing=True, initial=True,
                                   r=0.001, q=0.001, force=1)
    sys = ddk.pendulum(sys_conf) # Validation
    return sys_conf, sys
    

def main():
    test = False
    #nsim = 20
    n_series = 2
    device = ddk.get_device(gpu=True)
    sys_conf, sys = env_init(n_series)
    T = sys_conf.T
    Ts = sys_conf.Ts
    N = int(T/Ts)

    if test:
        t = np.arange(0, T, Ts)
        print(sys.A)
        print(sys.B)
        for i in range(sys.y.shape[0]):
            plt.figure()
            plt.plot(t, sys.y[i])
            plt.title('Simulation')
            plt.xlabel('t [s]')
            plt.ylabel('y(t)')
            plt.grid()
            plt.show()
        return

    A, H, R, Q = sys.A, sys.H, sys.R, sys.Q
    # APA^T - APH^T (HPH^T + R)^(-1) HPA^T + Q - P = 0
    P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
    L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

    P_init = scipy.linalg.solve_discrete_are(A.T,H.T,2*Q,2*R)
    L_init = A @ P_init @ H.T @ np.linalg.inv(H @ P_init @ H.T + R)

    train_conf = dspo_config(initial = L_init, device=device)
    y = torch.Tensor(sys.y).to(device)
    model = ddk.dspo_model(A, H, train_conf)
    losses, preds, weights = ddk.sgd(model, y, N, train_conf)

    # Validation
    sys_conf, sys = env_init(n_series)
    y = torch.Tensor(sys.y).to(device)
    preds = model(y,N).cpu().detach().numpy()

    ret = dspo_result(
        T = T, Ts = Ts, n_series = n_series, L = L, P = P,
        losses = losses, preds = preds, weights = weights,
    )
    ddk.print_info(sys,ret)
    plot(sys,ret)
    
if __name__ == "__main__":
    main()
