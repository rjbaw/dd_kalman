#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
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
    #sys_conf = pendulum_sys_config(c_forcing=False, initial=True, force=1)
    #sys = ddk.pendulum(sys_conf) # Validation
    return sys_conf, sys

def main():
    n_series = 3
    device = ddk.get_device(gpu=True)
    sys_conf, sys = env_init(n_series)
    T = sys_conf.T
    Ts = sys_conf.Ts
    N = int(T/Ts)
    A, H, R, Q = sys.A, sys.H, sys.R, sys.Q
    P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
    L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

    P_init = scipy.linalg.solve_discrete_are(A.T,H.T,Q*2,R*2)
    L_init = A @ P_init @ H.T @ np.linalg.inv(H @ P_init @ H.T + R)
    train_conf = dspo_config(initial = L_init, device=device)
    y = torch.Tensor(sys.y).to(device)
    model = ddk.dspo_model(A, H, train_conf)
    losses, preds, weights = ddk.sgd(model, y, N, train_conf)

    Tinit = 2
    β = 2 * 1/np.log(T)
    λ = 1
    #y = torch.Tensor(sys.y).to(device)
    #regret_preds, regret_losses = ddk.log_regret(Tinit, β, λ, sys.y)

    # Validation
    y_prev = sys.y
    sys_conf, sys = env_init(n_series)
    y = torch.Tensor(sys.y).to(device)
    preds = model(y,N).cpu().detach().numpy()
    #regret_preds, regret_losses = ddk.log_regret(Tinit, β, λ, sys.y)
    regret_preds, regret_losses = ddk.log_regret(Tinit, β, λ, sys.y, y_prev)

    t = np.arange(0, T, Ts)
    n, m = sys.A.shape[0], sys.H.shape[0]
    val_loss_dspo = (sys.y-preds)**2
    val_loss_regret = (sys.y-regret_preds)**2
    
    for i in range(m):
        plt.figure()
        plt.plot(t, sys.y[i])
        plt.plot(t, preds[i], color='r')
        plt.plot(t, regret_preds[i], color='g')
        plt.title('Simulation of Mass-Spring-Damper System')
        plt.xlabel('t [s]')
        plt.ylabel('y(t)')
        plt.grid()
        plt.legend(["y", "dspo y", "log-regret y"])
        #plt.savefig("img/unsorted/y_T-{}_Ts-{}_nseries-{}.png".format(T, Ts, n_series))
        plt.show()

        plt.figure()
        plt.scatter(range(train_conf.steps), val_loss_dspo[i])
        plt.scatter(range(train_conf.steps), val_loss_regret[i])
        plt.title('Loss Function')
        plt.xlabel('n [steps]')
        plt.grid()
        plt.legend(["dspo losses", "log regret losses"])
        #plt.savefig("img/unsorted/trainloss_T-{}_Ts-{}_nseries-{}.png".format(T, Ts, n_series))
        plt.show()

        plt.figure()
        plt.scatter(val_loss_dspo[i], val_loss_regret[i])
        plt.title('dspo vs regret loss')
        plt.ylabel('dspo')
        plt.xlabel('regret')
        plt.grid()
        #plt.legend(["dspo losses", "log regret losses"])
        #plt.savefig("img/unsorted/trainloss_T-{}_Ts-{}_nseries-{}.png".format(T, Ts, n_series))
        plt.show()

    ret = dspo_result(
        T = T, Ts = Ts, n_series = n_series, L = L, P = P,
        losses = losses, preds = preds, weights = weights,
    )
    
if __name__ == "__main__":
    main()
