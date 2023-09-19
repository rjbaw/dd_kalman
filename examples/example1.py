#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dd_kalman as ddk

@dataclass
class training_config:
    initial: np.array
    device: torch.device 
    steps: int = 1000
    gpu: bool = False
    lr: float = 5e-3
    opt_params: str = 'sgd'
    momentum: float = 0.9

@dataclass
class system_config:
    cdamp: float = 4
    kspring: float = 2
    mass: float = 20
    force: float = 10 
    q: float = 0.1
    r: float = 0.1
    P_0: float = 0.05
    n_series: int = 1
    forcing: bool = True
    c_forcing: bool = False
    initial: bool = False

@dataclass
class result:
    T: float
    Ts: float
    n_series: int
    L: np.array
    P: np.array
    losses: np.array
    preds: np.array
    weights: np.array

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

def main():
    #nsim = 20
    #batch_sizes = [10, 20, 30]
    #torch.manual_seed(100)
    #torch.random.seed(100)

    T = 100
    Ts = 0.1
    N = int(T/Ts)
    n_series = 10
    device = ddk.get_device(gpu=True)

    sys_conf = system_config(n_series=n_series)
    sys = ddk.series_mass_spring_damp(T, Ts, sys_conf)
    A, H, R, Q = sys.A, sys.H, sys.R, sys.Q
    # APA^T - APH^T (HPH^T + R)^(-1) HPA^T + Q - P = 0
    P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
    L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    P_init = scipy.linalg.solve_discrete_are(A.T,H.T,Q*2,R*2)
    L_init = A @ P_init @ H.T @ np.linalg.inv(H @ P_init @ H.T + R)

    train_conf = training_config(initial = L_init, device=device, lr=5e-3 * n_series**2)
    y = torch.Tensor(sys.y).to(device)
    model = ddk.dspo_model(A, H, train_conf)
    losses, preds, weights = ddk.sgd(model, y, N, train_conf)
    sys = ddk.series_mass_spring_damp(T, Ts, sys_conf) # Validation
    y = torch.Tensor(sys.y).to(device)
    preds = model(y,N).cpu().detach().numpy()

    ret = result(
        T = T, Ts = Ts, n_series = n_series, L = L, P = P,
        losses = losses, preds = preds, weights = weights,
    )
    ddk.print_info(sys,ret)
    plot(sys,ret)
    
if __name__ == "__main__":
    main()
