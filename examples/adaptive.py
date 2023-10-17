#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dd_kalman as ddk

#@dataclass
#class training_config:
#    initial: np.array
#    device: torch.device 
#    steps: int = 1000
#    gpu: bool = False
#    lr: float = 5e-3
#    opt_params: str = 'sgd'
#    momentum: float = 0.9

@dataclass
class msd_sys_config:
    cdamp: float = 10
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

#@dataclass
#class msdp_sys_config:
#    cdamp: float = 4
#    kspring: float = 2
#    smass: float = 5
#    bmass: float = 20
#    force: float = 10 
#    q: float = 0.1
#    r: float = 0.1
#    P_0: float = 0.05
#    n_series: int = 1
#    forcing: bool = True
#    c_forcing: bool = False
#    initial: bool = False
#    length: float = 2
#
#@dataclass
#class pendulum_sys_config:
#    mass: float = 20
#    force: float = 10 
#    length: float = 1
#    mu: float = 0
#    q: float = 0.1
#    r: float = 0.1
#    P_0: float = 0.05
#    forcing: bool = True
#    c_forcing: bool = False
#    initial: bool = False
#
#@dataclass
#class result:
#    T: float
#    Ts: float
#    n_series: int
#    L: np.array
#    P: np.array
#    losses: np.array
#    preds: np.array
#    weights: np.array

#def plot(sys,ret):
#    t = np.arange(0, ret.T, ret.Ts)
#    L, weights, preds, losses = ret.L, ret.weights, ret.preds, ret.losses
#    n, m = sys.A.shape[0], sys.H.shape[0]
#    for i in range(m):
#        plt.figure()
#        plt.plot(t, sys.y[i])
#        plt.plot(t, preds[i], color='r')
#        plt.title('Simulation of Mass-Spring-Damper System')
#        plt.xlabel('t [s]')
#        plt.ylabel('y(t)')
#        plt.grid()
#        plt.legend(["y", "filtered y"])
#        plt.savefig("img/unsorted/y_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
#        plt.show()
#
#        plt.figure()
#        plt.plot(losses)
#        plt.title('Loss Function')
#        plt.xlabel('n [steps]')
#        plt.grid()
#        plt.savefig("img/unsorted/trainloss_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
#        plt.show()
#
#        for j in range(n):
#            plt.figure()
#            plt.plot(weights[j][i])
#            plt.axhline(L[j][i], linestyle='--')
#            plt.title('Kalman Gain')
#            plt.xlabel('n [steps]')
#            plt.legend(["Optimal Gain", "DARE Gain"])
#            plt.grid()
#            plt.savefig("img/unsorted/gain_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
#            plt.show()
#
#            plt.figure()
#            plt.semilogy((L[j][i] - weights[j][i])**2)
#            plt.title('MSE Kalman Gain')
#            plt.xlabel('n [steps]')
#            plt.grid()
#            plt.savefig("img/unsorted/msegain_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
#            plt.show()

def estimate_R(H,L):
    S = 1
    R = (np.eye(2) - H@L) @ S
    return R

def estimate_L():
    pass

def estimate_QPP():
    pass

def estimate_LSPPQR():
    pass

def main():
    #nsim = 20
    #batch_sizes = [10, 20, 30]
    #torch.manual_seed(100)
    #torch.random.seed(100)

    T = 100
    Ts = 0.1
    N = int(T/Ts)
    n_series = 3
    #device = ddk.get_device(gpu=True)

        #sys_conf = msdp_sys_config(n_series=n_series, q=0, r=0)
        #sys = ddk.series_mass_spring_damp_pendulum(T, Ts, sys_conf)
        #sys_conf = pendulum_sys_config(c_forcing=True, initial=False, r=0, q=0, force=1)
        #sys = ddk.pendulum(T, Ts, sys_conf) # Validation

    sys_conf = msd_sys_config(n_series=n_series, q=0, r=0)
    sys = ddk.series_mass_spring_damp(T, Ts, sys_conf)

    A, H, R, Q = sys.A, sys.H, sys.R, sys.Q
    # APA^T - APH^T (HPH^T + R)^(-1) HPA^T + Q - P = 0
    P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
    L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

#    P_init = scipy.linalg.solve_discrete_are(A.T,H.T,2*Q,2*R)
#    L_init = A @ P_init @ H.T @ np.linalg.inv(H @ P_init @ H.T + R)
#
#    ret = result(
#        T = T, Ts = Ts, n_series = n_series, L = L, P = P,
#        losses = losses, preds = preds, weights = weights,
#    )
#    ddk.print_info(sys,ret)
#    plot(sys,ret)
    
if __name__ == "__main__":
    main()
