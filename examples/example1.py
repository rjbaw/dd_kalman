#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from kalman import test

@dataclass
class training_config:
    initial: np.array
    steps: int = 300
    gpu: bool = False
    lr: float = 1e-3
    opt_params: str = 'adam'

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
class system:
    x: torch.Tensor
    y: torch.Tensor
    A: torch.Tensor
    H: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor
    B: torch.Tensor
    K: torch.Tensor
    C: torch.Tensor

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
        x = torch.Tensor(x[:,:-1]),
        y = torch.Tensor(y),
        A = torch.Tensor(A),
        H = torch.Tensor(H),
        Q = torch.Tensor(Q),
        R = torch.Tensor(R),
        K = torch.Tensor(K),
        C = torch.Tensor(C),
        B = torch.Tensor(B),
    )
    return sys 

def print_info(sys=None, ret=None):
    if ret is not None:
        print('|====================================================================================================|')
        print('  Horizon:', ret.T)
        print('  Time Steps:', ret.Ts)
        print('  Mass-Spring-Damper num of series:', ret.n_series)
        print("  Optimal Kalman Gain L: \n", ret.weights[:,:,-1])
        print("      dimensions:", ret.weights.shape)
        print("  DARE Kalman Gain L")
        print("  L: \n", ret.L)
        print("  P: \n", ret.P)
    if sys is not None:
        print('|----------------------------------------------------------------------------------------------------|')
        n, m = sys.A.shape[0], sys.H.shape[0]
        print('  n, m: ', n, m)
        print('  K: \n', sys.K)
        print('  C: \n', sys.C)
        print('  B: \n', sys.B)
        print('  A: \n', sys.A)
        print('  Q: \n', sys.Q)
        print('  H: \n', sys.H)
        print('  R: \n', sys.R)
        print('|===================================================================================================|')

def plot(sys,ret):
    t = torch.arange(0, ret.T, ret.Ts)
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
        plt.savefig("y_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
        plt.show()

        plt.figure()
        plt.plot(losses)
        plt.title('Loss Function')
        plt.xlabel('n [steps]')
        plt.grid()
        plt.savefig("trainloss_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
        plt.show()

        for j in range(n):
            plt.figure()
            plt.plot(weights[j][i])
            plt.axhline(L[j][i], linestyle='--')
            plt.title('Kalman Gain')
            plt.xlabel('n [steps]')
            plt.legend(["Optimal Gain", "DARE Gain"])
            plt.grid()
            plt.savefig("gain_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
            plt.show()

            plt.figure()
            plt.semilogy((L[j][i] - weights[j][i])**2)
            plt.title('MSE Kalman Gain')
            plt.xlabel('n [steps]')
            plt.grid()
            plt.savefig("msegain_T-{}_Ts-{}_nseries-{}.png".format(ret.T, ret.Ts, ret.n_series))
            plt.show()
    
def main():
    #nsim = 20
    #batch_sizes = [10, 20, 30]
    #torch.manual_seed(100)
    #torch.random.seed(100)

    T = 100
    Ts = 0.1
    N = int(T/Ts)
    n_series = 2
    sys_conf = system_config(n_series=n_series)

    sys = series_mass_spring_damp(T, Ts, sys_conf)
    A, H, R, Q = sys.A, sys.H, sys.R, sys.Q
    
    # APA^T - APH^T (HPH^T + R)^(-1) HPA^T + Q - P = 0
    P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
    P = torch.Tensor(P)
    L = A @ P @ H.T @ torch.linalg.inv(H @ P @ H.T + R)

    train_conf = training_config(initial = L)
    model = test.Model(A, H, train_conf)
    losses, preds, weights = test.sgd(model, sys.y, N, train_conf)
    sys = series_mass_spring_damp(T, Ts, sys_conf) # Validation
    preds = model(sys.y,N).cpu().detach().numpy()

    ret = result(
        T = T,
        Ts = Ts,
        n_series = n_series,
        L = L,
        P = P,
        losses = losses,
        preds = preds,
        weights = weights,
    )
    print_info(sys,ret)
    plot(sys,ret)

def main_range():
    T = 100
    Ts = 0.1
    N = int(T/Ts)
    n_series_range = 3
    plt.figure()
    for n_series in range(n_series_range):
        sys_conf = system_config(n_series=n_series+1)
        sys = series_mass_spring_damp(T, Ts, sys_conf)
        A, H, R, Q = sys.A, sys.H, sys.R, sys.Q
        P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
        P = torch.Tensor(P)
        L = A @ P @ H.T @ torch.linalg.inv(H @ P @ H.T + R)
        train_conf = training_config(initial = L)
        model = test.Model(A, H, train_conf)
        losses, preds, weights = test.sgd(model, sys.y, N, train_conf)
        sys = series_mass_spring_damp(T, Ts, sys_conf) # Validation
        preds = model(sys.y,N).cpu().detach().numpy()
        mse = (np.expand_dims(L,axis=-1) - weights)**2
        plt.semilogy( np.sum(mse, axis=(0,1)), label='Num Series {}'.format(n_series+1) )

        #ret = result(
        #    T = T,
        #    Ts = Ts,
        #    n_series = n_series,
        #    L = L,
        #    losses = losses,
        #    preds = preds,
        #    weights = weights,
        #)
        #print_info(sys,ret)
        #plot(sys,ret)

    plt.title('MSE Kalman Gain')
    plt.xlabel('n [steps]')
    plt.legend()
    plt.grid()
    plt.savefig('mse_nseries.png')
    plt.show()

    
if __name__ == "__main__":
    main_range()
    #main()

