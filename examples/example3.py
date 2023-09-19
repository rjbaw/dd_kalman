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
    forcing: bool = False
    c_forcing: bool = False
    initial: bool = True

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

@dataclass
class log_regret_result:
    T: float
    Ts: float
    n_series: int
    V: np.array
    G: np.array

def main_range():
    T = 100
    Ts = 0.1
    N = int(T/Ts)
    n_series_range = 30
    device = ddk.get_device(gpu=True)
    plt.figure()
    for n_series in range(n_series_range):
        sys_conf = system_config(n_series=n_series+1)
        sys = ddk.series_mass_spring_damp(T, Ts, sys_conf)
        A, H, R, Q = sys.A, sys.H, sys.R, sys.Q
        P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
        L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        P_init = scipy.linalg.solve_discrete_are(A.T,H.T,Q*20,R*20)
        L_init = A @ P_init @ H.T @ np.linalg.inv(H @ P_init @ H.T + R)
        train_conf = training_config(initial = L_init, device=device, lr=5e-3 * n_series**4)
        y = torch.Tensor(sys.y).to(device)
        model = ddk.dspo_model(A, H, train_conf)
        losses, preds, weights = ddk.sgd(model, y, N, train_conf)
        sys = ddk.series_mass_spring_damp(T, Ts, sys_conf) # Validation
        y = torch.Tensor(sys.y).to(device)
        preds = model(y,N).cpu().detach().numpy()
        mse = (np.expand_dims(L,axis=-1) - weights)**2
        plt.semilogy( np.sum(mse, axis=(0,1)),
                      label='Num Series {}'.format(n_series+1) )
        ret = result(
            T = T, Ts = Ts, n_series = n_series, L = L, P = P,
            losses = losses, preds = preds, weights = weights,
        )

        #Tinit = 10
        #β = 10 * 1/np.log(T)
        #λ = 1
        ##y = torch.Tensor(sys.y).to(device)
        #regret_preds, regret_losses, G, V = ddk.log_regret(Tinit, β, λ, sys.y)

        ddk.print_info(sys,ret)
        #plot(sys,ret)

        np.save("data/y-{}.npy".format(n_series+1), sys.y)
        np.save("data/losses-{}.npy".format(n_series+1), losses)
        np.save("data/preds-{}.npy".format(n_series+1), preds)
        np.save("data/weights-{}.npy".format(n_series+1), weights)
        np.save("data/L-{}.npy".format(n_series+1), L)
        np.save("data/P-{}.npy".format(n_series+1), P)

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
        y = np.load("data/y-{}.npy".format(n_series+1))
        losses = np.load("data/losses-{}.npy".format(n_series+1))
        preds = np.load("data/preds-{}.npy".format(n_series+1))
        weights = np.load("data/weights-{}.npy".format(n_series+1))
        L = np.load("data/L-{}.npy".format(n_series+1))
        P = np.load("data/P-{}.npy".format(n_series+1))
        mse[i] = np.sum((np.expand_dims(L,axis=-1) - weights)**2 , axis=(0,1))[-1]
    plt.figure()
    plt.title('Final MSE Kalman Gain')
    plt.xlabel('n series dimension')
    plt.grid()
    plt.plot(np.arange(n_series_range)+1, mse)
    plt.show()

def main_regret():
    T = 100
    Ts = 0.1
    N = int(T/Ts)
    n_series = 3
    device = ddk.get_device(gpu=True)
    sys_conf = system_config(n_series=n_series)
    sys = ddk.series_mass_spring_damp(T, Ts, sys_conf)
    A, H, R, Q = sys.A, sys.H, sys.R, sys.Q
    P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
    L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

    P_init = scipy.linalg.solve_discrete_are(A.T,H.T,Q*2,R*2)
    L_init = A @ P_init @ H.T @ np.linalg.inv(H @ P_init @ H.T + R)
    train_conf = training_config(initial = L_init, device=device)
    y = torch.Tensor(sys.y).to(device)
    model = ddk.dspo_model(A, H, train_conf)
    losses, preds, weights = ddk.sgd(model, y, N, train_conf)

    Tinit = 2
    β = 10 * 1/np.log(T)
    λ = 100
    #y = torch.Tensor(sys.y).to(device)
    #regret_preds, regret_losses, G, V = ddk.log_regret(Tinit, β, λ, sys.y)

    # Validation
    sys = ddk.series_mass_spring_damp(T, Ts, sys_conf)
    y = torch.Tensor(sys.y).to(device)
    preds = model(y,N).cpu().detach().numpy()
    regret_preds, regret_losses, G, V = ddk.log_regret(Tinit, β, λ, sys.y)
    #regret_preds, regret_losses, G, V = ddk.log_regret(Tinit, β, λ, sys.y, (V,G))

    t = np.arange(0, T, Ts)
    n, m = sys.A.shape[0], sys.H.shape[0]
    val_loss_dspo = (y-preds)**2
    val_loss_regret = (y-regret_preds)**2
    
    for i in range(m):
        plt.figure()
        plt.plot(t, y[i])
        plt.plot(t, preds[i], color='r')
        plt.plot(t, regret_preds[i], color='g')
        plt.title('Simulation of Mass-Spring-Damper System')
        plt.xlabel('t [s]')
        plt.ylabel('y(t)')
        plt.grid()
        plt.legend(["y", "filtered y", "regret y"])
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

    ret = result(
        T = T, Ts = Ts, n_series = n_series, L = L, P = P,
        losses = losses, preds = preds, weights = weights,
    )

    #print_info(sys,ret)
    #plot(sys,ret)
    
if __name__ == "__main__":
    #main_range()
    #n_series_range=10
    #plot_data(n_series_range)
    main_regret()
