#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import dd_kalman as ddk
import soundfile as sf 
from conf import *

def main():
    device = ddk.get_device(gpu=True)
    sys_conf = audio_sys_config(input='data/audio/organfinale.wav')
    #sys_conf = audio_sys_config(input='data/audio/compression.wav')
    #sys_conf = audio_sys_config(input='data/audio/bad_guy.flac')
    sys = ddk.audio(sys_conf)
    A, H, R, Q = sys.A, sys.H, sys.R, sys.Q

    #P = scipy.linalg.solve_discrete_are(A.T,H.T,Q,R)
    #L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    #P_init = scipy.linalg.solve_discrete_are(A.T,H.T,Q*2,R*2)
    #L_init = A @ P_init @ H.T @ np.linalg.inv(H @ P_init @ H.T + R)
    #train_conf = training_config(initial = L_init, device=device)
    #y = torch.Tensor(sys.y).to(device)
    #model = ddk.dspo_model(A, H, train_conf)
    #losses, preds, weights = ddk.sgd(model, y, N, train_conf)

    y = sys.y
    sr = sys.x
    T = y.shape[1]
    Tinit = 2
    β = 1 * 1/np.log(T)
    λ = 10
    regret_preds, regret_losses = ddk.log_regret(Tinit, β, λ, y)
    sf.write("data/audio/out.flac", regret_preds.T, sr)
    t = np.arange(0, T)
    n, m = 2, 2
    
    for i in range(m):
        plt.figure()
        plt.plot(t, y[i])
        plt.plot(t, regret_preds[i], color='g')
        plt.title('Audio file')
        plt.xlabel('t [s]')
        plt.ylabel('y(t)')
        plt.grid()
        plt.legend(["y", "regret y"])
        plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.suptitle('Audio file vs Prediction')
        axes[0].plot(t, y[i])
        axes[0].grid()
        axes[0].set_title('Audio file')
        axes[0].set_xlabel('t [s]')
        axes[0].set_ylabel('y(t)')
        axes[1].plot(t, regret_preds[i], color='g')
        axes[1].grid()
        axes[1].set_title('Prediction')
        axes[1].set_xlabel('t [s]')
        axes[1].set_ylabel('y(t)')
        plt.show()

        plt.figure()
        plt.plot(t, (y[i]-regret_preds[i])**2)
        plt.title('Audio file')
        plt.xlabel('t [s]')
        plt.ylabel('y(t)')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()
