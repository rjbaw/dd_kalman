#!/usr/bin/env python
# coding: utf-8

import torch

def get_device(gpu):
    if gpu:
        if torch.cuda.is_available():
            gpu = 'cuda'
        elif torch.backends.mps.is_available():
            gpu = 'mps'
        else:
            gpu = 'cpu'
    else:
        gpu = 'cpu'
    device = torch.device(gpu)
    print('Using device:', device)
    return device


def print_info(sys=None, dspo=None, log_regret=None):
    if dspo is not None:
        print('|====================================================================================================|')
        print('  Horizon:', dspo.T)
        print('  Time Steps:', dspo.Ts)
        print('  Mass-Spring-Damper num of series:', dspo.n_series)
        print("  Optimal Kalman Gain L: \n", dspo.weights[:,:,-1])
        print("  DARE Kalman Gain L")
        print("  L: \n", dspo.L)
        print("  P: \n", dspo.P)
        print('|===================================================================================================|')
    if log_regret is not None:
        print('|====================================================================================================|')
        print('  Horizon:', log_regret.T)
        print('  Time Steps:', log_regret.Ts)
        print('  Mass-Spring-Damper num of series:', log_regret.n_series)
        print("  V: \n", log_regret.V)
        print("  G: \n", log_regret.G)
        print('|===================================================================================================|')
    if sys is not None:
        print('|===================================================================================================|')
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
