#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from dataclasses import dataclass

## Training config

@dataclass
class dspo_config:
    initial: np.array
    device: torch.device 
    steps: int = 1000
    gpu: bool = False
    lr: float = 5e-3
    opt_params: str = 'sgd'
    momentum: float = 0.9

## Environment config

@dataclass
class audio_sys_config:
    input: str = 'data/audio/mixture.flac'

@dataclass
class msd_sys_config:
    T: float = 100
    Ts: float = 0.1
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
    initial: bool = True

@dataclass
class msdp_sys_config:
    T: float = 100
    Ts: float = 5e-3
    cdamp: float = 4
    kspring: float = 2
    smass: float = 5
    bmass: float = 20
    force: float = 10 
    q: float = 0.1
    r: float = 0.1
    P_0: float = 0.05
    n_series: int = 1
    forcing: bool = True
    c_forcing: bool = False
    initial: bool = False
    length: float = 2

@dataclass
class pendulum_sys_config:
    T: float = 100
    Ts: float = 0.01
    mass: float = 20
    force: float = 10 
    length: float = 1
    mu: float = 0
    q: float = 0.1
    r: float = 0.1
    P_0: float = 0.05
    forcing: bool = True
    c_forcing: bool = False
    initial: bool = True

## Results config

@dataclass
class dspo_result:
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
