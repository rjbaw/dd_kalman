#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import numba
import torch
from tqdm import tqdm

@numba.jit(nopython=True, parallel=True)
def gaussians(x, means, widths):
    '''Return the value of gaussian kernels.
    
    x - location of evaluation
    means - array of kernel means
    widths - array of kernel widths
    '''
    n = means.shape[0]
    result = np.exp( -0.5 * ((x - means) / widths)**2 ) / widths
    return result / np.sqrt(2 * np.pi) / n

@numba.jit(nopython=True, parallel=True)
def monte_carlo_pi_parallel(nsamples):
    acc = 0
    for _ in numba.prange(nsamples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

class Model(torch.nn.Module):
    '''
    Input (y, A, H, P)
    y, observation vector
    P, Covariance
    
    x(t+1) = A x(t) + ξ(t)
    y(t) = H x(t) + ω(t)
    x ∈ R^n, state
    y ∈ R^m, observation
    {ξ}_t∈Z, uncorrelated zero-mean process
    {ω}_t∈Z, measurement noise vectors
    E[ξ(t)ξ(t)^T] = Q
    E[ω(t)ω(t)^T] = R
    Q & R, unknown
    P, error covariance matrix
    ̂x(0) = m_0, P(t_0) = P_0

    Prediction problem
    ̂y_P(T) = H ̂x_P(T)
    ̂x(t+1) = A ̂x(t) + L(t) ( y(t) - H x̂(t) )

    Kalman gain
    L(t) := A P(t) H^T ( H P(t) H^T + R )^(-1)
    P(t) = E[(x(t) - ̂x(t))(x(t) - ̂x(t))^T]
    P(t+1) = ( A - L(t) H ) P(t) A^T + Q
    P converges when (A,H) is observable and the (A, Q^(1/2)) is controllable
    L_∞ = A P_∞ H^T ( H P_∞ H^T + R )^(-1)
    '''
    def __init__(self, A, H):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        weights = torch.distributions.Uniform(0, 0.5).sample()
        self.A = torch.Tensor(A).to(self.device)
        self.H = torch.Tensor(H).to(self.device)
        self.weights = torch.nn.Parameter(weights)

    def forward(self, y, N):
        A = self.A
        H = self.H
        L = self.weights
        y = torch.Tensor(y).to(self.device)
        n = A.shape[0]
        m = y.shape[0]
        xhat = torch.zeros((n,N+1), device=self.device)
        yhat = torch.zeros((m,N), device=self.device)
        for k in range(N):
            xhat[:,k+1] = A @ xhat[:,k] + L * (y[:,k] - H @ xhat[:,k]) 
            yhat[:,k] = H @ xhat[:,k]
        return yhat

def sgd(model, y, N, steps=1000, lr=1e-3, opt_params='adam'):
    '''
    y, observation vector
    N, time steps
    steps, optimising num of loops
    lr, learning rate
    '''
    losses = []
    weights = []

    if opt_params == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_params == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        assert True, "Not supported optimizer"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.Tensor(y).to(device)
    model = model.to(device)

    pbar = tqdm(total=steps, unit='epochs')
    for i in range(steps):
        preds = model(y,N)
        loss = torch.functional.F.mse_loss(preds, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
        weights.append(float(model.weights))
        pbar.update(1)
    pbar.close()
    preds = model(y,N).cpu().detach().numpy()
    return losses, preds, weights
