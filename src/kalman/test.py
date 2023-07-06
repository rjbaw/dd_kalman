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
    def __init__(self, y, A, H, P):
        super().__init__()
        #L = A @ P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        weights = torch.distributions.Uniform(0, 0.5).sample()
        self.A = A
        self.H = H
        self.y = y
        self.weights = torch.nn.Parameter(weights)
    def forward(self, N):
        A = torch.Tensor(self.A)
        H = torch.Tensor(self.H)
        y = torch.Tensor(self.y)
        L = self.weights

        #m_0 = 0
        #partial_sum = 0
        #for t in range(N - 1):
        #    partial_sum += H@(A - L*H)**(N-1-t) * L * y[t]
        #yhat = H@(A - H*L)**(N) * m_0 + partial_sum
        #print(yhat)

    #    B = np.array([0, Ts*(1/m)])
    #    for k in range(N-1):
    #        x[:,k+1] = A @ x[:,k] + B * F + ξ[k]
        xhat = torch.zeros((2,N))
        yhat = torch.zeros(N)
        for k in range(N-1):
            xhat[:,k+1] = A @ xhat[:,k] + L * (y[k] - H @ xhat[:,k]) 
            yhat[k+1] = H @ xhat[:,k]
        return yhat

def sgd(model, y, N, n=1000, lr=1e-3, opt_params='adam'):
    '''
    y, observation vector
    N, Horizon
    '''
    losses = []
    weights = []
    if opt_params == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_params == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        assert True, "Not supported optimizer"
    pbar = tqdm(total=n, unit='epochs')
    for i in range(n):
        preds = model(N)
        loss = torch.functional.F.mse_loss(preds, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
        weights.append(float(model.weights))
        pbar.update(1)
    print(model.weights)
    preds = model(N).detach().numpy()
    return losses, preds, weights
