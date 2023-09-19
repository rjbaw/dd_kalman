#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from tqdm import tqdm

class dspo_model(torch.nn.Module):
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

    def __init__(self, A, H, conf):
        super().__init__()
        self.device = conf.device
        self.n = A.shape[0]
        self.m = H.shape[0]
        self.A = torch.Tensor(A).to(self.device)
        self.H = torch.Tensor(H).to(self.device)
        #weights = torch.distributions.Uniform(0,0.5).sample((self.n,self.m))
        init = torch.Tensor(conf.initial).to(self.device)
        #self.weights = torch.nn.Parameter(init + weights)
        self.weights = torch.nn.Parameter(init)

    def forward(self, y, N):
        A = self.A
        H = self.H
        L = self.weights
        xhat = torch.zeros((self.n,N+1), device=self.device)
        yhat = torch.zeros((self.m,N), device=self.device)
        for k in range(N):
            if torch.isnan(xhat[:,k][0]):
                print("Infeasible solution, retry with new initial conditions")
                print(A @ xhat[:,k-2])
                print(L @ (y[:,k-2] - H @ xhat[:,k-2]))
                print(xhat[:,k-1])
                breakpoint()
            xhat[:,k+1] = A @ xhat[:,k] + L @ (y[:,k] - H @ xhat[:,k]) 
            yhat[:,k] = H @ xhat[:,k]
        return yhat

def sgd(model, y, N, conf):
    '''
    y, observation vector
    N, time steps
    steps, optimising num of loops
    lr, learning rate
    '''
    losses = np.zeros(conf.steps)
    gains = np.zeros((model.H.shape[1],model.H.shape[0],conf.steps))

    if conf.opt_params == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.opt_params == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum)
    else:
        assert True, "Not supported optimizer"

    model = model.to(conf.device)
    pbar = tqdm(total=conf.steps, unit='epochs')
    for i in range(conf.steps):
        preds = model(y,N)
        loss = torch.functional.F.mse_loss(preds, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses[i] = loss
        gains[:,:,i] = model.weights.cpu().detach()
        pbar.update(1)
    pbar.close()
    preds = model(y,N).cpu().detach().numpy()
    return losses, preds, gains
