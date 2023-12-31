"""
Scripts for ARMA models for time series analysis
"""

import numpy as np
import torch

from math import sqrt

def ACVF(X, h):
    """
    Compute the autocovariance function of a univariate time series
    Input:
      X = 1D numpy array
      h = Integer, time lag up to which we compute the ACVF
    Output:
      acvf = 1D numpy array
    """
    assert isinstance(h, int), \
        'The lag of the autocovariance should be an integer'
    assert h >= 1, \
        'The lag should be higher or equal to 1'

    acvf = np.zeros(h + 1)
    N = np.shape(X)[0]
    Xbar = np.mean(X)

    for i in range(0, h + 1):       
        Xt = X[0 : (N - i)]
        Xth = X[i : N]
        acvf[i] = np.sum((Xth - Xbar) * (Xt - Xbar)) / N

    return acvf

def ACF(X, h):
    """
    Compute the autocorrelation function of a univariate time series
    Input:
      X = 1D numpy array
      h = Integer, time lag up to which we compute the ACF
    Output:
      acf = 1D numpy array
    """
    assert isinstance(h, int), \
        'The lag of the autocovariance should be an integer'
    assert h >= 1, \
        'The lag should be higher or equal to 1'

    acvf = ACVF(X, h)
    gamma0 = acvf[0]
    acf = acvf / gamma0

    return acf

def LD_recursions(gamma):
    """
    Use Levinson-Durbin recursions to compute the coefficients
    of the best linear predictor for the time series
    Input:
      gamma = 1D numpy array, autocovariance function of the time series
    Output:
      Phi = 2D numpy array, Phi[i - 1, j - 1] = phi_ij, i=1,..,N j=1,...,i
            Pn Xn+1 = phi_n1 Xn + ... + Phi_nn X1
      V = 1D numpy array, mean squared error of best linear predictor
            Vn = E(Xn+1 - PnXn+1)^2 
    """
    assert len(gamma) >= 3, \
       'The autocovariance must have lag h >= 2'
    N = len(gamma) - 1
    Phi = np.zeros((N, N))
    Phi[0, 0] = gamma[1] / gamma[0]
    V = np.zeros(N + 1)
    V[0] = gamma[0]
    V[1] = V[0] * (1 - Phi[0, 0]**2)
    for n in range(2, N + 1):
        Phi[n - 1, n - 1] = (gamma[n] - np.sum(Phi[n - 2, 0:(n - 1)] * \
            np.flip(gamma[1:n]))) / V[n - 1]
        Phi[n - 1, 0:(n - 1)] = Phi[n - 2, 0:(n - 1)] - Phi[n - 1, n - 1] * \
            np.flip(Phi[n - 2, 0:(n - 1)])
        V[n] = V[n - 1] * (1 - (Phi[n - 1, n - 1] ** 2))
    return (Phi, V)

def PACF(X, h):
    """
    Compute the partial autocorrelation function
    Input:
      X = 1D numpy array
      h = Integer, time lag up to which we compute the ACVF
    Output:
      pacf = 1D numpy array
    """
    assert isinstance(h, int), \
        'The lag of the partial autocorrelation should be an integer'
    assert h >= 1, \
        'The lag should be higher or equal to 1'

    gamma = ACVF(X, h)
    (Phi, V) = LD_recursions(gamma)
    pacf = np.diag(Phi)
    return pacf

def innovations(gamma):
    """
    Use the innovation algorithm to compute the coefficients
    of the best linear predictor for the time series
    Input:
      gamma = 1D numpy array, autocovariance function of the time series
    Output:
      Theta = 2D numpy array, Theta[i - 1, j - 1] = theta_ij, i=1,..,N j=1,...,i
            Pn Xn+1 = theta_n1 (Xn - Pn-1 Xn) + ... + theta_nn (X1 - P0 X1)
      V = 1D numpy array, mean squared error of best linear predictor
            Vn = E(Xn+1 - PnXn+1)^2 
    """
    assert len(gamma) >= 3, \
       'The autocovariance must have lag h >= 2'
    N = len(gamma) - 1
    Theta = np.zeros((N + 1, N + 1))
    Theta[1, 0] = gamma[1] / gamma[0]
    V = np.zeros(N + 1)
    V[0] = gamma[0]
    V[1] = gamma[0] - (Theta[1, 0] ** 2) * V[0]
    for n in range(2, N + 1):
        for k in range(0, n):
            Theta[n, k] = (gamma[n - k] -
                np.sum(Theta[k, 0:k] * Theta[n, 0:k] * V[0:k])) / V[k]
        V[n] = gamma[0] - np.sum(np.square(Theta[n, 0:n]) * V[0:n]) 
    return (Theta, V)

def ARP_yule_walker(X, p):
    """
    Estimate the parameters phi_p and sigma2 of an AR(p) process
    using Yule-Walker estimation
    Input:
      X = 1D numpy array
      p = Order of th AR(p) process
    Output:
      phi = 1D numpy array
      sigma = Standard deviation of the white noise
    """
    assert isinstance(p, int), \
        'The order of the AR(p) process should be an integer'
    assert p >= 1, \
        'The order of the AR(p) process should be higher or equal to 1'

    acvf = ACVF(X, p)
    Gamma = np.zeros((p, p))
    gamma = np.zeros(p)
    for i in range(0, p):
        gamma[i] = acvf[i + 1]
        for j in range(0, p):
            Gamma[i, j] = acvf[abs(i - j)]
    phi = np.linalg.solve(Gamma, gamma)
    sigma = sqrt(acvf[0] - np.sum(phi * gamma))
    return (phi, sigma)

def MA1_grid_search(X, thetas):
    """
    Estimate the best value of theta by minimizing the sum of the 
    squared errors of MA(1) process.
    """
    N = len(X)
    errors = np.zeros(len(thetas))
    for count, theta in enumerate(thetas):
        for i in range(2, N + 1):
            errors[count] = errors[count] + (X[i - 1] + \
                np.sum(np.power(-theta, np.arange(1, i)) * np.flip(X[0:(i - 1)]))) ** 2.0
    index = np.argmin(errors)
    return (thetas[index], errors)

def step_least_squares(X, theta, alpha):
    """
    Compute one step of the gradient descent
    for the least square method
    """
    LS = 0
    N = X.size()[0]
    for i in range(2, N + 1):
        LS = LS + (X[i - 1] + \
            torch.sum(torch.pow(-theta, torch.arange(1, i)) * \
            torch.flip(X[0:(i - 1)], [0]))) ** 2.0
    LS.backward()
    dtheta = theta.grad
    theta = theta - alpha * dtheta
    theta.retain_grad()
    return (theta, LS)

def MA1_gradient(X, max_iter, alpha):
    """
    Estimate the best value of theta by minimizing the sum of the 
    squared errors of MA(1) process.
    """
    theta = torch.rand(1, requires_grad=True)
    X = torch.from_numpy(X)
    i_iter = 0
    thetas = [theta.detach().numpy()[0]]
    loss = []
    while (i_iter < max_iter):
        (theta, LS) = step_least_squares(X, theta, alpha)
        thetas.append(theta.detach().numpy()[0])
        loss.append(LS.detach().numpy())
        i_iter = i_iter + 1
    return (thetas, loss)
    