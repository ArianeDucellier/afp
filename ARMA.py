"""
Scripts for ARMA models for time series analysis
"""

import numpy as np

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

def LD_recursions(gamma, N):
    """
    Use Levinson-Durbin recursions to compute the coefficients
    of the best linear predictor for the time series
    """
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
    """
    gamma = ACVF(X, h)
    (Phi, V) = LD_recursions(gamma, h)
    pacf = np.diag(Phi)
    return pacf
