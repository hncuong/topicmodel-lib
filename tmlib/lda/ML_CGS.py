# -*- coding: utf-8 -*-
"""

@author: doanphongtung
"""
from __future__ import division
import numpy as np
from scipy.special import psi
import time
import util_funcs


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])

class MLCGS:
    
    def __init__(self, W, K, alpha, tau0, kappa, B, S, beta=None):
        self._W = W
        self._K = K
        self._alpha = alpha
        self._tau0 = tau0
        self._kappa = kappa
        self._B = B # burn-in
        self._S = S # samples
        self._sweeps = B + S
        self.update_unit = 1. / S 
        self._update_t = 1
        
        # initialize the variational distribution q(beta|lambda)
        if beta != None:
            self._lambda = beta
        else:
            self._lambda = 1 * np.random.gamma(100., 1./100., (self._K, self._W))
        # normalize lambda
        _lambda_norm = self._lambda.sum(axis = 1)
        self._lambda /= _lambda_norm[:, np.newaxis]
        
    def static_online(self, wordtks, lengths):
        batch_size = len(lengths)
        # E step
        start = time.time()
        (Ndk_mean, z) = self.sample_z(batch_size, wordtks, lengths)
        end1 = time.time()
        # M step
        self.update_lambda(batch_size, wordtks, lengths, Ndk_mean)
        end2 = time.time()
        return(end1 - start, end2 - end1, Ndk_mean)
        
    def sample_z(self, batch_size, wordtks, lengths):
        batch_N = sum(lengths)
        uni_rvs = np.random.uniform(size = (batch_N) * (self._sweeps + 1))
        z = [{} for d in range(0, batch_size)]
        Ndk = np.zeros((batch_size, self._K), dtype = np.uint32)
        Nkw_mean = np.zeros((self._K, self._W), dtype = np.float64)
        Ndk_mean = np.zeros((batch_size, self._K), dtype = np.float64)
        util_funcs.sampling(Ndk, Nkw_mean, Ndk_mean, self._lambda, uni_rvs, 
                            z, wordtks, lengths, self._alpha, self.update_unit,
                            self._S, self._B)
        # normalize Ndk_mean
        Ndk_mean_norm = Ndk_mean.sum(axis = 1)
        Ndk_mean /= Ndk_mean_norm[:, np.newaxis]
        return(Ndk_mean, z)
                    
    def update_lambda(self, batch_size, wordtks, lengths, Ndk_mean):
        _lambda = np.zeros((self._K, self._W))
        # compute unit lambda
        for d in range(batch_size):
            for j in range(lengths[d]):
                _lambda[:, wordtks[d][j]] += Ndk_mean[d]
        # normalize _lambda   
        _lambda_norm = _lambda.sum(axis = 1)
        _lambda /= _lambda_norm[:, np.newaxis]
        # update _lambda base on ML
        rhot = pow(self._tau0 + self._update_t, -self._kappa)
        self._rhot = rhot
        self._lambda *= (1- rhot) 
        self._lambda += _lambda * rhot
        self._update_t += 1
