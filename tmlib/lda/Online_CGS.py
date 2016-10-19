# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy.special import psi
import time
from .utils import util_funcs


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(np.sum(alpha)))
    return (psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


class OnlineCGS:
    def __init__(self, num_docs, num_terms, num_topics, alpha, eta, tau0, kappa, burn_in, samples, beta=None):
        self.num_docs = num_docs
        self.num_terms = num_terms
        self.num_topics = num_topics
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0
        self._kappa = kappa
        self._update_t = 1
        self.burn_in = burn_in  # burn-in
        self.samples = samples  # samples
        self._sweeps = burn_in + samples
        self.update_unit = 1. / samples

        # initialize the variational distribution q(beta|lambda)
        if beta != None:
            self._lambda = beta
        else:
            self._lambda = 1 * np.random.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)

    def static_online(self, wordtks, lengths):
        batch_size = len(lengths)
        # E step
        start = time.time()
        (sstats, theta, z) = self.sample_z(batch_size, wordtks, lengths)
        end1 = time.time()
        # M step
        self.update_lambda(batch_size, sstats)
        end2 = time.time()
        return (end1 - start, end2 - end1, theta)

    def sample_z(self, batch_size, wordtks, lengths):
        batch_N = sum(lengths)
        uni_rvs = np.random.uniform(size=(batch_N) * (self._sweeps + 1))
        z = [{} for d in range(0, batch_size)]
        Ndk = np.zeros((batch_size, self.num_topics), dtype=np.uint32)
        Nkw_mean = np.zeros((self.num_topics, self.num_terms), dtype=np.float64)
        Ndk_mean = np.zeros((batch_size, self.num_topics), dtype=np.float64)
        util_funcs.sampling(Ndk, Nkw_mean, Ndk_mean, self._expElogbeta, uni_rvs,
                            z, wordtks, lengths, self._alpha, self.update_unit,
                            self.samples, self.burn_in)
        return (Nkw_mean, Ndk_mean, z)

    def update_lambda(self, batch_size, sstats):
        rhot = pow(self._tau0 + self._update_t, -self._kappa)
        self._rhot = rhot
        self._lambda = self._lambda * (1 - rhot) + \
                       rhot * (self._eta + (self.num_docs / batch_size) * sstats)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        self._update_t += 1
