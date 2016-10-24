# -*- coding: utf-8 -*-
"""

@author: doanphongtung
"""
from __future__ import division
import numpy as np
from scipy.special import psi
import time
from .utils import util_funcs
from ldamodel import LdaModel
from ldalearning import LdaLearning


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(np.sum(alpha)))
    return (psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


class MLCGS(LdaLearning):
    def __init__(self, num_terms, num_topics=100, alpha=0.01, tau0=1.0, kappa=0.9, burn_in=25, samples=25,
                 lda_model=None):
        """

        Args:
            num_terms:
            num_topics:
            alpha:
            tau0:
            kappa:
            burn_in:
            samples:
            lda_model:
        """
        super(MLCGS, self).__init__(num_terms, num_topics, lda_model)
        self.num_terms = num_terms
        self.num_topics = num_topics
        self._alpha = alpha
        self._tau0 = tau0
        self._kappa = kappa
        self.burn_in = burn_in  # burn-in
        self.samples = samples  # samples
        self._sweeps = burn_in + samples
        self.update_unit = 1. / samples
        self._update_t = 1

        # initialize the variational distribution q(beta|lambda)
        if self.lda_model is None:
            self.lda_model = LdaModel(num_terms, num_topics)
        self.lda_model.normalize()

    def static_online(self, wordtks, lengths):
        # E step
        start = time.time()
        (Ndk_mean, z) = self.sample_z(wordtks, lengths)
        end1 = time.time()
        # M step
        self.update_lambda(wordtks, lengths, Ndk_mean)
        end2 = time.time()
        return (end1 - start, end2 - end1, Ndk_mean)

    def sample_z(self, wordtks, lengths):
        batch_size = len(lengths)
        batch_N = sum(lengths)
        uni_rvs = np.random.uniform(size=(batch_N) * (self._sweeps + 1))
        z = [{} for d in range(0, batch_size)]
        Ndk = np.zeros((batch_size, self.num_topics), dtype=np.uint32)
        Nkw_mean = np.zeros((self.num_topics, self.num_terms), dtype=np.float64)
        Ndk_mean = np.zeros((batch_size, self.num_topics), dtype=np.float64)
        util_funcs.sampling(Ndk, Nkw_mean, Ndk_mean, self.lda_model.model, uni_rvs,
                            z, wordtks, lengths, self._alpha, self.update_unit,
                            self.samples, self.burn_in)
        # normalize Ndk_mean
        Ndk_mean_norm = Ndk_mean.sum(axis=1)
        Ndk_mean /= Ndk_mean_norm[:, np.newaxis]
        return (Ndk_mean, z)

    def update_lambda(self, wordtks, lengths, Ndk_mean):
        batch_size = len(lengths)
        _lambda = np.zeros((self.num_topics, self.num_terms))
        # compute unit lambda
        for d in range(batch_size):
            for j in range(lengths[d]):
                _lambda[:, wordtks[d][j]] += Ndk_mean[d]
        # normalize _lambda   
        _lambda_norm = _lambda.sum(axis=1)
        _lambda /= _lambda_norm[:, np.newaxis]
        # update _lambda base on ML
        rhot = pow(self._tau0 + self._update_t, -self._kappa)
        self._rhot = rhot
        self.lda_model.model *= (1 - rhot)
        self.lda_model.model += _lambda * rhot
        self._update_t += 1

    def learn_model(self, formatted_data, batch_size=5000, shuffle=False, passes=1, save_model_every=0,
                    compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20,
                    vocab_file='', model_folder='model'):
        super(MLCGS, self).\
            learn_model(formatted_data, format_type='sq', batch_size=batch_size, shuffle=shuffle, passes=passes,
                        save_model_every=save_model_every, compute_sparsity_every=compute_sparsity_every,
                        save_statistic=save_statistic, save_top_words_every=save_top_words_every,
                        num_top_words=num_top_words, vocab_file=vocab_file, model_folder=model_folder)