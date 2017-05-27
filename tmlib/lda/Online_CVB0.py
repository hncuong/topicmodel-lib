# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:47:53 2014

@author: doanphongtung
"""

import time
import numpy as np
from ldamodel import LdaModel
from ldalearning import LdaLearning
from tmlib.datasets.utilizies import DataFormat, convert_corpus_format


class OnlineCVB0(LdaLearning):
    def __init__(self, data, num_topics=100, alpha=0.01, eta=0.01, tau_phi=1.0,
                 kappa_phi=0.9, s_phi=1.0, tau_theta=10.0,
                 kappa_theta=0.9, s_theta=1.0, burn_in=25, lda_model=None):
        """

        Args:
            num_tokens:
            num_terms:
            num_topics:
            alpha:
            eta:
            tau_phi:
            kappa_phi:
            s_phi:
            tau_theta:
            kappa_theta:
            s_theta:
            burn_in:
            lda_model:
        """
        super(OnlineCVB0, self).__init__(data, num_topics, lda_model)
        num_tokens = data.get_num_tokens()
        num_terms = data.get_num_terms()
        self.num_tokens = num_tokens
        self.num_terms = num_terms
        self.num_topics = num_topics
        self.alpha = alpha
        self.eta = eta
        self.eta_sum = num_topics * eta
        self.tau_phi = tau_phi
        self.kappa_phi = kappa_phi
        self.s_phi = s_phi
        self.tau_theta = tau_theta
        self.kappa_theta = kappa_theta
        self.s_theta = s_theta
        self.burn_in = burn_in
        self.updatect = 1

        # self.N_phi = np.random.rand(num_topics, num_terms)
        # replace N_phi with lda model
        if self.lda_model is None:
            self.lda_model = LdaModel(num_terms, num_topics)
        self.N_Z = self.lda_model.model.sum(axis=1)

    def static_online(self, wordtks, lengths):
        # E step
        start1 = time.time()
        (N_phi, N_Z, N_theta) = self.e_step(wordtks, lengths)
        end1 = time.time()
        # M step
        start2 = time.time()
        self.m_step(N_phi, N_Z)
        end2 = time.time()
        return end1 - start1, end2 - start2, N_theta

    def e_step(self, wordtks, lengths):
        batch_size = len(lengths)
        N_phi = np.zeros((self.num_topics, self.num_terms), dtype=float)
        N_Z = np.zeros(self.num_topics)
        N_theta = np.random.rand(batch_size, self.num_topics)
        # inference
        denominator = self.N_Z + self.eta_sum
        multiplier = self.num_tokens / sum(lengths)
        # for each document j im M
        for j in range(batch_size):
            # for zero or more "burn in" passes
            for b in range(self.burn_in):
                # for each token i
                for i in range(lengths[j]):
                    # update gamma_ij
                    gamma_ij = self.lda_model.model[:, wordtks[j][i]] + self.eta
                    numerator = N_theta[j] + self.alpha
                    gamma_ij = gamma_ij * numerator / denominator
                    gamma_ij = gamma_ij / sum(gamma_ij)
                    # update N_theta
                    rhot = self.s_theta * pow(self.tau_theta + i + 1, -self.kappa_theta)
                    N_theta[j] = (1 - rhot) * N_theta[j] + rhot * lengths[j] * gamma_ij
            # for each token i
            for i in range(lengths[j]):
                # update gamma_ij
                gamma_ij = self.lda_model.model[:, wordtks[j][i]] + self.eta
                numerator = N_theta[j] + self.alpha
                gamma_ij = gamma_ij * numerator / denominator
                gamma_ij = gamma_ij / sum(gamma_ij)
                # update N_theta
                rhot = self.s_theta * pow(self.tau_theta + i + 1, -self.kappa_theta)
                N_theta[j] = (1 - rhot) * N_theta[j] + rhot * lengths[j] * gamma_ij
                temp = multiplier * gamma_ij
                # N_w_ij(phi) := N_w_ij(phi) + C / |M| * gamma_ij
                N_phi[:, wordtks[j][i]] += temp
                # N_Z := N_Z + C / |M| * gamma_ij
                N_Z += temp
        return N_phi, N_Z, N_theta

    def m_step(self, N_phi, N_Z):
        rhot = self.s_phi * pow(self.tau_phi + self.updatect, -self.kappa_phi)
        self.rhot_phi = rhot
        self.lda_model.model *= (1 - rhot)
        self.lda_model.model += rhot * N_phi
        self.N_Z *= (1 - rhot)
        self.N_Z += rhot * N_Z
        self.updatect += 1

    def learn_model(self):
        self.data.set_output_format(DataFormat.TERM_SEQUENCE)
        return super(OnlineCVB0, self).learn_model()

    def infer_new_docs(self, new_corpus):
        docs = convert_corpus_format(new_corpus, DataFormat.TERM_SEQUENCE)
        N_phi, N_Z, theta = self.e_step(docs.word_ids_tks, docs.cts_lens)
        return theta