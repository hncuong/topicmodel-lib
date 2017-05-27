# -*- coding: utf-8 -*-

import time
import numpy as np
from ldamodel import LdaModel
from ldalearning import LdaLearning
from tmlib.datasets.utilizies import DataFormat, convert_corpus_format


class MLOPE(LdaLearning):
    """
    Implements ML-OPE for LDA as described in "Inference in topic models II: provably guaranteed algorithms". 
    """

    def __init__(self, data, num_topics=100, alpha=0.01, tau0=1.0, kappa=0.9, iter_infer=50,
                 lda_model=None):
        """
        Arguments:
            num_terms: Number of unique terms in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            alpha: Hyperparameter for prior on topic mixture theta.
            tau0: A (positive) learning parameter that downweights early iterations.
            kappa: Learning rate: exponential decay rate should be between
                   (0.5, 1.0] to guarantee asymptotic convergence.
            iter_infer: Number of iterations of FW algorithm 

        Note that if you pass the same set of all documents in the corpus every time and
        set kappa=0 this class can also be used to do batch OPE.
        """
        super(MLOPE, self).__init__(data, num_topics, lda_model)
        self.num_topics = num_topics
        self.num_terms = data.get_num_terms()
        self.alpha = alpha
        self.tau0 = tau0
        self.kappa = kappa
        self.updatect = 1
        self.INF_MAX_ITER = iter_infer

        # Initialize beta (topics)
        if self.lda_model is None:
            self.lda_model = LdaModel(self.num_terms, num_topics)
        self.lda_model.normalize()

    def static_online(self, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        topics in M step.
		
        Arguments:
        batch_size: Number of documents of the mini-batch.
        wordids: A list whose each element is an array (terms), corresponding to a document.
                 Each element of the array is index of a unique term, which appears in the document,
                 in the vocabulary.
        wordcts: A list whose each element is an array (frequency), corresponding to a document.
                 Each element of the array says how many time the corresponding term in wordids appears
                 in the document.
        Returns time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch.        		
        """
        # E step
        start1 = time.time()
        theta = self.e_step(wordids, wordcts)
        end1 = time.time()
        # M step
        start2 = time.time()
        self.m_step(wordids, wordcts, theta)
        end2 = time.time()
        return (end1 - start1, end2 - start2, theta)

    def e_step(self, wordids, wordcts):
        """
        Does e step 
		
        Returns topic mixtures theta.
        """
        # Declare theta of minibatch
        batch_size = len(wordids)
        theta = np.zeros((batch_size, self.num_topics))
        # Inference
        for d in range(batch_size):
            thetad = self.infer_doc(wordids[d], wordcts[d])
            theta[d, :] = thetad
        return (theta)

    def infer_doc(self, ids, cts):
        """
        Does inference for a document using Online MAP Estimation algorithm.
        
        Arguments:
        ids: an element of wordids, corresponding to a document.
        cts: an element of wordcts, corresponding to a document.

        Returns inferred theta.
        """
        # locate cache memory
        beta = self.lda_model.model[:, ids]
        # Initialize theta randomly
        theta = np.random.rand(self.num_topics) + 1.
        theta /= sum(theta)
        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.dot(theta, beta)
        # Loop
        T = [1, 0]
        for l in range(1, self.INF_MAX_ITER):
            # Pick fi uniformly
            T[np.random.randint(2)] += 1
            # Select a vertex with the largest value of  
            # derivative of the function F
            df = T[0] * np.dot(beta, cts / x) + T[1] * (self.alpha - 1) / theta
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x
            x = x + alpha * (beta[index, :] - x)
        return (theta)

    def m_step(self, wordids, wordcts, theta):
        """
        Does m step: update global variables beta.
        """
        # Compute intermediate beta which is denoted as "unit beta"
        batch_size = len(wordids)
        beta = np.zeros((self.num_topics, self.num_terms), dtype=float)
        for d in range(batch_size):
            beta[:, wordids[d]] += np.outer(theta[d], wordcts[d])
        # Check zeros index
        beta_sum = beta.sum(axis=0)
        ids = np.where(beta_sum != 0)[0]
        unit_beta = beta[:, ids]
        # Normalize the intermediate beta
        unit_beta_norm = unit_beta.sum(axis=1)
        unit_beta /= unit_beta_norm[:, np.newaxis]
        # Update beta    
        rhot = pow(self.tau0 + self.updatect, -self.kappa)
        self.rhot = rhot
        self.lda_model.model *= (1 - rhot)
        self.lda_model.model[:, ids] += unit_beta * rhot
        self.updatect += 1

    def infer_docs(self, new_corpus):
        docs = convert_corpus_format(new_corpus, DataFormat.TERM_FREQUENCY)
        theta = self.e_step(docs.word_ids_tks, docs.cts_lens)
        return theta