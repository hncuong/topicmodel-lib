# -*- coding: utf-8 -*-

import time
import numpy as np
from ldamodel import LdaModel
from ldalearning import LdaLearning
from tmlib.datasets.utilizies import DataFormat, convert_corpus_format


class StreamingOPE(LdaLearning):
    """
    Implements Streaming-OPE for LDA as described in "Inference in topic models II: provably guaranteed algorithms". 
    """

    def __init__(self, data=None, num_topics=100, alpha=0.01, eta=0.01, iter_infer=50, lda_model=None):
        """
        Arguments:
            num_terms: Number of unique terms in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            alpha: Hyperparameter for prior on topic mixture theta.
            eta: Hyperparameter for prior on topics beta.
            iter_infer: Number of iterations of FW algorithm.
        """
        super(StreamingOPE, self).__init__(data, num_topics, lda_model)

        self.num_topics = num_topics
        self.alpha = alpha
        self.eta = eta
        self.INF_MAX_ITER = iter_infer

        if self.data is not None or self.lda_model is not None:
            if self.data is not None:
                self.num_terms = data.get_num_terms()

            if self.lda_model is not None:
                self.num_topics, self.num_terms = self.lda_model.model.shape
            else:
                # Initialize lambda (variational parameters of topics beta)
                # beta_norm stores values, each of which is sum of elements in each row
                # of _lambda.
                self.lda_model = LdaModel(self.num_terms, num_topics)
            self.beta_norm = self.lda_model.model.sum(axis=1)



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
        beta /= self.beta_norm[:, np.newaxis]
        # Initialize theta randomly
        theta = np.random.rand(self.num_topics) + 1.
        theta /= sum(theta)
        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.dot(theta, beta)
        # loop
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
        Does m step
        """
        # Compute sufficient statistics
        batch_size = len(wordids)
        sstats = np.zeros((self.num_topics, self.num_terms), dtype=float)
        for d in range(batch_size):
            theta_d = theta[d, :]
            phi_d = self.lda_model.model[:, wordids[d]] * theta_d[:, np.newaxis]
            phi_d_norm = phi_d.sum(axis=0)
            sstats[:, wordids[d]] += (wordcts[d] / phi_d_norm) * phi_d
        # Update
        self.lda_model.model += sstats + self.eta
        self.beta_norm = self.lda_model.model.sum(axis=1)

    def infer_new_docs(self, new_corpus):
        docs = convert_corpus_format(new_corpus, DataFormat.TERM_FREQUENCY)
        theta = self.e_step(docs.word_ids_tks, docs.cts_lens)
        return theta

    def estimate_topic_proportions(self, param_theta):
        return param_theta
