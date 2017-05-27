# -*- coding: utf-8 -*-

import time
import numpy as np
from ldamodel import LdaModel
from ldalearning import LdaLearning
from tmlib.datasets.utilizies import DataFormat, convert_corpus_format


class StreamingFW(LdaLearning):
    """
    Implements Streaming-FW for LDA as described in "Inference in topic models I: sparsity and trade-off". 
    """

    def __init__(self, data, num_topics=100, eta=0.01, iter_infer=50, lda_model=None):
        """
        Arguments:
            num_docs: Number of documents in the corpus.
            num_terms: Number of unique terms in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            eta: Hyperparameter for prior on topics beta.
            iter_infer: Number of iterations of FW algorithm.
        """
        super(StreamingFW, self).__init__(data, num_topics, lda_model)
        num_terms = data.get_num_terms()
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.eta = eta
        self.INF_MAX_ITER = iter_infer

        # Initialize lambda (variational parameters of topics beta)
        # beta_norm stores values, each of which is sum of elements in each row
        # of _lambda.
        if self.lda_model is None:
            self.lda_model = LdaModel(num_terms, num_topics)
        self.beta_norm = self.lda_model.model.sum(axis=1)

        # Generate values used for initilaization of topic mixture of each document
        self.theta_init = [1e-10] * num_topics
        self.theta_vert = 1. - 1e-10 * (num_topics - 1)

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
        (theta, index) = self.e_step(wordids, wordcts)
        end1 = time.time()
        # M step
        start2 = time.time()
        self.m_step(wordids, wordcts, theta, index)
        end2 = time.time()
        return (end1 - start1, end2 - start2, theta)

    def e_step(self, wordids, wordcts):
        """
        Does e step 
		
        Returns topic mixtures and their nonzero elements' indexes of all documents in the mini-batch.
        
        Note that, FW can provides sparse solution (theta:topic mixture) when doing inference
        for each documents. It means that the theta have few non-zero elements whose indexes
        are stored in list of lists 'index'.		
        """
        # Declare theta (topic mixtures) of minibatch and list of non-zero indexes
        batch_size = len(wordids)
        theta = np.zeros((batch_size, self.num_topics))
        index = [{} for d in range(batch_size)]
        # Inference
        for d in range(batch_size):
            (thetad, indexd) = self.infer_doc(wordids[d], wordcts[d])
            theta[d, :] = thetad
            index[d] = indexd
        return (theta, index)

    def infer_doc(self, ids, cts):
        """
        Does inference for a document using Frank Wolfe algorithm.
        
        Arguments:
        ids: an element of wordids, corresponding to a document.
        cts: an element of wordcts, corresponding to a document.

        Returns inferred theta and list of indexes of non-zero elements of the theta.
        """
        # locate cache memory
        beta = self.lda_model.model[:, ids]
        beta /= self.beta_norm[:, np.newaxis]
        logbeta = np.log(beta)
        nonzero = set()
        # Initialize theta to be a vertex of unit simplex 
        # with the largest value of the objective function
        theta = np.array(self.theta_init)
        f = np.dot(logbeta, cts)
        index = np.argmax(f);
        nonzero.add(index)
        theta[index] = self.theta_vert
        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.copy(beta[index, :])
        # Loop
        for l in range(0, self.INF_MAX_ITER):
            # Select a vertex with the largest value of  
            # derivative of the objective function
            df = np.dot(beta, cts / x)
            index = np.argmax(df);
            nonzero.add(index)
            beta_x = beta[index, :] - x
            alpha = 2. / (l + 3)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x
            x += alpha * (beta_x)
        return (theta, np.array(list(nonzero)))

    def m_step(self, wordids, wordcts, theta, index):
        """
        Does m step
        """
        # Compute sufficient sstatistics
        batch_size = len(wordids)
        sstats = np.zeros((self.num_topics, self.num_terms))
        for d in range(batch_size):
            phi_d = self.lda_model.model[index[d], :]
            phi_d = phi_d[:, wordids[d]]
            theta_d = theta[d, index[d]]
            phi_d *= theta_d[:, np.newaxis]
            phi_norm = phi_d.sum(axis=0)
            phi_d *= (wordcts[d] / phi_norm)
            for i in range(len(index[d])):
                sstats[index[d][i], wordids[d]] += phi_d[i, :]
        # Update
        self.lda_model.model += sstats + self.eta
        self.beta_norm = self.lda_model.model.sum(axis=1)

    def infer_new_docs(self, new_corpus):
        docs = convert_corpus_format(new_corpus, DataFormat.TERM_FREQUENCY)
        theta, index = self.e_step(docs.word_ids_tks, docs.cts_lens)
        return theta