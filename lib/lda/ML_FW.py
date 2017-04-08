# -*- coding: utf-8 -*-

import time
import numpy as np
from ldamodel import LdaModel
from ldalearning import LdaLearning
from tmlib.datasets.base import convert_corpus_format, DataFormat


class MLFW(LdaLearning):
    """
    Implements ML-FW for LDA as described in "Inference in topic models I: sparsity and trade-off". 
    """

    def __init__(self, num_terms, num_topics=100, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None):
        """
        Arguments:
            num_terms: Number of unique terms in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            tau0: A (positive) learning parameter that downweights early iterations.
            kappa: Learning rate: exponential decay rate should be between
                   (0.5, 1.0] to guarantee asymptotic convergence.
            iter_infer: Number of iterations of FW algorithm 

        Note that if you pass the same set of all documents in the corpus every time and
        set kappa=0 this class can also be used to do batch FW.
        """
        super(MLFW, self).__init__(num_terms, num_topics, lda_model)
        self.num_terms = num_terms
        self.num_topics = num_topics
        self.tau0 = tau0
        self.kappa = kappa
        self.updatect = 1
        self.INF_MAX_ITER = iter_infer

        # Initialize beta (topics)
        if self.lda_model is None:
            self.lda_model = LdaModel(num_terms, num_topics)
        self.lda_model.normalize()
        self.logbeta = np.log(self.lda_model.model)

        # Generate values used for initilization of topic mixture of each document 
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
        self.sparse_m_step(wordids, wordcts, theta, index)
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
        # Declare theta (topic mixtures) of mini-batch and list of non-zero indexes
        batch_size = len(wordids)
        theta = np.zeros((batch_size, self.num_topics))
        index = [{} for d in range(batch_size)]
        # Do inference for each document
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
        # Locate cache memory
        beta = self.lda_model.model[:, ids]
        logbeta = self.logbeta[:, ids]
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
            alpha = 2. / (l + 3)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x
            beta_x = beta[index, :] - x
            x += alpha * (beta_x)
        return (theta, list(nonzero))

    def sparse_m_step(self, wordids, wordcts, theta, index):
        """
        Does m step: update global variables beta, exploiting sparseness of the 
        solutions returned by Frank-Wolfe algorithm from e step as well as 
        that of wordids and wordcts lists.
        """
        # Compute un-normalized intermediate beta:  
        # \hat{beta}_{kj} = sum(over d in C_t) d_j * theta_{dk}.  
        # For each document, the computation only take nonzero elements of 
        # theta_d into consideration.
        batch_size = len(wordids)
        beta = np.zeros((self.num_topics, self.num_terms))
        for d in range(batch_size):
            for i in index[d]:
                beta[i, wordids[d]] += theta[d, i] * wordcts[d]
        # Check nonzero columns in the intermediate beta matrix above. Documents 
        # in the minibatch possibly contains a relatively fewer number of terms 
        # in comparison with vocabulary size that make the intermediate beta 
        # matrix may have too many zero columns.
        ids = list()
        for j in range(self.num_terms):
            if (sum(beta[:, j]) != 0):
                ids.append(j)
        # Normalize the intermediate beta
        for k in range(self.num_topics):
            if sum(beta[k, ids]) == 0:
                beta[k, ids] = 0.
            else:
                beta[k, ids] /= sum(beta[k, ids])
        # Update beta    
        rhot = pow(self.tau0 + self.updatect, -self.kappa)
        self.rhot = rhot
        self.lda_model.model *= (1 - rhot)
        self.lda_model.model[:, ids] += beta[:, ids] * rhot
        self.logbeta = np.log(self.lda_model.model)
        self.updatect += 1

    def m_step(self, batch_size, wordids, wordcts, theta, index):
        """
        Does m step: update global variables beta without considering the sparseness.
        """
        # Compute the intermediate topics
        beta = np.zeros((self.num_topics, self.num_terms))
        for d in range(batch_size):
            beta[:, wordids[d]] += np.outer(theta[d, :], wordcts[d])
        # normalize unit lambda
        beta_norm = beta.sum(axis=1)
        beta /= beta_norm[:, np.newaxis]
        # Update _lambda base on ML 
        rhot = pow(self.tau0 + self.updatect, -self.kappa)
        self.rhot = rhot
        self.lda_model.model *= (1 - rhot)
        self.lda_model.model += beta * rhot
        self.updatect += 1

    def infer_docs(self, new_corpus):
        docs = convert_corpus_format(new_corpus, DataFormat.TERM_FREQUENCY)
        theta, index = self.e_step(docs.word_ids_tks, docs.cts_lens)
        return theta