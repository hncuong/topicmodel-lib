import time
import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
meanchangethresh = 1e-5
changethreshold = 1e-5


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(n.sum(alpha)))
    return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])


class StreamingVB:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, num_terms, num_topics, alpha, eta, conv_infer, iter_infer, beta=None):
        self.num_topics = num_topics
        self.num_terms = num_terms
        self._alpha = alpha
        self._eta = eta
        self._conv_infer = conv_infer
        self._iter_infer = iter_infer

        # Initialize the variational distribution q(beta|lambda)
        if beta != None:
            self._lambda = beta
        else:
            self._lambda = 1 * n.random.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def static_online(self, wordids, wordcts):
        batch_size = len(wordids)
        # E step
        start = time.time()
        (gamma, sstats) = self.do_e_step(batch_size, wordids, wordcts)
        end1 = time.time()
        # M step
        self.update_lambda(batch_size, sstats)
        end2 = time.time()
        return (end1 - start, end2 - end1, gamma)

    def do_e_step(self, batch_size, wordids, wordcts):
        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1 * n.random.gamma(100., 1. / 100., (batch_size, self.num_topics))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)
        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        for d in range(0, batch_size):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            bound = 1e-10  ##
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-10
            # Iterate between gamma and phi until convergence
            for it in range(0, self._iter_infer):
                # save current bound
                lastbound = bound  ##
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                                       n.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-10

                # recompute bound
                bound = self.approx_bound(ids, cts, gammad, phinorm)  ##
                # If bound hasn't changed much, we're done.
                change = n.abs((lastbound - bound) / lastbound)  ##
                if change < self._conv_infer:  ##
                    break  ##

            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts / phinorm)
        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta
        return ((gamma, sstats))

    def approx_bound(self, ids, wordcts, gamma, phinorm):
        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        # E[log p(docs | theta, beta)]
        cts = n.array(wordcts)
        score += n.sum(cts * phinorm)
        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma) * Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += gammaln(self._alpha * self.num_topics) - gammaln(sum(gamma))
        # E[log p(beta | eta) - log q (beta | lambda)]
        temp = 0
        temp += n.sum((self._eta - self._lambda) * self._Elogbeta)
        temp += n.sum(gammaln(self._lambda) - gammaln(self._eta))
        temp += n.sum(gammaln(self._eta * self.num_terms) -
                      gammaln(n.sum(self._lambda, 1)))
        score += temp
        return (score)

    def update_lambda(self, batch_size, sstats):
        self._lambda = self._lambda + sstats
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
