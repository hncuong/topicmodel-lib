import time
import numpy as n
from scipy.special import gammaln, psi
from ldamodel import LdaModel
from ldalearning import LdaLearning
from tmlib.datasets.utilizies import DataFormat, convert_corpus_format

n.random.seed(100000001)


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(n.sum(alpha)))
    return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])


class OnlineVB(LdaLearning):
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, data, num_topics=100, alpha=0.01, eta=0.01,
                 tau0=1.0, kappa=0.9,
                 conv_infer=0.0001, iter_infer=50, lda_model=None):
        super(OnlineVB, self).__init__(data, num_topics, lda_model)
        num_terms = data.get_num_terms()
        self.num_docs = 0
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0
        self._kappa = kappa
        self._updatect = 1
        self._conv_infer = conv_infer
        self._iter_infer = iter_infer

        # Initialize the variational distribution q(beta|lambda)
        if self.lda_model is None:
            self.lda_model = LdaModel(num_terms, num_topics, 1)
        self._Elogbeta = dirichlet_expectation(self.lda_model.model)
        self._expElogbeta = n.exp(self._Elogbeta)

    def static_online(self, wordids, wordcts):
        batch_size = len(wordids)
        # E step
        start = time.time()
        (gamma, sstats) = self.e_step(wordids, wordcts)
        end1 = time.time()
        # M step
        self.update_lambda(batch_size, sstats)
        end2 = time.time()
        return (end1 - start, end2 - end1, gamma)

    def e_step(self, wordids, wordcts):
        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        batch_size = len(wordids)
        gamma = 1 * n.random.gamma(100., 1. / 100., (batch_size, self.num_topics))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)
        sstats = n.zeros(self.lda_model.model.shape)
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
        return gamma, sstats

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
        temp += n.sum((self._eta - self.lda_model.model) * self._Elogbeta)
        temp += n.sum(gammaln(self.lda_model.model) - gammaln(self._eta))
        temp += n.sum(gammaln(self._eta * self.num_terms) -
                      gammaln(n.sum(self.lda_model.model, 1)))
        score += temp
        return (score)

    def update_lambda(self, batch_size, sstats):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Update lambda based on documents.
        self.lda_model.model = self.lda_model.model * (1 - rhot) + \
                                rhot * (self._eta + self.num_docs * sstats / batch_size)
        self._Elogbeta = dirichlet_expectation(self.lda_model.model)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

    def learn_model(self):
        self.num_docs += self.data.get_total_docs()
        return super(OnlineVB, self).learn_model()

    def infer_new_docs(self, new_corpus):
        docs = convert_corpus_format(new_corpus, DataFormat.TERM_FREQUENCY)
        gamma, sstats = self.e_step(docs.word_ids_tks, docs.cts_lens)
        gamma_norm = gamma.sum(axis=1)
        theta = gamma / gamma_norm[:, n.newaxis]
        return theta
