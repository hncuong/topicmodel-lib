import sys
import numpy as np


class LdaModel(object):
	"""docstring for ClassName"""
	def __init__(self, num_topics, num_terms, random_type=0, normarlize=False):
		# super(ClassName, self).__init__()
		self.num_topics = num_topics
		self.num_terms = num_terms
		self.random_type = random_type
		self.normarlize = normarlize
		if self.random_type ==0:
			self.model = np.random.rand(self.num_topics, self.num_terms) + 1e-10
		else:
			self.model = 1*np.random.gamma(100., 1./100., (self.num_topics, self.num_terms))
			self._Elogbeta = dirichlet_expectation(self.model)
        	self._expElogbeta = n.exp(self._Elogbeta)
		# if self.normarlize:

	def print_top_words(self, num_words, topic_list=None):



