import sys, os
import math
import numpy as np
from scipy.special import psi

n.random.seed(100000001)

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

class LdaModel(object):
    """docstring for ClassName"""
    def __init__(self, num_topics, num_terms, random_type=0, normarlize=False):
	# super(ClassName, self).__init__()
	self.num_topics = num_topics
	self.num_terms = num_terms
	self.random_type = random_type
	self.normarlize = normarlize
	if self.random_type ==0:
	    self.model_param = np.random.rand(self.num_topics, self.num_terms) + 1e-10
	else:
	    self.model_param = 1*np.random.gamma(100., 1./100., (self.num_topics, self.num_terms))
	    self._Elogbeta = dirichlet_expectation(self.beta)
            self._expElogbeta = n.exp(self._Elogbeta)
	if self.normarlize:
	    beta_norm = self.beta.sum(axis = 1)
	    self.beta /= beta_norm[:, np.newaxis]
	    self.logbeta = np.log(self.beta)
	    

    def print_top_words(self, num_words, vocab_file, result_file=None):
    	# get the vocabulary
	if not os.path.isfile(vocab_file):
	    print('Unknown file %s' %vocab_file)
	    exit()
    	vocab = open(vocab_file, 'r').readlines()
    	vocab = map(lambda x: x.strip(), vocab) 
	if result_file is not None:   
    	    # open file to write    
    	    fp = open(result_file, 'w')
    	    topic_no = 0
    	    for topic in self.beta:
                fp.write('topic %03d\n' % (topic_no))
		index_sorted = np.argsort(topic)[::-1]	#sort in decending order
                for i in range(num_words):
                    index = index_sorted[i]
            	    fp.write ('   %s \t\t %f\n' % (vocab[index], topic[index]))
                topic_no = topic_no + 1
                fp.write( '\n')
    	    fp.close()
	else:
	    # display to screen
	    topic_no = 0
	    for topic in self.beta:
		print('topic %03d\n' %(topic_no))
		index_sorted = np.argsort(topic)[::-1]
		for i in range(num_words):
		    index = index_sorted[i]
		    print('   %s \t\t %f\n' %(vocab[index], topic[index]))
		topic_no += 1
		print('\n')

    def load(self, beta_file, 



