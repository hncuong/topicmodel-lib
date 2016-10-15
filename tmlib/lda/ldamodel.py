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
    def __init__(self, num_topics, num_terms, random_type=0):
	# super(ClassName, self).__init__()
	self.num_topics = num_topics
	self.num_terms = num_terms
	self.random_type = random_type
	self.normarlize = normarlize
	if self.random_type ==0:
	    self.model = np.random.rand(self.num_topics, self.num_terms) + 1e-10
	else:
	    self.model = 1*np.random.gamma(100., 1./100., (self.num_topics, self.num_terms))
	    self._Elogbeta = dirichlet_expectation(self.beta)
            self._expElogbeta = n.exp(self._Elogbeta)
	    

    """ normarlize if nessesary (model parameter which is loaded or init is lambda"""
    def normarlize(self):
	beta_norm = self.model.sum(axis = 1)
	beta = self.model / beta_norm[:, np.newaxis]
	#self.logbeta = np.log(self.beta)
	return beta
	    
    """
	display top words of topics:
	*** num_words: number of words which is displayed in each topic
	*** vocab_file: vocabulary file of copus
	*** <optional> show_topics: number of topics which we want to show; otherwise, it'll show all topics
	*** <optional> result_file: write topics with words and probability respectly into this file; 
				    otherwise, print into screen 
    """
    def print_top_words(self, num_words, vocab_file, show_topics=None, result_file=None):
    	# get the vocabulary
	if not os.path.isfile(vocab_file):
	    print('Unknown file %s' %vocab_file)
	    exit()
    	vocab = open(vocab_file, 'r').readlines()
    	vocab = map(lambda x: x.strip(), vocab)
	if show_topics is not None:
	    index_list = np.random.random_randint(self.num_topics, size=show_topics)
	    topic_list = self.beta[index_list,:]
	else:
	    topic_list = self.beta
	if result_file is not None:   
    	    # open file to write    
    	    fp = open(result_file, 'w')
    	    topic_no = 0
    	    for topic in topic_list:
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
	    for topic in topic_list:
		print('topic %03d\n' %(topic_no))
		index_sorted = np.argsort(topic)[::-1]
		for i in range(num_words):
		    index = index_sorted[i]
		    print('   %s \t\t %f\n' %(vocab[index], topic[index]))
		topic_no += 1
		print('\n')

    """ load model (beta or lambda) from a file which is learnt to learn continue"""
    def load(self, beta_file)
        if isfile(beta_file):
            f = open(beta_file)
            lines = f.readlines()
            words = lines[0].strip().split()
            K = len(lines)
            W = len(words)
            beta = np.zeros((K,W))
            for i in xrange(K):
                words = lines[0].strip().split()
                if len(words) != W:
                    print('File %s is error' %beta_file)
                    exit()
                for j in xrange(W):
                    beta[i][j] = float(words[j])
	    self.model = beta
            return beta
        else:
            print('Unknown file %s' %beta_file)
            exit()

    " save model into a file"""
    def save(self, file_beta):
    	f = open(file_beta, 'w')
    	for k in range(self.num_topics):
            for i in range(self.num_terms - 1):
                f.write('%.10f ' % (self.model[k][i]))
            f.write('%.10f\n' % (self.model[k][num_terms - 1]))
        f.close()
