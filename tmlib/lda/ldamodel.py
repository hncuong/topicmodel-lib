import sys, os
import math
import numpy as np
import logging


class LdaModel(object):
    """docstring for ClassName"""

    def __init__(self, num_terms, num_topics, random_type=0):
        """

        Args:
            num_topics:
            num_terms:
            random_type:
        """
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.random_type = random_type
        if self.random_type == 0:
            self.model = np.random.rand(self.num_topics, self.num_terms) + 1e-10
        else:
            self.model = 1 * np.random.gamma(100., 1. / 100., (self.num_topics, self.num_terms))

    def normalize(self):
        """
        normalize if necessary,used in regularize methods
        Returns:

        """
        beta_norm = self.model.sum(axis=1)
        self.model = self.model / beta_norm[:, np.newaxis]

    def print_top_words(self, num_words, vocab_file, show_topics=None, result_file=None):
        """
        display top words of topics:
        Args:
            num_words: number of words which is displayed in each topic
            vocab_file: vocabulary file of copus
            show_topics: show_topics: number of topics which we want to show; otherwise, it'll show all topics
            result_file: result_file: write topics with words and probability respectly into this file;
                    otherwise, print into screen

        Returns:

        """
        # get the vocabulary
        if vocab_file:
            if not os.path.isfile(vocab_file):
                logging.error('Unknown file %s', vocab_file)
            vocab = open(vocab_file, 'r').readlines()
            vocab = map(lambda x: x.strip(), vocab)
        else:
            vocab = range()
        if show_topics is not None:
            index_list = np.random.randint(self.num_topics, size=show_topics)
            topic_list = self.model[index_list, :]
        else:
            topic_list = self.model
        if result_file is not None:
            # open file to write
            fp = open(result_file, 'w')
            topic_no = 0
            for topic in topic_list:
                fp.write('topic %03d\n' % topic_no)
                index_sorted = np.argsort(topic)[::-1]  # sort in decending order
                for i in range(num_words):
                    index = index_sorted[i]
                    fp.write('   %s \t\t %f\n' % (vocab[index], topic[index]))
                topic_no = topic_no + 1
                fp.write('\n')
            fp.close()
        else:
            # display to screen
            topic_no = 0
            for topic in topic_list:
                print('topic %03d\n' % (topic_no))
                index_sorted = np.argsort(topic)[::-1]
                for i in range(num_words):
                    index = index_sorted[i]
                    print('   %s \t\t %f\n' % (vocab[index], topic[index]))
                topic_no += 1
                print('\n')

    def load(self, model_file):
        """
        load model (beta or lambda) from a file which is learnt to learn continue
        Args:
            model_file:

        Returns:

        """
        if os.path.isfile(model_file):
            tail = model_file.split('.')[-1]
            assert tail != 'txt' or tail != 'npy', \
                'Unsupported format.Please convert to .txt (text file) or .npy (binary file)!'
            if tail == 'txt':
                f = open(model_file)
                lines = f.readlines()
                words = lines[0].strip().split()
                K = len(lines)
                W = len(words)
                beta = np.zeros((K, W))
                for i in xrange(K):
                    words = lines[0].strip().split()
                    if len(words) != W:
                        print('File %s is error' % model_file)
                        exit()
                    for j in xrange(W):
                        beta[i][j] = float(words[j])
                f.close()
            elif tail == 'npy':
                beta = np.load(model_file)
            self.model = beta
            self.num_topics = beta.shape[0]
            self.num_terms = beta.shape[1]
            return beta
        else:
            print('Unknown file %s' % model_file)
            exit()

    def save(self, model_file, file_type='binary'):
        """
            save model into a file.
            <optional>: type file default is binary, file is saved with tail is .npy
                        type file is text, file is saved with format .txt
        """
        type_file = file_type.lower()
        if type_file == 'text':
            tail = model_file.split('.')[-1]
            filename = model_file[:-(len(tail))] + 'txt'
            f = open(filename, 'w')
            for k in range(self.num_topics):
                for i in range(self.num_terms - 1):
                    f.write('%.10f ' % (self.model[k][i]))
                f.write('%.10f\n' % (self.model[k][self.num_terms - 1]))
            f.close()
        else:
            tail = model_file.split('.')[-1]
            filename = model_file[:-(len(tail))] + 'npy'
            np.save(filename, self.model)


if __name__ == '__main__':
    lda = LdaModel(10, 10)
    lda.save('abc.txt')
