import sys, os
import pandas as pd
import numpy as np
import logging


class LdaModel(object):
    """docstring for ClassName"""

    def __init__(self, num_terms=None, num_topics=None, random_type=0):
        """

        Args:
            num_topics:
            num_terms:
            random_type:
        """
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.random_type = random_type
        if num_topics != None and num_terms != None:
            self.presence_score = np.zeros(num_topics)
            if self.random_type == 0:
                self.model = np.random.rand(self.num_topics, self.num_terms) + 1e-10
            else:
                self.model = 1 * np.random.gamma(100., 1. / 100., (self.num_topics, self.num_terms))

    def save(self, path_file):
        store = pd.HDFStore(path_file)
        model_frame = pd.DataFrame(self.model)
        presence_score_frame = pd.DataFrame(self.presence_score)
        if '/model' in store.keys():
            store.remove('model')
        if '/presence_score' in store.keys():
            store.remove('presence_score')
        store['model'] = model_frame
        store['presence_score'] = presence_score_frame
        store.close()

    def load(self, path_file):
        store = pd.HDFStore(path_file, 'r')
        self.model = store['model'].values
        self.presence_score = store['presence_score'][0].values
        self.num_topics = self.model.shape[0]
        self.num_terms = self.model.shape[1]
        store.close()

    def normalize(self):
        """
        normalize if necessary,used in regularize methods
        Returns:

        """
        beta_norm = self.model.sum(axis=1)
        self.model = self.model / beta_norm[:, np.newaxis]
        return self.model

    def print_top_words(self, num_words, vocab_file, show_topics=None, display_result=None, type='word', distribution=False):
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
        if not os.path.isfile(vocab_file):
            logging.error('Unknown file %s', vocab_file)
            exit()
        fvocab = open(vocab_file, 'r')
        vocab = fvocab.readlines()
        fvocab.close()
        vocab = map(lambda x: x.strip().split()[-1], vocab)

        if show_topics is not None:
            index_list = np.random.randint(self.num_topics, size=show_topics)
            topic_list = self.model[index_list, :]
        else:
            topic_list = self.model
        top_words = list()
        if display_result is not None and display_result != 'screen':
            # open file to write
            fp = open(display_result, 'w')
            topic_no = 0
            for topic in topic_list:
                str_topic = 'topic %d: ' % topic_no
                #fp.write('topic %d\n' % topic_no)
                index_sorted = np.argsort(topic)[::-1]  # sort in decending order
                words = list()
                for i in range(num_words):
                    index = index_sorted[i]
                    if distribution:
                        if type == 'word':
                            str_topic += '(%s %f), ' %(vocab[index], topic[index])
                        else:
                            str_topic += '(%d %f), ' %(index, topic[index])
                    else:
                        if type == 'word':
                            str_topic += '%s, ' %(vocab[index])
                        else:
                            str_topic += '%d, ' %(index)
                    #fp.write('   %s \t\t %f\n' % (vocab[index], topic[index]))
                    words.append(vocab[index])
                fp.write('%s\n' %str_topic)
                top_words.append(words)
                topic_no = topic_no + 1
                fp.write('\n')
            fp.close()
        else:
            # display to screen
            topic_no = 0
            for topic in topic_list:
                str_topic = 'topic %d: ' % topic_no
                index_sorted = np.argsort(topic)[::-1]
                words = list()
                for i in range(num_words):
                    index = index_sorted[i]
                    if display_result == 'screen':
                        if distribution:
                            if type == 'word':
                                str_topic += '(%s %f), ' % (vocab[index], topic[index])
                            else:
                                str_topic += '(%d %f), ' % (index, topic[index])
                        else:
                            if type == 'word':
                                str_topic += '%s, ' % (vocab[index])
                            else:
                                str_topic += '%d, ' % (index)
                    words.append(vocab[index])
                if display_result == 'screen':
                    print('%s' %str_topic)
                top_words.append(words)
                topic_no += 1

        return top_words

    def load_model(self, path_file):
        """
        load model (beta or lambda) from a file which is learnt to learn continue
        Args:
            path_file:

        Returns:

        """
        if os.path.isfile(path_file):
            tail = path_file.split('.')[-1]
            assert tail != 'txt' or tail != 'npy', \
                'Unsupported format.Please convert to .txt (text file) or .npy (binary file)!'
            if tail == 'txt':
                f = open(path_file)
                lines = f.readlines()
                words = lines[0].strip().split()
                K = len(lines)
                W = len(words)
                beta = np.zeros((K, W))
                for i in xrange(K):
                    words = lines[0].strip().split()
                    if len(words) != W:
                        print('File %s is error' % path_file)
                        exit()
                    for j in xrange(W):
                        beta[i][j] = float(words[j])
                f.close()
            elif tail == 'npy':
                beta = np.load(path_file)
            self.model = beta
            self.num_topics = beta.shape[0]
            self.num_terms = beta.shape[1]
            self.presence_score = np.zeros(self.num_topics)

            return beta
        else:
            print('Unknown file %s' % path_file)
            exit()

    def save_model(self, path_file, file_type='binary'):
        """
            save model into a file.
            <optional>: type file default is binary, file is saved with tail is .npy
                        type file is text, file is saved with format .txt
        """
        type_file = file_type.lower()
        if type_file == 'text':
            tail = path_file.split('.')[-1]
            filename = path_file[:-(len(tail))] + 'txt'
            f = open(filename, 'w')
            for k in range(self.num_topics):
                for i in range(self.num_terms - 1):
                    f.write('%.10f ' % (self.model[k][i]))
                f.write('%.10f\n' % (self.model[k][self.num_terms - 1]))
            f.close()
        else:
            tail = path_file.split('.')[-1]
            filename = path_file[:-(len(tail))] + 'npy'
            np.save(filename, self.model)

    def save_presence_score_topics(self, path_file):
        np.save(path_file, self.presence_score)

    def load_presence_score_topics(self, path_file):
        arr = np.load(path_file)
        self.presence_score = arr
        return arr

if __name__ == '__main__':
    lda = LdaModel(10, 10)
    lda.save_model('abc.txt')
