import sys
import os
from ..datasets import base
from ldamodel import LdaModel

import logging

FORMAT = "%(levelname)s> In %(module)s.%(funcName)s line %(lineno)d at %(asctime)-s> %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningStatistics(object):
    """docstring for ClassName"""

    def __init__(self):
        self.e_step_time = []
        self.m_step_time = []
        self.iter_time = []
        self.sparsity_record = []

    def record_time(self, time_e, time_m):
        self.e_step_time.append(time_e)
        self.m_step_time.append(time_m)
        self.iter_time.append(time_e + time_m)

    def reset_time_record(self):
        self.e_step_time = []
        self.m_step_time = []
        self.iter_time = []

    def record_sparsity(self, sparsity):
        self.sparsity_record.append(sparsity)

    def reset_sparsity_record(self):
        self.sparsity_record = []

    def save_time(self, file_name, reset=False):
        if reset:
            f = open(file_name, 'a')
        else:
            f = open(file_name, 'w')
        for idx in range(len(self.iter_time)):
            f.write('%f, %f, %f,\n' % (self.e_step_time[idx], self.m_step_time[idx],
                                       self.iter_time[idx]))

        f.close()
        if reset:
            self.reset_time_record()

    def save_sparsity(self, file_name, reset=False):
        f = open(file_name, 'a')
        for sparsity in self.sparsity_record:
            f.write('%.10f,' % (sparsity))

        f.close()
        if reset:
            self.reset_sparsity_record()


class LdaLearning(object):
    """docstring for LdaLearning"""

    def __init__(self, num_terms, num_topics, lda_model=None):
        self.statistics = LearningStatistics()
        self.num_terms = num_terms
        self.num_topics = num_topics
        # Checking shape of input model
        if lda_model is not None:
            assert lda_model.model.shape != (num_topics, num_terms), "Shape error: model must be shape of " \
                                                                     "(num_topics * num_terms)"
        self.lda_model = lda_model

    def static_online(self, word_ids_tks, cts_lens):
        raise NotImplementedError("Should have implemented static_online!")

    def __getitem__(self, docs):
        raise NotImplementedError("Should have implemented this!")

    def learn_model(self, formatted_data, format_type='tf', batch_size=5000, shuffle=False, passes=1, save_model_every=0,
                    compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20,
                    vocab_file='', model_folder='model'):
        """

        Args:
            formatted_data:
            format_type:
            batch_size:
            shuffle:
            passes:
            save_model_every:
            compute_sparsity_every:
            save_statistic:
            save_top_words_every:
            num_top_words:
            vocab_file:
            model_folder:

        Returns:

        """
        mini_batch_no = 0
        logger.info("Start learning Lda model, %i passes over", passes)

        # Check format
        if format_type == 'tf':
            assert formatted_data.term_freq, \
                'For this learning method, formatted data must by type term- frequency'
        elif format_type == 'sq':
            assert formatted_data.term_seq, \
                'For this learning method, formatted data must by type term sequences'
        else:
            raise AssertionError("Corpus format type must be term-frequency (tf) or sequences (sq)!")
        # Iterating
        for pass_no in range(passes):
            logger.info('Pass no: %s', pass_no)
            train_file = formatted_data.data_path
            if shuffle:
                train_file = formatted_data.shuffle()
            datafp = open(train_file, 'r')
            while True:
                mini_batch = formatted_data.load_mini_batch(datafp, batch_size)
                if len(mini_batch.word_ids_tks) == 0:
                    break
                mini_batch_no += 1
                logger.info("Mini batch no: %s", mini_batch_no)

                # run expectation - maximization algorithms
                time_e, time_m, theta = self.static_online(mini_batch.word_ids_tks, mini_batch.cts_lens)
                self.statistics.record_time(time_e, time_m)

                # compute documents sparsity
                if compute_sparsity_every > 0 and (mini_batch_no % compute_sparsity_every) == 0:
                    sparsity = base.compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
                    self.statistics.record_sparsity(sparsity)

                # save model : lambda, beta, N_phi
                if save_model_every > 0 and (mini_batch_no % save_model_every) == 0:
                    model_file = model_folder + '/model' + str(mini_batch_no)
                    self.lda_model.save(model_file)

                # save top words
                if save_top_words_every > 0 and (mini_batch_no % save_top_words_every) == 0:
                    top_words_file = model_folder + '/top_words_' + str(mini_batch_no) + '.txt'
                    self.lda_model.print_top_words(num_top_words, vocab_file, top_words_file)
            datafp.close()

        # save learning statistic
        if save_statistic:
            time_file = model_folder + '/time' + str(mini_batch_no) + '.csv'
            sparsity_file = model_folder + '/sparsity' + str(mini_batch_no) + '.csv'
            self.statistics.save_time(time_file)
            self.statistics.save_sparsity(sparsity_file)
        # Finish
        logger.info('Finish training!!!')
        return self.lda_model
