import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
from tmlib.datasets import utilizies
from ldamodel import LdaModel
from tmlib.datasets.dataset import DataSet

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

    def __init__(self, data=None, num_topics=None, lda_model=None):
        self.statistics = LearningStatistics()
        self.data = data
        self.num_topics = num_topics
        self.lda_model = lda_model
        if data is not None:
            self.num_terms = self.data.get_num_terms()
            # Checking shape of input model
            if lda_model is not None:
                assert lda_model.model.shape == (num_topics, self.num_terms), "Shape error: model must be shape of " \
                                                                              "(num_topics * num_terms)"

    def static_online(self, word_ids_tks, cts_lens):
        raise NotImplementedError("Should have implemented static_online!")

    def estimate_topic_proportions(self, param_theta):
        raise NotImplementedError("Should have implemented estimate_topic_proportions!")

    def learn_model(self, save_statistic=False, save_model_every=0, compute_sparsity_every=0,
                    save_top_words_every=0, num_top_words=10, model_folder=None, save_topic_proportions=None):
        """

        Args:
            data:
            save_model_every:
            compute_sparsity_every:
            save_statistic:
            save_top_words_every:
            num_top_words:
            model_folder:

        Returns:

        """
        mini_batch_no = 0
        # create model_folder
        if model_folder is not None:
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
        if save_topic_proportions is not None:
            self.data.init_database(save_topic_proportions)

        logger.info("Start learning Lda model, passes over")

        # Iterating
        while not self.data.check_end_of_data():
            mini_batch = self.data.load_mini_batch()

            # This using for streaming method
            if self.num_terms != self.data.get_num_terms():
                self.num_terms = self.data.get_num_terms()
                new_model = LdaModel(self.num_terms, self.num_topics, random_type=1)
                new_model.model[:, :self.lda_model.model.shape[1]] = self.lda_model.model
                self.lda_model = new_model

            # run expectation - maximization algorithms
            time_e, time_m, param_theta = self.static_online(mini_batch.word_ids_tks, mini_batch.cts_lens)
            theta = self.estimate_topic_proportions(param_theta)
            if save_topic_proportions is not None:
                self.data.store_topic_proportions(theta)
            self.lda_model.presence_score += theta.sum(axis=0)
            del theta
            self.statistics.record_time(time_e, time_m)

            # compute documents sparsity
            if compute_sparsity_every > 0 and (self.data.mini_batch_no % compute_sparsity_every) == 0:
                sparsity = utilizies.compute_sparsity(param_theta, param_theta.shape[0], param_theta.shape[1], 't')
                self.statistics.record_sparsity(sparsity)

            # save model : lambda, beta, N_phi
            if save_model_every > 0 and (self.data.mini_batch_no % save_model_every) == 0:
                model_file = model_folder + '/model_batch' + str(mini_batch_no) + '.txt'
                self.lda_model.save(model_file)

            # save top words
            if save_top_words_every > 0 and (self.data.mini_batch_no % save_top_words_every) == 0:
                top_words_file = model_folder + '/top_words_batch_' + str(mini_batch_no) + '.txt'
                self.lda_model.print_top_words(num_top_words, vocab_file=self.data.vocab_file, display_result=top_words_file)

            if self.data.end_of_file and not self.data.check_end_of_data():
                self.lda_model.presence_score *= 0
            mini_batch_no += 1

        # save learning statistic
        if save_statistic:
            time_file = model_folder + '/time' + str(self.data.mini_batch_no) + '.csv'
            self.statistics.save_time(time_file)
            if compute_sparsity_every > 0:
                sparsity_file = model_folder + '/sparsity' + str(self.data.mini_batch_no) + '.csv'
                self.statistics.save_sparsity(sparsity_file)
        # Finish
        logger.info('Finish training!!!')
        return self.lda_model
