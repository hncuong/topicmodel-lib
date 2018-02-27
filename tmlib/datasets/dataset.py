import logging, os, re
import numpy as np
import pandas as pd
import utilizies
from .utilizies import Corpus, DataIterator, DataFormat

logger = logging.getLogger(__name__)


class DataSet(DataIterator):
    def __init__(self, data_path=None, batch_size=None, passes=1, shuffle_every=None,
                 vocab_file=None, label=False):
        """
        A class for loading mini-batches of data from a file
        Args:
            data_path: path to file, type(str)
            batch_size: size of a mini-batch
            passes: number of passes over the origin data
            shuffle_every: shuffle the origin data after a number of passes
        """
        super(DataSet, self).__init__()
        # pre process data if needed
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.passes = passes
        self.shuffle_every = shuffle_every
        if data_path is not None:
            self.input_format = utilizies.check_input_format(data_path)
            self.original_data_path = data_path

            if self.input_format == DataFormat.RAW_TEXT:
                folder_data, vocab_file, tf_file = utilizies.pre_process(data_path)
                self.folder_data = folder_data
                self.data_path = tf_file
                self.vocab_file = vocab_file
                self.data_format = DataFormat.TERM_FREQUENCY
            else:
                if vocab_file is None:
                    logger.error('File input is formatted, but you must supply file vocabulary!')
                    raise ValueError('Value of vocab_file should not be None')
                if not os.path.isfile(vocab_file):
                    logger.error("Can't find the vocabulary file")
                    raise ValueError('The vocabulary file is not exist')
                file_name = os.path.split(data_path)[-1]
                self.folder_data = utilizies.get_data_home() + '/' + file_name.split('.')[0]
                if not os.path.exists(self.folder_data):
                    os.mkdir(self.folder_data)
                self.data_path = data_path
                self.data_format = self.input_format
            self.output_format = DataFormat.TERM_FREQUENCY

            logging.info('Get number of documents in corpus...')
            self.num_docs = self.get_num_docs_per_pass()
            if self.num_docs == 0:
                logging.info('Corpus is empty!')
                exit()
            self.num_batches_per_pass = np.ceil(float(self.num_docs) / self.batch_size)

            self.pass_no = 1
            self.batch_no_in_pass = 0

            if self.pass_no == self.passes:
                self.work_path = self.data_path
            else:
                self.work_path = utilizies.shuffle_formatted_data_file(self.data_path, self.folder_data)

            self.fp = None
            self.end_of_file = False
            self.database_path = None
            
            self.label = label


    def load_mini_batch(self):
        """

        Returns:

        """

        if self.end_of_file:
            # reset batch no and increase pass no
            self.batch_no_in_pass = 0
            self.pass_no += 1
            self.end_of_file = False
            logger.info('Pass no: %s', self.pass_no)
            # shuffle after number of passes
            if self.shuffle_every > 0 and (self.pass_no % self.shuffle_every == 0) and not (self.pass_no==self.passes):
                self.work_path = utilizies.shuffle_formatted_data_file(self.data_path, self.folder_data)
            if self.pass_no == self.passes:
                self.work_path = self.data_path

        if self.batch_no_in_pass == 0:
            self.fp = open(self.work_path, 'r')

        logger.info("Mini batch no: %s", self.batch_no_in_pass)

        mini_batch = self.load_mini_batch_and_state_with_format()

        # check end of file
        if (self.batch_no_in_pass + 1) == self.num_batches_per_pass:
            self.end_of_file = True
            self.fp.close()

        self.batch_no_in_pass += 1
        self.mini_batch_no += 1

        return mini_batch

    def load_mini_batch_and_state_with_format(self):
        """

        Returns:

        """
        if self.output_format == DataFormat.TERM_FREQUENCY:
            if self.data_format == DataFormat.TERM_FREQUENCY:
                return self.load_mini_batch_term_frequency_from_term_frequency_file()
            else:
                return self.load_mini_batch_term_frequency_from_sequence_file()
        else:
            if self.data_format == DataFormat.TERM_FREQUENCY:
                return self.load_mini_batch_term_sequence_from_term_frequency_file()
            else:
                return self.load_mini_batch_term_sequence_from_sequence_file()

    def load_mini_batch_term_sequence_from_sequence_file(self):
        """

        Args:
            fp:
            batch_size:

        Returns:

        """
        mini_batch = Corpus(DataFormat.TERM_SEQUENCE)
        # end_file = False
        try:
            for i in range(0, self.batch_size):
                doc = self.fp.readline()
                # check end file
                if len(doc) < 1:
                    # end_file = True
                    break
                list_word = doc.strip().split()
                N = len(list_word)
                doc_terms = np.zeros(N, dtype=np.int32)
                for j in range(N):
                    doc_terms[j] = int(list_word[j])
                doc_length = N
                mini_batch.append_doc(doc_terms, doc_length)

            return mini_batch
        except Exception as inst:
            logging.error(inst)

    def load_mini_batch_term_sequence_from_term_frequency_file(self):
        """

        Args:
            fp:
            batch_size:

        Returns:

        """
        mini_batch = Corpus(DataFormat.TERM_SEQUENCE, label=self.label)
        # end_file = False
        try:
            for i in range(0, self.batch_size):
                doc = self.fp.readline()
                # check end file
                if len(doc) < 1:
                    # end_file = True
                    break
                list_word = doc.strip().split()
                N = int(list_word[0])
                if self.label:
                    mini_batch.append_label(N)
                else:
                    if N + 1 != len(list_word):
                        logging.error("Line in file Term frequency is error!")
                tokens = list()
                for j in range(1, len(list_word)):
                    tf = list_word[j].split(":")
                    for k in range(0, int(tf[1])):
                        tokens.append(int(tf[0]))
                mini_batch.append_doc(np.array(tokens), len(tokens))

            return mini_batch  # , end_file
        except Exception as inst:
            logging.error(inst)

    def load_mini_batch_term_frequency_from_term_frequency_file(self):
        """

        Args:
            fp:
            batch_size:

        Returns:

        """
        mini_batch = Corpus(DataFormat.TERM_FREQUENCY, label=self.label)
        # end_file = False
        try:
            for i in range(0, self.batch_size):
                doc = self.fp.readline()
                # check end file
                if len(doc) < 1:
                    # end_file = True
                    break
                list_word = doc.strip().split()
                N = int(list_word[0])
                if self.label:
                    mini_batch.append_label(N)
                else:
                    if N + 1 != len(list_word):
                        logging.error("Line in file Term frequency is error!")
                doc_terms = np.zeros(len(list_word)-1, dtype=np.int32)
                doc_frequency = np.zeros(len(list_word)-1, dtype=np.int32)
                for j in range(1, len(list_word)):
                    tf = list_word[j].split(":")
                    doc_terms[j - 1] = int(tf[0])
                    doc_frequency[j - 1] = int(tf[1])
                mini_batch.append_doc(doc_terms, doc_frequency)

            return mini_batch  # , end_file
        except Exception as inst:
            logging.error(inst)

    def load_mini_batch_term_frequency_from_sequence_file(self):
        """

        Args:
            fp:
            batch_size:

        Returns:

        """
        mini_batch = Corpus(DataFormat.TERM_FREQUENCY)
        # end_file = False
        try:
            for i in range(0, self.batch_size):
                doc = self.fp.readline()
                # check end file
                if len(doc) < 1:
                    # end_file = True
                    break
                list_word = doc.strip().split()
                term_frequency_dict = dict()
                for term in list_word:
                    term = int(term)
                    if term not in term_frequency_dict:
                        term_frequency_dict[term] = 1
                    else:
                        term_frequency_dict[term] += 1
                mini_batch.append_doc(np.array(term_frequency_dict.keys()),
                                      np.array(term_frequency_dict.values()))

            return mini_batch  # , end_file
        except Exception as inst:
            logging.error(inst)

    def load_new_documents(self, path_file, vocab_file=None, label=False):
        format = utilizies.check_input_format(path_file)
        if format == DataFormat.RAW_TEXT:
            if vocab_file is None:
                logging.error("File vocabulary not found")
                raise ValueError("File vocabulary not found")
            elif not os.path.isfile(vocab_file):
                logging.error("File vocabulary not found")
                raise ValueError("File vocabulary not found")

            # read vocab and save as dictionary vocab[word] = index
            list_vocab = open(path_file).readlines()
            dict_vocab = dict()
            for word in list_vocab:
                word = word.lower()
                word = re.sub(r'[^a-z]', '', word)
                dict_vocab[word] = len(dict_vocab)
            del list_vocab

            docs = utilizies.load_batch_raw_text(path_file)
            corpus = Corpus(DataFormat.TERM_FREQUENCY)
            for d in range(0, len(docs)):
                docs[d] = docs[d].lower()
                docs[d] = re.sub(r'-', ' ', docs[d])
                docs[d] = re.sub(r'[^a-z ]', '', docs[d])
                docs[d] = re.sub(r' +', ' ', docs[d])
                words = docs[d].split()
                ddict = dict()
                for word in words:
                    if (word in dict_vocab):
                        wordtoken = dict_vocab[word]
                        if (not wordtoken in ddict):
                            ddict[wordtoken] = 0
                        ddict[wordtoken] += 1
                corpus.append_doc(ddict.keys(), ddict.values())
            return corpus
        elif format == DataFormat.TERM_FREQUENCY:
            corpus = Corpus(DataFormat.TERM_FREQUENCY, label=label)
            try:
                fp = open(path_file)
                line = fp.readline()
                while line:
                    doc = line.strip()
                    # check end file
                    if len(doc) < 1:
                        # end_file = True
                        break
                    list_word = doc.strip().split()
                    N = int(list_word[0])
                    if label:
                        corpus.append_label(label)
                    else:
                        if N + 1 != len(list_word):
                            logging.error("Line in file Term frequency is error!")
                    doc_terms = np.zeros(len(list_word)-1, dtype=np.int32)
                    doc_frequency = np.zeros(len(list_word)-1, dtype=np.int32)
                    for j in range(1, len(list_word)):
                        tf = list_word[j].split(":")
                        doc_terms[j - 1] = int(tf[0])
                        doc_frequency[j - 1] = int(tf[1])
                    corpus.append_doc(doc_terms, doc_frequency)
                    line = fp.readline()
                fp.close()

                return corpus  # , end_file
            except Exception as inst:
                logging.error(inst)
        else:
            corpus = Corpus(DataFormat.TERM_SEQUENCE)
            try:
                fp = open(path_file)
                line = fp.readline()
                while line:
                    doc = line.strip()
                    # check end file
                    if len(doc) < 1:
                        # end_file = True
                        break
                    list_word = doc.split()
                    N = len(list_word)
                    doc_terms = np.zeros(N, dtype=np.int32)
                    for j in range(N):
                        doc_terms[j] = int(list_word[j])
                    doc_length = N
                    corpus.append_doc(doc_terms, doc_length)
                    line = fp.readline()
                fp.close()

                return corpus
            except Exception as inst:
                logging.error(inst)

    def check_end_of_data(self):
        if self.end_of_file and self.pass_no == self.passes:
            self.end_of_data = True
            if self.database_path is not None:
                if os.path.exists(self.database_path):
                    self.database.close()

        return self.end_of_data

    def set_output_format(self, output_format):
        assert (output_format == DataFormat.TERM_SEQUENCE or output_format == DataFormat.TERM_FREQUENCY), \
            'Corpus format type must be term-frequency (tf) or sequences (sq)!'
        self.output_format = output_format

    def get_num_docs_per_pass(self):
        num_docs = 0
        for line in open(self.data_path, 'r'):
            num_docs += 1
        return num_docs

    def get_total_docs(self):
        return self.num_docs * self.passes

    def get_num_tokens(self):
        num_tokens = 0
        if self.data_format == DataFormat.TERM_FREQUENCY:
            fp = open(self.data_path)
            line = fp.readline()
            while line:
                arr = line.strip().split()
                N = int(arr[0])
                for i in range(1, N+1):
                    tf = arr[i].split(':')
                    num_tokens += int(tf[1])
                line = fp.readline()
        elif self.data_format == DataFormat.TERM_SEQUENCE:
            fp = open(self.data_path)
            line = fp.readline()
            while line:
                tks = line.strip().split()
                num_tokens += len(tks)
                line = fp.readline()
        else:
            logger.error('Format of data input is not valid')
            raise ValueError('Format of data input is not valid')
        return num_tokens

    def get_num_terms(self):
        f = open(self.vocab_file, 'r')
        list_terms = f.readlines()
        return len(list_terms)

    def init_database(self, database_path):
        self.database_path = database_path
        self.database = pd.HDFStore(database_path, 'w')

    def store_topic_proportions(self, theta):
        #self.table_name_topic_propotions = table_name
        if self.pass_no == self.passes:
            dist_topics = ['dist_topic' + str(i) for i in range(theta.shape[1])]
            start = (self.batch_no_in_pass-1)*self.batch_size
            if self.batch_no_in_pass < self.num_batches_per_pass:
                end = self.batch_no_in_pass * self.batch_size
                theta_frame = pd.DataFrame(theta, columns=dist_topics, index=list(range(start,end)))
            else:
                end = start+theta.shape[0]
                theta_frame = pd.DataFrame(theta, columns=dist_topics, index=list(range(start, end)))
            self.database.append('theta', theta_frame, data_columns=True, complevel=9, complib='blosc')

if __name__ == '__main__':
    data = DataSet(data_path='/home/khangtg/Documents/news20.dat', batch_size=15935, vocab_file='/home/khangtg/Documents/vocab.txt', label=True)
    corpus = data.load_mini_batch()
    print corpus.labels[:10]