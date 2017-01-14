import os, os.path
import sys
import shutil
from os.path import isdir, isfile, join
import numpy as np
from time import time
import logging
from tmlib.preprocessing import preprocessing
from tmlib import config

# Name of current path directory which contains this file
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_data_home(data_home=None):
    """Return the path of the tmlib data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'tmlib_data'
    in the user home folder.
    The '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = config.get_config('datasets', 'TMLIB_DATA_HOME')
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache."""
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


class DataFormat(object):
    TERM_FREQUENCY = 'tf'
    TERM_SEQUENCE = 'sq'
    RAW_TEXT = 'txt'


def check_input_format(file_path):
    """
    check format of input file(text formatted or raw text)
    Args:
        file_path: path of file input

    Returns:

    """
    if isfile(file_path):
        f = open(file_path)
        line = f.readline().strip()
        while len(line) == 0:
            line = f.readline().strip()

        if line == '<DOC>':
            result = DataFormat.RAW_TEXT
        else:
            result = DataFormat.TERM_SEQUENCE
            l = len(line)
            for i in range(0, l):
                if line[i].isalnum() or line[i] == ' ' or line[i] == ':':
                    if line[i] == ':':
                        result = DataFormat.TERM_FREQUENCY
                else:
                    logging.error('File %s is not true format!', file_path)
            f.close()
        return result
    else:
        logging.error('Unknown file %s', file_path)


def pre_process(file_path):
    #self.is_raw_text = True
    """
    :param file_path: path of file input
    :return: list which respectly include vocabulary file, tf, sq file after preprocessing
    """

    logging.info("Pre-processing raw text format file %s", file_path)
    p = preprocessing.PreProcessing(file_path)
    p.process()
    folder_data = get_data_home() + '/' + p.main_name_file
    if not os.path.exists(folder_data):
        os.makedirs(folder_data)

    p.extract_vocab(folder_data)
    p.save_format_tf(folder_data)
    p.save_format_sq(folder_data)

    return p.path_file_vocab, p.path_file_tf, p.path_file_sq


def reformat_file_to_term_sequence(file_path):
    format_type = check_input_format(file_path)
    if format_type == DataFormat.TERM_FREQUENCY:
        file = open(file_path)
        lines = file.readlines()
        doc_tks = list()
        for line in lines:
            list_words = line.strip().split()
            N = int(list_words[0])
            tokens = list()
            lens = list()
            for i in range(1,N+1):
                id_f = list_words[i].split(':')
                for k in range(0, int(id_f[1])):
                    tokens.append(int(id_f[0]))
            doc_tks.append(tokens)
        file_name = file_path.split('\\')[-1].split('/')[-1]
        main_name = file_name.split('.')[0]
        folder = get_data_home() + '/' + main_name
        if not os.path.exists(folder):
            os.mkdir(folder)
        fout = open(join(folder, main_name+'.sq'), 'w')
        for i in range(len(doc_tks)):
            N = len(doc_tks[i])
            fout.write('%d' %N)
            for j in range(N):
                fout.write(' %d' %doc_tks[i][j])
                if j == N-1 and i < (len(doc_tks)-1):
                    fout.write('\n')
        fout.close()
        return folder + '/' + main_name + '.sq'
    elif format_type == DataFormat.TERM_SEQUENCE:
        return file_path
    else:
        logging.error('File %s need to preprocessing' %file_path)


def reformat_file_to_term_frequency(file_path):
    format_type = check_input_format(file_path)
    if format_type == DataFormat.TERM_SEQUENCE:
        file = open(file_path)
        lines = file.readlines()
        doc_ids = list()
        doc_cts = list()
        for line in lines:
            list_tks = line.strip().split()
            N = int(list_tks[0])
            ids = list()
            cts = list()
            for i in range(1,N+1):
                id = int(list_tks[i])
                if id not in ids:
                    ids.append(id)
                    cts.append(1)
                else:
                    index = ids.index(id)
                    cts[index] += 1
            doc_ids.append(np.array(ids))
            doc_cts.append(np.array(cts))
        file_name = file_path.split('\\')[-1].split('/')[-1]
        main_name = file_name.split('.')[0]
        folder = get_data_home() + '/' + main_name
        if not os.path.exists(folder):
            os.mkdir(folder)
        fout = open(join(folder, main_name+'.tf'), 'w')
        for i in range(len(doc_cts)):
            N = len(doc_ids[i])
            fout.write('%d' %N)
            for j in range(N):
                fout.write(' %d:%d' %(doc_ids[i][j], doc_cts[i][j]))
                if j == (N-1) and i < (len(doc_cts)-1):
                    fout.write('\n')
        fout.close()
        return folder + '/' + main_name + '.tf'
    elif format_type == DataFormat.TERM_FREQUENCY:
        return file_path
    else:
        logging.error('File %s need to preprocessing' %file_path)


class Corpus(object):
    def __init__(self, format_type):
        assert format_type == DataFormat.TERM_FREQUENCY or format_type == DataFormat.TERM_SEQUENCE, \
            "Corpus format type must be term-frequency (tf) or sequences (sq)!"
        self.word_ids_tks = []
        self.cts_lens = []
        self.format_type = format_type

    def append_doc(self, ids_tks, cts_len):
        self.word_ids_tks.append(ids_tks)
        self.cts_lens.append(cts_len)


def convert_corpus_format(corpus, data_format):
    assert data_format == DataFormat.TERM_FREQUENCY or data_format == DataFormat.TERM_SEQUENCE, \
        "Corpus format type must be term-frequency (tf) or sequences (sq)!"
    if data_format == DataFormat.TERM_SEQUENCE:
        return convert_corpus_to_term_sequence(corpus)
    else:
        return convert_corpus_to_term_frequency(corpus)


def convert_corpus_to_term_sequence(corpus):
    try:
        if corpus.format_type == DataFormat.TERM_FREQUENCY:
            formatted_corpus = Corpus(DataFormat.TERM_SEQUENCE)
            for idx , doc_terms in enumerate(corpus.word_ids_tks):
                doc_frequencies = corpus.cts_lens[idx]
                doc_sequence = []
                for _idx, term in enumerate(doc_terms):
                    frequency = doc_frequencies[_idx]
                    for count in range(frequency):
                        doc_sequence.append(term)
                doc_length = len(doc_sequence)
                formatted_corpus.append_doc(doc_sequence, doc_length)
            return formatted_corpus
        return corpus
    except Exception as inst:
        logging.error(inst)


def convert_corpus_to_term_frequency(corpus):
    try:
        if corpus.format_type == DataFormat.TERM_SEQUENCE:
            formatted_corpus = Corpus(DataFormat.TERM_FREQUENCY)
            for doc_terms in corpus.word_ids_tks:
                term_frequency_dict = dict()
                for term in doc_terms:
                    if term not in term_frequency_dict:
                        term_frequency_dict[term] = 1
                    else:
                        term_frequency_dict[term] += 1
                formatted_corpus.append_doc(np.array(term_frequency_dict.keys()),
                                            np.array(term_frequency_dict.values()))
            return formatted_corpus
        return corpus
    except Exception as inst:
        logging.error(inst)


class DataIterator(object):
    """docstring for DataIterator"""
    def __init__(self):
        self.mini_batch_no = 0
        self.end_of_data = False

    def load_mini_batch(self):
        raise NotImplementedError("This functions need to be implemented")

    def end_of_data(self):
        raise NotImplementedError("This functions need to be implemented")


def load_mini_batch_term_sequence_from_sequence_file(fp, batch_size):
    """

    Args:
        fp:
        batch_size:

    Returns:

    """
    mini_batch = Corpus(DataFormat.TERM_SEQUENCE)
    end_file = False
    try:
        for i in range(0, batch_size):
            doc = fp.readline()
            # check end file
            if len(doc) < 1:
                end_file = True
                break
            list_word = doc.strip().split()
            N = len(list_word)
            doc_terms = np.zeros(N, dtype=np.int32)
            for j in range(N):
                doc_terms[j] = int(list_word[j])
            doc_length = N
            mini_batch.append_doc(doc_terms, doc_length)
        return mini_batch, end_file
    except Exception as inst:
        logging.error(inst)


def load_mini_batch_term_sequence_from_term_frequency_file(fp, batch_size):
    """

    Args:
        fp:
        batch_size:

    Returns:

    """
    mini_batch = Corpus(DataFormat.TERM_SEQUENCE)
    end_file = False
    try:
        for i in range(0, batch_size):
            doc = fp.readline()
            # check end file
            if len(doc) < 1:
                end_file = True
                break
            list_word = doc.strip().split()
            N = int(list_word[0])
            if N + 1 != len(list_word):
                logging.error("Line in file Term frequency is error!")
            tokens = list()
            for j in range(1, N + 1):
                tf = list_word[j].split(":")
                for k in range(0, int(tf[1])):
                    tokens.append(int(tf[0]))
            mini_batch.append_doc(np.array(tokens), len(tokens))
        return mini_batch, end_file
    except Exception as inst:
        logging.error(inst)


def load_mini_batch_term_frequency_from_term_frequency_file(fp, batch_size):
    """

    Args:
        fp:
        batch_size:

    Returns:

    """
    mini_batch = Corpus(DataFormat.TERM_SEQUENCE)
    end_file = False
    try:
        for i in range(0, batch_size):
            doc = fp.readline()
            # check end file
            if len(doc) < 1:
                end_file = True
                break
            list_word = doc.strip().split()
            N = int(list_word[0])
            if N + 1 != len(list_word):
                logging.error("Line in file Term frequency is error!")
            doc_terms = np.zeros(N, dtype=np.int32)
            doc_frequency = np.zeros(N, dtype=np.int32)
            for j in range(1, N + 1):
                tf = list_word[j].split(":")
                doc_terms[j - 1] = int(tf[0])
                doc_frequency[j - 1] = int(tf[1])
            mini_batch.append_doc(doc_terms, doc_frequency)
        return mini_batch, end_file
    except Exception as inst:
        logging.error(inst)


def load_mini_batch_term_frequency_from_sequence_file(fp, batch_size):
    """

    Args:
        fp:
        batch_size:

    Returns:

    """
    mini_batch = Corpus(DataFormat.TERM_SEQUENCE)
    end_file = False
    try:
        for i in range(0, batch_size):
            doc = fp.readline()
            # check end file
            if len(doc) < 1:
                end_file = True
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
        return mini_batch, end_file
    except Exception as inst:
        logging.error(inst)


def shuffle_formatted_data_file(data_path, batch_size):
    """
    shuffle input and write into file file_shuffled.txt, return path of this file
    Returns:

    """
    fin = open(data_path)
    for i, l in enumerate(fin):
        pass
    size = i + 1
    fin.seek(0, 0)
    if size % batch_size == 0:
        num_batch = int(size / batch_size)
    else:
        num_batch = int(size / batch_size) + 1
    list_pointers = list()
    for ip in xrange(num_batch):
        fp = open('.'.join([data_path,"shuffle%d"%ip]), "w+")
        list_pointers.append(fp)
    for i in xrange(size):
        rand = np.random.randint(0, num_batch)
        line = fin.readline()
        list_pointers[rand].write(line)
    file_shuffled = '.'.join([data_path, 'shuffled'])
    fout = open(file_shuffled, "w")
    for ip in xrange(0, num_batch):
        list_pointers[ip].seek(0, 0)
        line = list_pointers[ip].readline()
        while line:
            fout.write(line)
            line = list_pointers[ip].readline()
        list_pointers[ip].close()
        os.remove('.'.join([data_path,"shuffle%d"%ip]))
    fout.close()


def compute_sparsity(doc_tp, batch_size, num_topics, _type):
    sparsity = np.zeros(batch_size, dtype=np.float)
    if _type == 'z':
        for d in range(batch_size):
            N_z = np.zeros(num_topics, dtype=np.int)
            N = len(doc_tp[d])
            for i in range(N):
                N_z[doc_tp[d][i]] += 1.
            sparsity[d] = len(np.where(N_z != 0)[0])
    else:
        for d in range(batch_size):
            sparsity[d] = len(np.where(doc_tp[d] > 1e-10)[0])
    sparsity /= num_topics
    return np.mean(sparsity)


def write_topic_mixtures(theta, file_name):
    batch_size = theta.shape[0]
    num_topics = theta.shape[1]
    f = open(file_name, 'a')
    for d in range(batch_size):
        for k in range(num_topics - 1):
            f.write('%.5f ' % (theta[d][k]))
        f.write('%.5f\n' % (theta[d][num_topics - 1]))
    f.close()


if __name__ == '__main__':
    shuffle_formatted_data_file('D:\\python\\grolier\\grolier_train.txt', 200)
    # print(reformat_file_to_term_frequency('/home/ubuntu/tmlib_data/ap/ap.sq'))
