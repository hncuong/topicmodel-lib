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


class InputFormat(object):
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
            result = 'txt'
        else:
            result = InputFormat.TERM_SEQUENCE
            l = len(line)
            for i in range(0, l):
                if line[i].isalnum() or line[i] == ' ' or line[i] == ':':
                    if line[i] == ':':
                        result = InputFormat.TERM_FREQUENCY
                else:
                    logging.error('File %s is not true format!', file_path)
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
    if format_type == InputFormat.TERM_FREQUENCY:
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
    elif format_type == InputFormat.TERM_SEQUENCE:
        return file_path
    else:
        logging.error('File %s need to preprocessing' %file_path)


def reformat_file_to_term_frequency(file_path):
    format_type = check_input_format(file_path)
    if format_type == InputFormat.TERM_SEQUENCE:
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
    elif format_type == InputFormat.TERM_FREQUENCY:
        return file_path
    else:
        logging.error('File %s need to preprocessing' %file_path)


class Corpus(object):
    def __init__(self, format_type):
        assert format_type == InputFormat.TERM_FREQUENCY or format_type == InputFormat.TERM_SEQUENCE, \
            "Corpus format type must be term-frequency (tf) or sequences (sq)!"
        self.word_ids_tks = []
        self.cts_lens = []
        self.format_type = format_type


class DataIterator(object):
    """docstring for DataIterator"""
    def __init__(self):
        self.mini_batch_no = 0
        self.end_of_pass = False

    def load_mini_batch(self):
        raise NotImplementedError("This functions need to be implemented")

    def end_of_data(self):
        raise NotImplementedError("This functions need to be implemented")


class Dataset(object):
    """

    """
    def __init__(self, path):
        """

        Args:
            path:
        """
        self.vocab_path = None
        self.is_raw_text = False
        self.data = None
        if isfile(path):
            logging.info("Path %s is a file", path)
            self.file_path = path
            self.dir_path = None
            name_file = self.file_path.split("\\")
            name_file = name_file[-1].split("/")
            main_name = name_file[-1]
            self.main_name_file = main_name[:-4]
            self.data = self.load_dataset(self.file_path)
        elif isdir(path):
            self.dir_path = path
            self.file_path = None
            logging.info("Path %s is a directory", path)
        else:
            self.dir_path = None
            self.file_path = None
            logging.error("Unknown path %s!", path)

    def load_dataset(self, file_path, format_type=InputFormat.TERM_FREQUENCY): #term_seq=False, term_freq=True):
        """
        read file input, check format is raw input or term frequency or term sequence
        Args:
            file_path:
            term_seq:
            term_freq:

        Returns:

        """
        format_file = check_input_format(file_path)
        if format_file == 'txt':
            (vocab, file_tf, file_sq) = pre_process(file_path)
            if format_type == InputFormat.TERM_FREQUENCY:
                data_path = file_tf
            elif format_type == InputFormat.TERM_SEQUENCE:
                data_path = file_sq
        elif format_file == InputFormat.TERM_SEQUENCE:
            dir_folder = get_data_home() + self.main_name_file
            # Create folder which include file sq if it doesn't exist
            if os.path.exists(dir_folder):
                shutil.rmtree(dir_folder)
            os.makedirs(dir_folder)
            data_path = dir_folder + '/' + self.main_name_file+'.sq'
            print("Copy file %s => %s" % (file_path, data_path))
            shutil.copyfile(file_path, data_path)
            format_type = InputFormat.TERM_SEQUENCE
        elif format_file == InputFormat.TERM_FREQUENCY:
            dir_folder = get_data_home() + self.main_name_file
            # Create folder which include file sq if it doesn't exist
            if os.path.exists(dir_folder):
                shutil.rmtree(dir_folder)
            os.makedirs(dir_folder)
            data_path = dir_folder + '/' + self.main_name_file+'.tf'
            print("Copy file %s => %s" % (file_path, data_path))
            shutil.copyfile(file_path, data_path)
            format_type = InputFormat.TERM_FREQUENCY
        else:
            print("File %s is not true format!" % file_path)

        bunch = self.Bunch(data_path, format_type)
        return bunch

    class Bunch:
        """
        inner class with methods load data from formatted file
        """
        def __init__(self, data_path, format_type):
            """

            Args:
                data_path:
                format_type
            """
            self.data_path = data_path
            # get directory of this data: .../file_name => ...
            file_name = data_path.split('\\')[-1].split('/')[-1]
            self.folder_data = data_path[:-(len(file_name)+1)]
            self.format_type = format_type
            #self.term_seq = term_seq
            #self.term_freq = term_freq
            # load number of documents
            cnt = 0
            with open(data_path, 'r') as f:
                for cnt, line in enumerate(f):
                    pass
                self.num_doc = cnt + 1
            f.close()

        def shuffle(self):
            """
            shuffle input and write into file file_shuffled.txt, return path of this file
            Returns:

            """
            f = open(self.data_path)
            lines = f.readlines()
            self.num_doc = len(lines)
            np.random.shuffle(lines)
            f.close()
            f_out = open(join(self.folder_data, "file_shuffled.txt"), "w")
            for line in lines:
                f_out.write("%s" % line)
            f_out.close()
            del lines
            return join(self.folder_data, "file_shuffled.txt")

        def load_mini_batch(self, fp, batch_size):
            """

            Args:
                fp:
                batch_size:

            Returns:

            """
            if self.format_type == InputFormat.TERM_FREQUENCY:
                return self.load_mini_batch_term_freq(fp, batch_size)
            elif self.format_type == InputFormat.TERM_SEQUENCE:
                return self.load_mini_batch_term_seq(fp, batch_size)

        def load_mini_batch_term_seq(self, fp, batch_size):
            """
            read mini-batch data and store with format term sequence
            fp is file pointer after shuffled
            Args:
                fp:
                batch_size:

            Returns:

            """
            mini_batch = Corpus(InputFormat.TERM_SEQUENCE)
            for i in range(0, batch_size):
                doc = fp.readline()
                # check end file
                if len(doc) < 1:
                    break
                list_word = doc.strip().split()
                N = int(list_word[0])
                if N + 1 != len(list_word):
                    logging.error("Line %d in file %s is error!", i + 1, self.data_path)
                if self.format_type == InputFormat.TERM_FREQUENCY:
                    tokens = list()
                    for j in range(1, N + 1):
                        tf = list_word[j].split(":")
                        for k in range(0, int(tf[1])):
                            tokens.append(int(tf[0]))
                    mini_batch.word_ids_tks.append(np.array(tokens))
                    mini_batch.cts_lens.append(len(tokens))
                elif self.format_type == InputFormat.TERM_SEQUENCE:
                    doc_t = np.zeros(N, dtype=np.int32)
                    for j in range(1, N + 1):
                        doc_t[j - 1] = int(list_word[j])
                    doc_l = N
                    mini_batch.word_ids_tks.append(doc_t)
                    mini_batch.cts_lens.append(doc_l)
                del list_word
            return mini_batch

        def load_mini_batch_term_freq(self, fp, batch_size):
            """
            read mini-batch data and store with format term frequency
            fp is file pointer after shuffled
            Args:
                fp:
                batch_size:

            Returns:

            """
            mini_batch = Corpus(InputFormat.TERM_FREQUENCY)
            for i in range(0, batch_size):
                doc = fp.readline()
                if len(doc) < 1:
                    break
                list_word = doc.strip().split()
                N = int(list_word[0])
                if N + 1 != len(list_word):
                    print("Line %d in file %s is error!" % (i + 1, self.data_path))
                    sys.exit()
                if self.format_type == InputFormat.TERM_FREQUENCY:
                    doc_t = np.zeros(N, dtype=np.int32)
                    doc_f = np.zeros(N, dtype=np.int32)
                    for j in range(1, N + 1):
                        tf = list_word[j].split(":")
                        doc_t[j - 1] = int(tf[0])
                        doc_f[j - 1] = int(tf[1])
                    mini_batch.word_ids_tks.append(doc_t)
                    mini_batch.cts_lens.append(doc_f)
                elif self.format_type == InputFormat.TERM_SEQUENCE:
                    terms = []
                    freqs = []
                    k = 0
                    for j in range(1, N + 1):
                        if int(list_word[j]) not in terms:
                            terms.append(int(list_word[j]))
                            freqs.append(1)
                        else:
                            index = terms.index(int(list_word[j]))
                            freqs[index] += 1
                    mini_batch.word_ids_tks.append(np.array(terms))
                    mini_batch.cts_lens.append(np.array(freqs))
                del list_word
            return mini_batch


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
    print(reformat_file_to_term_frequency('/home/ubuntu/tmlib_data/ap/ap.sq'))
