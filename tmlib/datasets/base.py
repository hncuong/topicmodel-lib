import os, os.path
import sys
import shutil
from os.path import isdir, isfile, join
from ..preprocessing import preprocessing
import numpy as np
from time import time

# Name of current path directory which contains this file
dir_path = os.path.dirname(os.path.realpath(__file__))


# check format of input file(text formatted or raw text)
def check_format(line, c):
    result = True
    l = len(line)
    for i in range(0, l):
        if line[i].isalpha():
            result = False
            break
        elif not line[i].isalnum() and line[i] != ' ' and line[i] != c:
            result = False
            break
    return result


class Dataset:
    def __init__(self, path):
        self.vocab_path = None
        self.is_raw_text = False
        if isfile(path):
            print("Path %s is a file" % path)
            self.path_isfile = path
            self.path_isdir = None
            name_file = self.path_isfile.split("\\")
            name_file = name_file[-1].split("/")
            main_name = name_file[-1]
            self.main_name_file = main_name[:-4]
        elif isdir(path):
            self.path_isdir = path
            self.path_isfile = None
            print("Path %s is a directory" % path)
        else:
            self.path_isdir = None
            self.path_isfile = None
            print("Unknown path %s!" % path)
            exit()

    """
        read file input, check format is raw input or term frequency or term sequence
    """

    def load_dataset(self, term_seq=False, term_freq=True):
        if self.path_isfile:
            f = open(self.path_isfile)
            line = f.readline().strip()
            while len(line) == 0:
                line = f.readline().strip()
            if line == "<DOC>":
                self.is_raw_text = True
                print("Pre-processing:")
                p = preprocessing.PreProcessing()
                p.process(self.path_isfile)
                p.extract_vocab()
                p.format_freq()
                p.format_seq()
                self.vocab_path = p.dir_path_data + "/vocab.txt"
                if term_freq:
                    data_path = p.dir_path_data + "/term_frequency.txt"
                    term_seq = False
                else:
                    data_path = p.dir_path_data + "/term_sequence.txt"
                    term_seq = True
            elif check_format(line, ' '):
                if 'wikipedia' in self.path_isfile:
                    data_path = self.path_isfile
                else:
                    dir_folder = dir_path + "/data/" + self.main_name_file
                    # Create model folder if it doesn't exist
                    if os.path.exists(dir_folder):
                        shutil.rmtree(dir_folder)
                    os.makedirs(dir_folder)
                    data_path = dir_folder + "/term_sequence.txt"
                    print("Copy file %s => %s" % (self.path_isfile, data_path))
                    shutil.copyfile(self.path_isfile, data_path)
                term_freq = False
                term_seq = True
            elif check_format(line, ':'):
                if 'wikipedia' in self.path_isfile:
                    data_path = self.path_isfile
                else:
                    dir_folder = dir_path + "/data/" + self.main_name_file
                    # Create model folder if it doesn't exist
                    if os.path.exists(dir_folder):
                        shutil.rmtree(dir_folder)
                    os.makedirs(dir_folder)
                    data_path = dir_folder + "/term_frequency.txt"
                    print("Copy file %s => %s" % (self.path_isfile, data_path))
                    shutil.copyfile(self.path_isfile, data_path)
                term_seq = False
                term_freq = True
            else:
                print("File %s is not true format!" % self.path_isfile)
                sys.exit()
            f.close()
        bunch = self.Bunch(data_path, term_seq, term_freq)
        return bunch

    """
        inner class with methods load data from formatted file
    """

    class Bunch:
        def __init__(self, data_path, term_seq, term_freq):
            self.data_path = data_path
            self.term_seq = term_seq
            self.term_freq = term_freq
            self.copus = []
            # load number of documents
            f = open(data_path, 'r')
            lines = f.readlines()
            self.num_doc = len(lines)
            del lines
            f.close()

        """
            shuffle input and write into file file_shuffled.txt, return path of this file
        """

        def shuffle(self):
            f = open(self.data_path)
            lines = f.readlines()
            self.num_doc = len(lines)
            np.random.shuffle(lines)
            f.close()
            if self.term_freq:
                d = self.data_path[:-19]
            elif self.term_seq:
                d = self.data_path[:-18]
            fout = open(join(d, "file_shuffled.txt"), "w")
            for line in lines:
                fout.write("%s" % line)
            fout.close()
            del lines
            return join(d, "file_shuffled.txt")

        """
            read mini-batch data and store with format term sequence
            fp is file pointer after shuffled
        """

        def load_minibatch_term_seq(self, fp, size_batch):
            doc_terms = []
            doc_lens = []
            for i in range(0, size_batch):
                doc = fp.readline()
                # check end file
                if len(doc) < 1:
                    break
                list_word = doc.strip().split()
                N = int(list_word[0])
                if N + 1 != len(list_word):
                    print("Line %d in file %s is error!" % (i + 1, self.data_path))
                    sys.exit()
                if self.term_freq:
                    tokens = list()
                    for j in range(1, N + 1):
                        tf = list_word[j].split(":")
                        for k in range(0, int(tf[1])):
                            tokens.append(int(tf[0]))
                    doc_terms.append(np.array(tokens))
                    doc_lens.append(len(tokens))
                elif self.term_seq:
                    doc_t = np.zeros(N, dtype=np.int32)
                    for j in range(1, N + 1):
                        doc_t[j - 1] = int(list_word[j])
                    doc_l = N
                    doc_terms.append(doc_t)
                    doc_lens.append(doc_l)
                del list_word
            return (doc_terms, doc_lens)

        """
            read mini-batch data and store with format term frequency
            fp is file pointer after shuffled
        """

        def load_minibatch_term_freq(self, fp, size_batch):
            doc_terms = []
            doc_freqs = []
            for i in range(0, size_batch):
                doc = fp.readline()
                if len(doc) < 1:
                    break
                list_word = doc.strip().split()
                N = int(list_word[0])
                if N + 1 != len(list_word):
                    print("Line %d in file %s is error!" % (i + 1, self.data_path))
                    sys.exit()
                if self.term_freq:
                    doc_t = np.zeros(N, dtype=np.int32)
                    doc_f = np.zeros(N, dtype=np.int32)
                    for j in range(1, N + 1):
                        tf = list_word[j].split(":")
                        doc_t[j - 1] = int(tf[0])
                        doc_f[j - 1] = int(tf[1])
                    doc_terms.append(doc_t)
                    doc_freqs.append(doc_f)
                elif self.term_seq:
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
                    doc_terms.append(np.array(terms))
                    doc_freqs.append(np.array(freqs))
                del list_word
            return (doc_terms, doc_freqs)


"""------------------------------------------------------------------------------------------------------------------"""

"""def read_setting(file_name):
    if isfile(file_name):
        f = open(file_name, 'r')
        settings = f.readlines()
        f.close()
        sets = list()
        vals = list()
        for i in range(len(settings)):
            # print'%s\n'%(settings[i])
            if settings[i].strip()[0] == '#':
                continue
            set_val = settings[i].strip().split(':')
            sets.append(set_val[0])
            vals.append(float(set_val[1]))
        ddict = dict(zip(sets, vals))
        #ddict['num_terms'] = int(ddict['num_terms'])
        ddict['num_topics'] = int(ddict['num_topics'])
        ddict['iter_train'] = int(ddict['iter_train'])
        ddict['iter_infer'] = int(ddict['iter_infer'])
        ddict['batch_size'] = int(ddict['batch_size'])
        ddict['num_crawling'] = int(ddict['num_crawling'])
        return (ddict)
    else:
        print("Can't find file!")
        sys.exit()"""

"""
    Compute document sparsity.
"""


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
    return (np.mean(sparsity))


"""
    Create list of top words of topics.
"""


def list_top(beta, tops):
    min_float = -sys.float_info.max
    num_tops = beta.shape[0]
    list_tops = list()
    for k in range(num_tops):
        top = list()
        arr = np.array(beta[k, :], copy=True)
        for t in range(tops):
            index = arr.argmax()
            top.append(index)
            arr[index] = min_float
        list_tops.append(top)
    return (list_tops)


"""------------------------------------------------------------------------------------------------------------------"""


def write_setting(ddict, file_name):
    keys = list(ddict.keys())
    vals = list(ddict.values())
    f = open(file_name, 'w')
    for i in range(len(keys)):
        f.write('%s: %f\n' % (keys[i], vals[i]))
    f.close()


def write_topics(beta, file_name):
    num_terms = beta.shape[1]
    num_topics = beta.shape[0]
    f = open(file_name, 'w')
    for k in range(num_topics):
        for i in range(num_terms - 1):
            f.write('%.10f ' % (beta[k][i]))
        f.write('%.10f\n' % (beta[k][num_terms - 1]))
    f.close()


def write_topic_mixtures(theta, file_name):
    batch_size = theta.shape[0]
    num_topics = theta.shape[1]
    f = open(file_name, 'a')
    for d in range(batch_size):
        for k in range(num_topics - 1):
            f.write('%.5f ' % (theta[d][k]))
        f.write('%.5f\n' % (theta[d][num_topics - 1]))
    f.close()


def write_time(i, j, time_e, time_m, file_name):
    f = open(file_name, 'a')
    f.write('tloop_%d_iloop_%d, %f, %f, %f,\n' % (i, j, time_e, time_m, time_e + time_m))
    f.close()


def write_loop(i, j, file_name):
    f = open(file_name, 'w')
    f.write('%d, %d' % (i, j))
    f.close()


def write_file(i, j, beta, time_e, time_m, theta, sparsity, list_tops, tops, model_folder):
    beta_file_name = '%s/beta_%d_%d.dat' % (model_folder, i, j)
    theta_file_name = '%s/theta_%d.dat' % (model_folder, i)
    # per_file_name = '%s/perplexities_%d.csv'%(model_folder, i)
    # top_file_name = '%s/top%d_%d_%d.dat'%(model_folder, tops, i, j)
    # spar_file_name = '%s/sparsity_%d.csv'%(model_folder, i)
    time_file_name = '%s/time_%d.csv' % (model_folder, i)
    loop_file_name = '%s/loops.csv' % (model_folder)

    # write beta
    if j % 10 == 1:
        write_topics(beta, beta_file_name)
    # write theta
    write_topic_mixtures(theta, theta_file_name)

    # write perplexities
    # write_perplexities(LD2, per_file_name)
    # write list top
    ##write_topic_top(list_tops, top_file_name)
    # write sparsity
    ##write_sparsity(sparsity, spar_file_name)
    # write time
    write_time(i, j, time_e, time_m, time_file_name)
    # write loop
    write_loop(i, j, loop_file_name)


if __name__ == '__main__':
    t0 = time()
    data = Dataset("D:\\UCR_TS_Archive_2015\\ap/ap.txt").load_dataset()
    (docs, freqs) = data.load_batch()
    print(len(docs))
    print(docs[0])
    (docs, lengs) = data.load_minibatch(2)
    print(docs)
    print(lengs)
    print("Done in %.3f" % (time() - t0))
