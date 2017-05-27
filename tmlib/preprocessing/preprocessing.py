import os, os.path, sys
from os.path import isfile, join, isdir
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

import logging

dir_path = os.path.dirname(os.path.realpath(__file__))
tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

def get_data_home(data_home=None):
    """Return the path of the tmlib data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'lib_data'
    in the user home folder.
    The '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = "~/tmlib_data"
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


class PreProcessing:
    def __init__(self, file_path, stemmed=False, remove_rare_word=3, remove_common_word=0.5):
        self.file_path = file_path
        self.file_name = file_path.split('\\')[-1].split('/')[-1]
        self.main_name_file = self.file_name.split('.')[0]
        self.list_doc = list()
        self.vocab = list()
        self.stemmed = stemmed
        self.remove_rare_word = remove_rare_word
        self.remove_common_word = remove_common_word
        self.id_doc = 0
        self.df = list()

    def pro_per_doc(self, doc, stop_list):
        words = tokenizer.tokenize(doc.lower())
        list_word = list()
        for word in words:
            if word not in stop_list:
                if word.isalpha() and len(word) > 2:
                    if self.stemmed:
                        word = p_stemmer.stem(word)
                    if word in self.vocab:
                        index = self.vocab.index(word)
                        if self.id_doc not in self.df[index]:
                            self.df[index].append(self.id_doc)
                            # list_word.append(word)
                    else:
                        self.vocab.append(word)
                        self.df.append([self.id_doc])
                    list_word.append(word)
        self.id_doc += 1
        return list_word

    def filter(self, rare_word, common_word):
        if self.vocab:
            V = len(self.vocab)
            remove_list = []
            i = 0
            while i < V:
                # print(V)
                freq = len(self.df[i])
                # print(freq)
                if freq <= rare_word or freq > common_word:
                    # remove_list.append(i)
                    docs = self.df[i]
                    word = self.vocab[i]
                    for j in docs:
                        while word in self.list_doc[j]:
                            self.list_doc[j].remove(word)
                    del self.vocab[i]
                    del self.df[i]
                    V = len(self.vocab)
                    i = i - 1
                i += 1
        else:
            print("Vocabulary is empty! Please run process first!")

    def process(self):
        # print(self.filename)
        fin = open(dir_path + "/stop_word_list.txt")
        stop_list = list()
        line = fin.readline()
        while line:
            line = line.strip()
            stop_list.append(line)
            line = fin.readline()
        fin.close()
        stop_list = stop_list + ['_', ]
        print("Waiting...")
        if isfile(self.file_path):
            #print 'hello'
            fread = open(self.file_path)
            line = fread.readline()
            num = 1
            while line:
                line = line.strip()
                if line == "<TEXT>":
                    print(num)
                    next_line = fread.readline().strip()
                    if len(next_line) > 10:
                        l = next_line
                    else:
                        l = fread.readline().strip()
                    doc = ''
                    while l != '</TEXT>' and len(l) > 10:
                        doc = doc + ' ' + l
                        l = fread.readline().strip()
                    num += 1
                    list_word = self.pro_per_doc(doc, stop_list)
                    self.list_doc.append(list_word)
                line = fread.readline()
        else:
            logging.error('Unknown file data %s' % self.file_path)
            exit()

        if self.remove_common_word <= 0:
            self.remove_common_word = int(self.id_doc * 0.5)
        elif self.remove_common_word >= 1:
            self.remove_common_word = self.id_doc
        else:
            self.remove_common_word = int(self.id_doc * self.remove_common_word)
        self.filter(self.remove_rare_word, self.remove_common_word)

        self.num_docs = len(self.list_doc)
        for d in range(0, self.num_docs):
            numWords = len(self.list_doc[d])
            for w in range(0, numWords):
                word = self.list_doc[d][w]
                self.list_doc[d][w] = self.vocab.index(word)

    def extract_vocab(self, folder=None):
        self.num_terms = len(self.vocab)
        if folder is None:
            folder = get_data_home() + '/' + self.main_name_file
        if not os.path.exists(folder):
            os.makedirs(folder)
        if isdir(folder):
            if self.vocab:
                fout = open(join(folder, "vocab.txt"), "w")
                self.path_file_vocab = folder + '/vocab.txt'
                for word in self.vocab:
                    fout.write("%s\n" % word)
                fout.close()
                del self.vocab
            else:
                logging.error("Can't create vocabulary. Please check again!")
                exit()
        else:
            logging.error('Unknown folder data %s' %folder)
            exit()

    def save_format_sq(self, folder=None):
        if folder is None:
            folder = get_data_home() + '/' + self.main_name_file
        if not os.path.exists(folder):
            os.makedirs(folder)
        if isdir(folder):
            if self.list_doc:
                fout = open(join(folder, self.main_name_file+".sq"), "w")
                self.path_file_sq = folder + '/' + self.main_name_file+".sq"
                for doc in self.list_doc:
                    #fout.write("%d " % len(doc))
                    for word in doc:
                        fout.write("%d " % word)
                    fout.write("\n")
                fout.close()
        else:
            logging.error('Unknown folder data %s' %folder)
            exit()

    def save_format_tf(self, folder=None):
        if folder is None:
            folder = get_data_home() + '/' + self.main_name_file
        if not os.path.exists(folder):
            os.makedirs(folder)
        if isdir(folder):
            fout = open(join(folder, self.main_name_file + ".tf"), "w")
            self.path_file_tf = folder + '/' + self.main_name_file + ".tf"
            for d in range(0, self.num_docs):
                list_word = []
                numWords = len(self.list_doc[d])
                for w in range(0, numWords):
                    inlist = False
                    for elem in list_word:
                        if self.list_doc[d][w] == elem[0]:
                            elem[1] += 1
                            inlist = True
                            break
                    if not inlist:
                        list_word.append([self.list_doc[d][w], 1])
                fout.write("%d " %len(list_word))
                for elem in list_word:
                    fout.write("%d:%d " %(elem[0], elem[1]))
                fout.write("\n")
            fout.close()
            #del self.list_doc
        else:
            logging.error('Unknown folder data %s' %folder)
            exit()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python preprocessing.py [path file]')
        exit()
    pathfile = sys.argv[1]
    if isfile(pathfile):
        p = PreProcessing(pathfile)
        p.process()
        p.extract_vocab('~/Desktop/ap')
        p.save_format_tf('/home/ubuntu/Desktop/ap')
        p.save_format_sq('/home/ubuntu/Desktop/ap')
    else:
        print('File not found!')
        exit()
