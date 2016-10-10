import os, os.path
from os.path import isfile, join, isdir
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

dir_path = os.path.dirname(os.path.realpath(__file__))
tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

class PreProcessing:
    def __init__(self, stemmed=False, remove_rare_word=True):
        self.list_doc = list()
        self.vocab = list()
        self.list_doc_freq = list()
        self.stemmed = stemmed
        self.remove_rare_word = remove_rare_word
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
                        #list_word.append(word)
                    else:
                        self.vocab.append(word)
                        self.df.append([self.id_doc])
                    list_word.append(word)
        self.id_doc += 1
        return list_word

    def filter(self):
        if self.vocab:
            V = len(self.vocab)
            remove_list = []
            i = 0
            while i < V:
                #print(V)
                freq = len(self.df[i])
                #print(freq)
                if freq <= 5 or freq > int(0.5*self.id_doc):
                    #remove_list.append(i)
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

    def process(self, path):
        name_file = path.split("\\")
        name_file = name_file[-1].split("/")
        name = name_file[-1].split(".")
        self.filename = name[0]
        #print(self.filename)
        fin = open(dir_path+"/stop_word_list.txt")
        stop_list = list()
        line = fin.readline()
        while line:
            line = line.strip()
            stop_list.append(line)
            line = fin.readline()
        fin.close()
        stop_list = stop_list + ['_',]
        print("Waiting...")
        if isfile(path):
            fread = open(path)
            line = fread.readline()
            num = 1
            while line:
                line = line.strip()
                if line == "<TEXT>":
                    doc = fread.readline().strip()
                    #print(num)
                    list_word = self.pro_per_doc(doc,stop_list)
                    self.list_doc.append(list_word)
                    num += 1
                line = fread.readline()
        if self.remove_rare_word:
            self.filter()

        numDocs = len(self.list_doc)
        for d in range(0,numDocs):
            list_word = []
            numWords = len(self.list_doc[d])
            for w in range(0,numWords):
                word = self.list_doc[d][w]
                self.list_doc[d][w] = self.vocab.index(word)
                inlist = False
                for elem in list_word:
                    if self.list_doc[d][w] == elem[0]:
                        elem[1] += 1
                        inlist = True
                        break
                if not inlist:
                    list_word.append([self.list_doc[d][w],1])
            self.list_doc_freq.append(list_word)

    def extract_vocab(self):
        if self.vocab:
            self.dir_path_data = dir_path[:-13] + "datasets/data/" + self.filename
            if not os.path.exists(self.dir_path_data):
                os.makedirs(self.dir_path_data)
            fout = open(join(self.dir_path_data,"vocab.txt"),"w")
            for word in self.vocab:
                fout.write("%s\n" %word)
            fout.close()

    def format_seq(self):
        if self.list_doc:
            fout = open(join(self.dir_path_data,"term_sequence.txt"), "w")
            for doc in self.list_doc:
                fout.write("%d " %len(doc))
                for word in doc:
                    fout.write("%d " %word)
                fout.write("\n")
            fout.close()

    def format_freq(self):
        if self.list_doc:
            fout = open(join(self.dir_path_data,"term_frequency.txt"), "w")
            for doc in self.list_doc_freq:
                fout.write("%d " %len(doc))
                for elem in doc:
                    fout.write("%d:%d " %(elem[0],elem[1]))
                fout.write("\n")
            fout.close()

if __name__ == '__main__':
    p = PreProcessing()
    p.process("D:\\UCR_TS_Archive_2015\\ap/ap.txt")
    p.extract_vocab()
    p.format_freq()
    p.format_seq()