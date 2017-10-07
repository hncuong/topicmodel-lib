import logging
import pandas as pd

from tmlib.datasets.utilizies import DataFormat

class DataBase(object):
    def __init__(self, database_path, subset=10000):

        self.reader = pd.HDFStore(database_path, 'a')
        if '/corpus' in self.reader:
            self.reader.remove('corpus')
        self.subset = subset
        try:
            num_docs = self.reader.get_storer('theta').nrows
        except:
            raise ValueError('Can not find topic propotions in database')
        if num_docs < subset:
            self.subset = num_docs

    def store_from_object(self, data):
        try:
            if data.input_format == DataFormat.RAW_TEXT:
                self.store_from_raw_text(data.original_data_path)
            elif data.input_format == DataFormat.TERM_FREQUENCY:
                self.store_from_term_frequency(data.data_path, data.vocab_file)
            else:
                self.store_from_term_sequence(data.data_path, data.vocab_file)
        except:
            raise ValueError('Can not identify object')

    def store_from_raw_text(self, data_path):

        fp = open(data_path, 'r')
        max_itemsize = 0
        for i in range(self.subset):
            line = fp.readline().strip()
            while True:
                if line == '<TEXT>':
                    next_line = fp.readline().strip()
                    if len(next_line) > 30:
                        l = next_line
                    else:
                        l = fp.readline().strip()
                    doc = ''
                    while l != '</TEXT>' and len(l) > 10:
                        doc = doc + ' ' + l
                        l = fp.readline().strip()
                    if max_itemsize < len(doc):
                        max_itemsize = len(doc)
                line = fp.readline().strip()
                if line == '</DOC>':
                    break

        fp.seek(0, 0)
        chunk = list()
        start = 0
        i = 0
        while i < self.subset:
            line = fp.readline().strip()
            title = ''
            doc = ''
            while True:
                #print 'hello'
                if '<TITLE>' in line:
                    title = line.replace('<TITLE>', '')
                    title = title.replace('</TITLE>', '')
                if line == '<TEXT>':
                    next_line = fp.readline().strip()
                    if len(next_line) > 10:
                        l = next_line
                    else:
                        l = fp.readline().strip()

                    while l != '</TEXT>' and len(l) > 10:
                        doc = doc + ' ' + l
                        l = fp.readline().strip()
                    if max_itemsize < len(doc):
                        max_itemsize = len(doc)
                    if len(title) == 0:
                        title = doc[:50] + '...'
                line = fp.readline().strip()
                #print line
                if line == '</DOC>':
                    break
            chunk.append([title, doc])
            #print title
                
            if len(chunk) == 1000 or i == self.subset - 1:
                chunk = pd.DataFrame(chunk, columns=['title', 'content'], index=list(range(start, i + 1)))
                self.reader.append('corpus', chunk, min_itemsize=max_itemsize,
                                   complevel=9, complib='blosc')
                start = i + 1
                del chunk
                chunk = list()
            i += 1
        fp.close()

    def store_from_term_frequency(self, data_path, vocab_file):

        logging.info('Get min_itemsize...')
        max_length_doc = 0
        fp = open(data_path)
        for i in range(self.subset):
            line = fp.readline().strip().split()
            if max_length_doc < int(line[0]):
                max_length_doc = int(line[0])
        fp.close()
        fp = open(vocab_file)
        line = fp.readline().strip()
        max_length_term = 0
        vocab = list()
        while line:
            term = line.split()[-1]
            vocab.append(term)
            if len(term) > max_length_term:
                max_length_term = len(term)
            line = fp.readline().strip()
        fp.close()

        min_itemsize = max_length_doc * max_length_term

        logging.info('Storing content...')
        chunk = list()
        fp = open(data_path)
        start = 0
        for i in range(self.subset):
            arr = fp.readline().strip().split()
            N = int(arr[0])
            content = ''
            title = '{'
            for j in range(1, N + 1):
                id_term = int(arr[j].split(':')[0])
                if j <= 3:
                    title = title + vocab[id_term] + ', '
                content = content + vocab[id_term] + ', '
            chunk.append([title[:-1] + '...}', '{' + content[:-2] + '}'])
            if (len(chunk) == 5000) or (i == self.subset - 1):
                chunk = pd.DataFrame(chunk, columns=['title', 'content'], index=list(range(start, i + 1)))
                self.reader.append('corpus', chunk, min_itemsize=min_itemsize,
                                   complevel=9, complib='blosc')
                chunk = list()
                start = i + 1
        fp.close()

    def store_from_term_sequence(self, data_path, vocab_file):
            max_length_doc = 0
            fp = open(data_path)
            for i in range(self.subset):
                line = fp.readline().strip().split()
                list_unique_terms = list(set(line))
                if max_length_doc < len(list_unique_terms):
                    max_length_doc = len(list_unique_terms)
            fp.close()

            fp = open(vocab_file)
            line = fp.readline().strip()
            max_length_term = 0
            vocab = list()
            while line:
                term = line.split()[-1]
                vocab.append(term)
                if len(term) > max_length_term:
                    max_length_term = len(term)
                line = fp.readline().strip()
            fp.close()

            min_itemsize = max_length_doc * max_length_term

            chunk = list()
            fp = open(data_path)
            start = 0
            for i in range(self.subset):
                id_terms = fp.readline().strip().split()
                id_terms = list(set(id_terms))
                content = ''
                title = '{'
                j = 0
                for id in id_terms:
                    if j < 3:
                        title = title + vocab[int(id_terms[j])] + ', '
                    content = content + vocab[int(id)] + ', '
                    j += 1

                chunk.append([title[:-1]+'...}', '{' + content[:-2] + '}'])
                if (len(chunk) == 5000) or (i == self.subset - 1):
                    chunk = pd.DataFrame(chunk, columns=['title', 'content'], index=list(range(start, i + 1)))
                    self.reader.append('corpus', chunk, min_itemsize=min_itemsize,
                                       complevel=9, complib='blosc')
                    chunk = list()
                    start = i + 1
            fp.close()
