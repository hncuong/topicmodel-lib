"""Reference: M.Hoffman - onlineldavb"""

import sys, os, urllib2, re, string, threading
import logging
from tmlib.datasets import utilizies
from tmlib.datasets.utilizies import Corpus, DataIterator, DataFormat
from tmlib.preprocessing.preprocessing import PreProcessing
import pandas as pd

# Name of current path directory which contains this file
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_random_wikipedia_article():
    """
    Downloads a randomly selected Wikipedia article (via
    http://en.wikipedia.org/wiki/Special:Random) and strips out (most
    of) the formatting, links, etc.

    This function is a bit simpler and less robust than the code that
    was used for the experiments in "Online VB for LDA."
    """
    failed = True
    while failed:
        articletitle = None
        failed = False
        try:
            req = urllib2.Request('http://en.wikipedia.org/wiki/Special:Random',
                                  None, {'User-Agent': 'x'})
            f = urllib2.urlopen(req)
            while not articletitle:
                line = f.readline()
                result = re.search(r'title="Edit this page" href="/w/index.php\?title=(.*)\&amp;action=edit"\/>', line)
                if (result):
                    articletitle = result.group(1)
                    break
                elif (len(line) < 1):
                    sys.exit(1)

            req = urllib2.Request('http://en.wikipedia.org/w/index.php?title=Special:Export/%s&action=submit' \
                                  % (articletitle),
                                  None, {'User-Agent': 'x'})
            f = urllib2.urlopen(req)
            all = f.read()
        except (urllib2.HTTPError, urllib2.URLError):
            print 'oops. there was a failure downloading %s. retrying...' \
                  % articletitle
            failed = True
            continue
        #print 'downloaded %s. parsing...' % articletitle

        try:
            all = re.search(r'<text.*?>(.*)</text', all, flags=re.DOTALL).group(1)
            all = re.sub(r'\n', ' ', all)
            all = re.sub(r'\{\{.*?\}\}', r'', all)
            all = re.sub(r'\[\[Category:.*', '', all)
            all = re.sub(r'==\s*[Ss]ource\s*==.*', '', all)
            all = re.sub(r'==\s*[Rr]eferences\s*==.*', '', all)
            all = re.sub(r'==\s*[Ee]xternal [Ll]inks\s*==.*', '', all)
            all = re.sub(r'==\s*[Ee]xternal [Ll]inks and [Rr]eferences==\s*', '', all)
            all = re.sub(r'==\s*[Ss]ee [Aa]lso\s*==.*', '', all)
            all = re.sub(r'http://[^\s]*', '', all)
            all = re.sub(r'\[\[Image:.*?\]\]', '', all)
            all = re.sub(r'Image:.*?\|', '', all)
            all = re.sub(r'\[\[.*?\|*([^\|]*?)\]\]', r'\1', all)
            all = re.sub(r'\&lt;.*?&gt;', '', all)
        except:
            # Something went wrong, try again. (This is bad coding practice.)
            print 'oops. there was a failure parsing %s. retrying...' \
                  % articletitle
            failed = True
            continue

    return (all, articletitle)


class WikiThread(threading.Thread):
    articles = list()
    articlenames = list()
    lock = threading.Lock()

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        (article, articlename) = get_random_wikipedia_article()
        WikiThread.lock.acquire()
        WikiThread.articles.append(article)
        WikiThread.articlenames.append(articlename)
        WikiThread.lock.release()


def get_random_wikipedia_articles(n):
    """
    Downloads n articles in parallel from Wikipedia and returns lists
    of their names and contents. Much faster than calling
    get_random_wikipedia_article() serially.
    """
    maxthreads = 8
    WikiThread.articles = list()
    WikiThread.articlenames = list()
    wtlist = list()
    for i in range(0, n, maxthreads):
        print 'downloaded %d/%d articles...' % (i, n)
        for j in range(i, min(i + maxthreads, n)):
            wtlist.append(WikiThread())
            wtlist[len(wtlist) - 1].start()
        for j in range(i, min(i + maxthreads, n)):
            wtlist[j].join()

    return (WikiThread.articles, WikiThread.articlenames)

def parse_doc_list(docs, vocab):
    """
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts.

    Arguments:
    docs:  List of D documents. Each document must be represented as
           a single string. (Word order is unimportant.) Any
           words not in the vocabulary will be ignored.
    vocab: Dictionary mapping from words to integer ids.

    Returns a pair of lists of lists.

    The first, wordids, says what vocabulary tokens are present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)

    The second, wordcts, says how many times each vocabulary token is
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)

    #wordids = list()
    #wordcts = list()
    copus = Corpus(DataFormat.TERM_FREQUENCY)
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        ddict = dict()
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word]
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        copus.append_doc(ddict.keys(), ddict.values())

    return copus

def read_vocab(path_vocab):
    if (os.path.isfile(path_vocab)):
        f = open(path_vocab, 'r')
        l_vocab = f.readlines()
        f.close()
    else:
        logging.error('Unknown file %s' % path_vocab)
    d_vocab = dict()
    for word in l_vocab:
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        d_vocab[word] = len(d_vocab)
    del l_vocab
    return d_vocab

def save_articles(articles, articlenames, data_wiki_folder):
    path_file_articles = os.path.join(data_wiki_folder, 'wiki_articles.txt')
    f = open(path_file_articles, 'a')
    for d in range(len(articles)):
        f.write('<DOC>\n')
        f.write('<TITLE>'+articlenames[d]+'</TITLE>\n')
        f.write('<TEXT>\n')
        f.write(articles[d])
        f.write('\n</TEXT>')
        f.write('\n</DOC>\n')
    f.close()
    return path_file_articles

def save_articles_per_batch(articles, articlenames, data_wiki_folder):
    path_file_articles = os.path.join(data_wiki_folder, 'wiki_articles_per_batch.txt')
    f = open(path_file_articles, 'w')
    for d in range(len(articles)):
        f.write('<DOC>\n')
        f.write('<TITLE>'+articlenames[d]+'</TITLE>\n')
        f.write('<TEXT>\n')
        f.write(articles[d])
        f.write('\n</TEXT>')
        f.write('\n</DOC>\n')
    f.close()
    return path_file_articles

def save(fp, wordids, wordcts):
    D = len(wordids)
    for d in xrange(D):
        fp.write('%d ' % len(wordids[d]))
        for i in xrange(len(wordids[d])):
            fp.write('%d:%d ' % (wordids[d][i], wordcts[d][i]))
        if d < D - 1:
            fp.write('\n')

class WikiStream(DataIterator):
    def __init__(self, batch_size, num_batch, vocab_file=None):
        """
        a class for crawl stream data from website wikipedia
        Args:
            batch_size:
            num_batch:
            save_into_file:
            vocab_file:
        """
        super(WikiStream, self).__init__()
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.end_of_file = False
        self.data_wiki_folder = os.path.join(utilizies.get_data_home(), 'WikiStream')
        if not os.path.exists(self.data_wiki_folder):
            os.mkdir(self.data_wiki_folder)
        self.output_format = DataFormat.TERM_FREQUENCY
        if vocab_file is None:
            self.vocab_file = dir_path + "/data/vocab.txt"
        else:
            self.vocab_file = vocab_file
        self.vocab = read_vocab(self.vocab_file)
        self.database_path = None

    def load_mini_batch(self):
        (docset, articlenames) = get_random_wikipedia_articles(self.batch_size)
        path_articles = save_articles_per_batch(docset, articlenames, self.data_wiki_folder)
        save_articles(docset, articlenames, self.data_wiki_folder)
        # add new terms to the vocabulary set
        raw_data = PreProcessing(path_articles, remove_rare_word=2, remove_common_word=0.5)
        raw_data.process()
        old_vocab = list()
        f_vocab = open(self.vocab_file)
        line = f_vocab.readline().strip()
        while line:
            old_vocab.append(line)
            line = f_vocab.readline().strip()
        f_vocab.close()
        new_vocab = set(raw_data.vocab)
        in_new_but_not_in_old = new_vocab - set(old_vocab)
        result_vocab = old_vocab + list(in_new_but_not_in_old)
        self.vocab_file = self.data_wiki_folder + '/current_vocab.txt'
        f_new_vocab = open(self.vocab_file, 'w')
        for term in result_vocab:
            f_new_vocab.write(term+'\n')
        f_new_vocab.close()
        # create corpus to store mini-batch
        dict_vocab = read_vocab(self.vocab_file)
        corpus = Corpus(DataFormat.TERM_SEQUENCE)
        for doc in raw_data.list_doc:
            for i in range(len(doc)):
                doc[i] = dict_vocab[raw_data.vocab[doc[i]]]
            if len(doc) > 0:
                corpus.append_doc(doc, len(doc))

        logging.info("Mini batch no: %s", self.mini_batch_no)
        if self.output_format == DataFormat.TERM_FREQUENCY:
            mini_batch = utilizies.convert_corpus_format(corpus, DataFormat.TERM_FREQUENCY)
        else:
            mini_batch = corpus
        self.mini_batch_no += 1
        return mini_batch

    def check_end_of_data(self):
        if self.mini_batch_no == self.num_batch:
            self.end_of_data = True
            if self.database_path is not None:
                if os.path.exists(self.database_path):
                    self.database.close()
        return self.end_of_data

    def set_output_format(self, output_format):
        assert (output_format == DataFormat.TERM_SEQUENCE or output_format == DataFormat.TERM_FREQUENCY), \
            'Corpus format type must be term-frequency (tf) or sequences (sq)!'
        self.output_format = output_format

    def get_num_terms(self):
        if self.vocab_file is None:
            self.vocab_file = dir_path + "/data/vocab.txt"
        f = open(self.vocab_file, 'r')
        list_terms = f.readlines()
        return len(list_terms)

    def init_database(self, database_path):
        self.database_path = database_path
        self.database = pd.HDFStore(database_path, 'w')
        self.end_index = 0

    def store_topic_proportions(self, theta):
        #self.table_name_topic_propotions = table_name
        #if self.pass_no == self.passes:
        dist_topics = ['dist_topic' + str(i) for i in range(theta.shape[1])]
        start = self.end_index + 1
        end = start+theta.shape[0]
        self.end_index = end - 1
        theta_frame = pd.DataFrame(theta, columns=dist_topics, index=list(range(start, end)))
        self.database.append('theta', theta_frame, data_columns=True, complevel=9, complib='blosc')

if __name__ == '__main__':
    wiki = WikiStream(8,3)
    end = wiki.check_end_of_data()
    while not end:
        wiki.load_mini_batch()
        end = wiki.check_end_of_data()
