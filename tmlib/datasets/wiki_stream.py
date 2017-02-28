"""Reference: M.Hoffman - onlineldavb"""

import sys, os, urllib2, re, time, threading
import logging
import base
from base import Corpus, DataIterator, DataFormat

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
        print 'downloaded %s. parsing...' % articletitle

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

def save(fp, wordids, wordcts):
    D = len(wordids)
    for d in xrange(D):
        fp.write('%d ' % len(wordids[d]))
        for i in xrange(len(wordids[d])):
            fp.write('%d:%d ' % (wordids[d][i], wordcts[d][i]))
        if d < D - 1:
            fp.write('\n')

class WikiStream(DataIterator):
    def __init__(self, batch_size, num_batch, save_into_file=False, vocab_file=None):
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
        if save_into_file:
            folder_data = base.get_data_home() + '/WikiStream'
            if not os.path.exists(folder_data):
                os.mkdir(folder_data)
            self.fp = open(os.path.join(folder_data, 'articles.tf'), "w")
        self.save_into_file = save_into_file
        self.output_format = DataFormat.TERM_FREQUENCY
        if vocab_file is None:
            self.vocab_file = dir_path + "/data/wikipedia/vocab.txt"
        else:
            self.vocab_file = vocab_file
        self.vocab = base.read_vocab(self.vocab_file)

    def load_mini_batch(self):
        (docset, articlenames) = get_random_wikipedia_articles(self.batch_size)
        logging.info("Mini batch no: %s", self.mini_batch_no)
        mini_batch = base.parse_doc_list(docset, self.vocab)
        if self.output_format == DataFormat.TERM_FREQUENCY:
            mini_batch = base.convert_corpus_format(mini_batch, DataFormat.TERM_FREQUENCY)
            if self.save_into_file:
                save(self.fp, mini_batch.word_ids_tks, mini_batch.cts_lens)
        else:
            mini_batch = base.convert_corpus_format(mini_batch, DataFormat.TERM_SEQUENCE)
        self.mini_batch_no += 1
        return mini_batch

    def check_end_of_data(self):
        if self.mini_batch_no == self.num_batch:
            self.end_of_data = True
        return self.end_of_data

    def set_output_format(self, output_format):
        assert (output_format == DataFormat.TERM_SEQUENCE or output_format == DataFormat.TERM_FREQUENCY), \
            'Corpus format type must be term-frequency (tf) or sequences (sq)!'
        self.output_format = output_format

    def get_num_terms(self):
        if self.vocab_file is None:
            self.vocab_file = dir_path + "/data/wikipedia/vocab.txt"
        f = open(self.vocab_file , 'r')
        list_terms = f.readlines()
        return len(list_terms)

    def get_total_docs(self):
	# The total number of documents in Wikipedia, refer Hoffman source code: https://github.com/blei-lab/onlineldavb/blob/master/onlinewikipedia.py
    	return 3.3e6


if __name__ == '__main__':
    wiki = WikiStream(8,3, save_into_file=True)
    end = wiki.end_of_num_batch()
    while not end:
        wiki.load_mini_batch()
        end = wiki.end_of_num_batch()
