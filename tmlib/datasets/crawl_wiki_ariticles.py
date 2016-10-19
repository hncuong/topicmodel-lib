"""Reference: M.Hoffman - onlineldavb"""

import sys, urllib2, re, string, time, threading
import os

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


def parse_doc_list(fp, docs, vocab):
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

    wordids = list()
    wordcts = list()
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
        wordids = ddict.keys()
        wordcts = ddict.values()
        fp.write('%d ' % len(wordids))
        for i in xrange(len(wordids)):
            fp.write('%d:%d ' % (wordids[i], wordcts[i]))
        if d < D - 1:
            fp.write('\n')

    del wordids
    del wordcts


def crawl(size_batch, num_crawling):
    path_vocab = dir_path + "/data/wikipedia/vocab.txt"
    if (os.path.isfile(path_vocab)):
        f = open(path_vocab, 'r')
        l_vocab = f.readlines()
        f.close()
    else:
        print('Unknown file %s' % path_vocab)
        exit()
    d_vocab = dict()
    for word in l_vocab:
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        d_vocab[word] = len(d_vocab)
    del l_vocab

    fdata = open(os.path.join(dir_path + "/data/wikipedia", "articles.txt"), 'w')
    fformat = open(os.path.join(dir_path + "/data/wikipedia", "input.txt"), 'w')

    for i in xrange(num_crawling):
        (docset, articlenames) = get_random_wikipedia_articles(size_batch)
        N = len(docset)
        for j in range(0, N):
            fdata.write("<DOC>\n")
            fdata.write("<TITLE> %s <\\TITLE>\n" % articlenames[j])
            fdata.write("<TEXT>\n")
            fdata.write("%s\n" % docset[j])
            fdata.write("<\DOC>\n")
            fdata.write("<\TEXT>\n")
        parse_doc_list(fformat, docset, d_vocab)
        if i < num_crawling - 1:
            fformat.write('\n')
    fdata.close()
    fformat.close()


if __name__ == '__main__':
    t0 = time.time()

    (articles, articlenames) = get_random_wikipedia_articles(4)
    for i in range(0, len(articles)):
        print articlenames[i]

    t1 = time.time()
    print 'took %f' % (t1 - t0)
