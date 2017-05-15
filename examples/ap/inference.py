import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from lib.datasets import utilizies
from lib.lda.Online_VB import OnlineVB
from lib.lda.ML_CGS import MLCGS
from lib.lda.ML_FW import MLFW
from lib.lda.ML_OPE import MLOPE
from lib.lda.Online_CGS import OnlineCGS
from lib.lda.Online_CVB0 import OnlineCVB0
from lib.lda.Online_FW import OnlineFW
from lib.lda.Online_OPE import OnlineOPE
from lib.lda.Streaming_FW import StreamingFW
from lib.lda.Streaming_OPE import StreamingOPE
from lib.lda.Streaming_VB import StreamingVB
from lib.lda.ldamodel import LdaModel
from lib.datasets.utilizies import get_data_home
import logging

#dir_path = os.path.dirname(os.path.realpath(__file__))
#new_file_path = dir_path + '/ap/ap_infer_raw.txt'
vocab_file_path = get_data_home() + '/ap_train_raw/vocab.txt'

def inference(method_name, file_beta_lambda, data_path='data/ap_infer_raw.txt', vocab_path=vocab_file_path):
    if not os.path.isfile(data_path):
        logging.error("Can't find file input: %s" %data_path)
    if not os.path.isfile(vocab_path):
        logging.error("Can't find file vocabulary %s" %vocab_path)
    if not os.path.isfile(file_beta_lambda):
        logging.error("Can't find file learned model %s" %file_beta_lambda)
    input_format = utilizies.check_input_format(data_path)
    if input_format == utilizies.DataFormat.RAW_TEXT:
        docs = utilizies.load_batch_raw_text(data_path)
        vocab_dict_format = utilizies.read_vocab(vocab_path)
        new_corpus = utilizies.parse_doc_list(docs, vocab_dict_format)
        print len(new_corpus.word_ids_tks)
    else:
        new_corpus = utilizies.load_batch_formatted_from_file(data_path)
        print len(new_corpus.word_ids_tks)

    # learned_model is a object of class LdaModel
    learned_model = LdaModel(0,0)
    beta_lambda = learned_model.load(file_beta_lambda)
    print(learned_model.model.shape)

    methods = ['ml-fw', 'ml-ope', 'online-fw', 'online-ope', 'streaming-fw', 'streaming-ope',
               'ml-cgs', 'online-cgs', 'online-cvb0', 'online-vb', 'streaming-vb']
    method_low = method_name.lower()
    num_terms = len(open(vocab_file_path, 'r').readlines())

    if method_low == 'ml-fw':
        object = MLFW(num_terms, lda_model=learned_model)
    elif method_low == 'ml-ope':
        object = MLOPE(num_terms, lda_model=learned_model)
    elif method_low == 'online-fw':
        object = OnlineFW(num_terms, lda_model=learned_model)
    elif method_low == 'online-ope':
        object = OnlineOPE(num_terms, lda_model=learned_model)
    elif method_low == 'streaming-fw':
        object = StreamingFW(num_terms, lda_model=learned_model)
    elif method_low == 'streaming-ope':
        object = StreamingOPE(num_terms, lda_model=learned_model)
    elif method_low == 'ml-cgs':
        object = MLCGS(num_terms, lda_model=learned_model)
    elif method_low == 'online-cgs':
        object = OnlineCGS(num_terms, lda_model=learned_model)
    elif method_low == 'online-cvb0':
        sq_corpus = utilizies.convert_corpus_format(new_corpus, utilizies.DataFormat.TERM_SEQUENCE)
        num_tokens = 0
        for N in sq_corpus.cts_lens:
            num_tokens += N
        object = OnlineCVB0(num_tokens, num_terms, lda_model=learned_model)
    elif method_low == 'online-vb':
        object = OnlineVB(num_terms, num_topics=20, alpha=0.05, eta=0.05, lda_model=learned_model)
    elif method_low == 'streaming-vb':
        object = StreamingVB(num_terms, lda_model=learned_model)
    else:
        print '\ninput wrong method name: %s\n' % (method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s' % (method)
        exit()
    theta = object.infer_new_docs(new_corpus)
    utilizies.write_topic_mixtures(theta, 'model/topic_mixtures.txt')

if __name__ == '__main__':
    inference('online-ope', 'model/lambda_final.txt')
