import os, logging

from lib.datasets.wiki_stream import WikiStream
from lib.datasets import base
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

def learn(method_name):
    data = WikiStream(64, 100)
    methods = ['ml-fw', 'ml-ope', 'online-fw', 'online-ope', 'streaming-fw', 'streaming-ope',
               'ml-cgs', 'online-cgs', 'online-vb', 'streaming-vb']

    method_low = method_name.lower()
    num_terms = data.get_num_terms()

    if method_low == 'ml-fw':
        object = MLFW(num_terms)
    elif method_low == 'ml-ope':
        object = MLOPE(num_terms)
    elif method_low == 'online-fw':
        object = OnlineFW(num_terms)
    elif method_low == 'online-ope':
        object = OnlineOPE(num_terms)
    elif method_low == 'streaming-fw':
        object = StreamingFW(num_terms)
    elif method_low == 'streaming-ope':
        object = StreamingOPE(num_terms)
    elif method_low == 'ml-cgs':
        object = MLCGS(num_terms)
    elif method_low == 'online-cgs':
        object = OnlineCGS(num_terms)
    elif method_low == 'online-vb':
        object = OnlineVB(num_terms)
    elif method_low == 'streaming-vb':
        object = StreamingVB(num_terms)
    else:
        print '\ninput wrong method name: %s\n' % (method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s' % (method)
        exit()

    model = object.learn_model(data, save_model_every=5, compute_sparsity_every=5,
                           save_statistic=True, save_top_words_every=5, num_top_words=10, model_folder='model_stream_vb')
    model.save('model_stream_vb/lambda_final.txt', file_type='text')
    model.normalize()
    model.print_top_words(10, data.vocab_file, result_file='model_stream_vb/top_words_final.txt')

def infer(method_name, file_beta_lambda='model_stream_vb/lambda_final.txt'):

    if not os.path.isfile(file_beta_lambda):
        logging.error("Can't find file learned model %s" % file_beta_lambda)

    data = WikiStream(10, 10)
    new_corpus = data.load_mini_batch()

    # learned_model is a object of class LdaModel
    learned_model = LdaModel(0, 0)
    beta_lambda = learned_model.load(file_beta_lambda)
    print(learned_model.model.shape)

    methods = ['ml-fw', 'ml-ope', 'online-fw', 'online-ope', 'streaming-fw', 'streaming-ope',
               'ml-cgs', 'online-cgs', 'online-vb', 'streaming-vb']
    method_low = method_name.lower()
    num_terms = data.get_num_terms()

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
    elif method_low == 'online-vb':
        object = OnlineVB(num_terms, lda_model=learned_model)
    elif method_low == 'streaming-vb':
        object = StreamingVB(num_terms, lda_model=learned_model)
    else:
        print '\ninput wrong method name: %s\n' % (method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s' % (method)
        exit()
    theta = object.infer_new_docs(new_corpus)
    base.write_topic_mixtures(theta, 'model_stream_vb/theta_new_docs.txt')

if __name__ == '__main__':
    learn('streaming-vb')
    infer('streaming-vb')
