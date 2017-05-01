import sys, os, shutil

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

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
from lib.datasets.dataset import DataSet
from lib.lda.ldamodel import LdaModel

#dir_path = os.path.dirname(os.path.realpath(__file__))
#train_file = dir_path + '/ap/ap_train_raw.txt'

def learning(method_name, train_file='data/ap_train_raw.txt', vocab_file=None):

    data = DataSet(train_file, 100, passes=10, shuffle_every=2, vocab_file=vocab_file)
    #f = open(data.vocab_file)
    #lines = f.readlines()
    #num_terms = len(lines)
    #del lines
    # Check method and run algorithm
    methods = ['ml-fw', 'ml-ope', 'online-fw', 'online-ope', 'streaming-fw', 'streaming-ope',
               'ml-cgs', 'online-cgs', 'online-cvb0', 'online-vb', 'streaming-vb']
    method_low = method_name.lower()
    #lda_model = LdaModel(num_terms, 100)
    num_terms = data.get_num_terms()
    #lda_model.load('model/model_batch19.npy')
    #print lda_model.model.shape
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
    elif method_low == 'online-cvb0':
        num_tokens = data.get_num_tokens()
        object = OnlineCVB0(num_tokens, num_terms)
    elif method_low == 'online-vb':
        object = OnlineVB(num_terms, num_topics=20, alpha=0.05, eta=0.05)
    elif method_low == 'streaming-vb':
        object = StreamingVB(num_terms)
    else:
        print '\ninput wrong method name: %s\n' % (method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s' % (method)
        exit()

    model = object.learn_model(data, save_model_every=5, compute_sparsity_every=5,
                               save_statistic=True, save_top_words_every=5, num_top_words=10, model_folder='model')
    model.save('model/lambda_final.txt', file_type='text')
    model.normalize()
    model.print_top_words(10, data.vocab_file, result_file='model/top_words_final.txt')


if __name__ == '__main__':
    learning('online-ope')
