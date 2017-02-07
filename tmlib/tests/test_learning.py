import sys, os, shutil

"""mypath = ['../lda/ML_FW', '../lda/ML_OPE', '../lda/Online_FW', '../lda/Online_OPE', '../lda/Streaming_FW',
          '../lda/Streaming_OPE', '../lda/ML_CGS', '../lda/Online_CGS', '../lda/Online_CVB0',
          '../lda', '../lda/Streaming_VB']
for temp in mypath:
    sys.path.insert(0, temp)"""
sys.path.insert(0, '../lda')
from Online_VB import OnlineVB
from ML_CGS import MLCGS
from ML_FW import MLFW
from ML_OPE import MLOPE
from Online_CGS import OnlineCGS
from Online_CVB0 import OnlineCVB0
from Online_FW import OnlineFW
from Online_OPE import OnlineOPE
from Streaming_FW import StreamingFW
from Streaming_OPE import StreamingOPE
from Streaming_VB import StreamingVB
from tmlib.datasets.dataset import DataSet
from tmlib.datasets.wiki_stream import WikiStream
from ldamodel import LdaModel

def main():
    # Check input
    if len(sys.argv) != 5:
        print"usage: python run.py [method name] [train file] [model folder] [vocab file]"
        exit()
    # Get environment variables
    method_name = sys.argv[1]
    train_file = sys.argv[2]
    model_folder = sys.argv[3]
    vocab_file = sys.argv[4]

    if vocab_file == "none":
        vocab_file = None
    data = WikiStream(10, 101)
    #f = open(data.vocab_file)
    #lines = f.readlines()
    #num_terms = len(lines)
    #del lines
    # Check method and run algorithm
    methods = ['ml-fw', 'ml-ope', 'online-fw', 'online-ope', 'streaming-fw', 'streaming-ope'
                                                                             'ml-cgs', 'online-cgs', 'online-cvb0',
               'online-vb', 'streaming-vb']
    method_low = method_name.lower()
    #lda_model = LdaModel(num_terms, 100)
    print data.get_num_terms()
    #lda_model.load('model/model_batch19.npy')
    #print lda_model.model.shape
    if method_low == 'ml-fw':
        object = MLFW(data.get_num_terms())
    elif method_low == 'ml-ope':
        object = MLOPE(data.get_num_terms())
    elif method_low == 'online-fw':
        object = OnlineFW(data.get_num_terms())
    elif method_low == 'online-ope':
        object = OnlineOPE(data.get_num_terms())
    elif method_low == 'streaming-fw':
        object = StreamingFW(data.get_num_terms())
    elif method_low == 'streaming-ope':
        object = StreamingOPE(data.get_num_terms())
    elif method_low == 'ml-cgs':
        object = MLCGS(data.get_num_terms())
    elif method_low == 'online-cgs':
        object = OnlineCGS(data.get_num_terms())
    elif method_low == 'online-cvb0':
        num_tokens = data.get_num_tokens()
        object = OnlineCVB0(num_tokens, num_terms)
    elif method_low == 'online-vb':
        object = OnlineVB(data.get_num_terms())
    elif method_low == 'streaming-vb':
        object = StreamingVB(data.get_num_terms())
    else:
        print '\ninput wrong method name: %s\n' % (method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s' % (method)
        exit()

    if model_folder == "none":
        model = object.learn_model(data, save_model_every=5, compute_sparsity_every=5,
                                   save_statistic=True, save_top_words_every=5, num_top_words=10, model_folder='model')
    else:
        model = object.learn_model(data, model_folder)

    model.save('model/lambda_final.txt', file_type='text')
    model.normalize()
    #model.print_top_words(10, data.vocab_file)


if __name__ == '__main__':
    main()
