import sys, os, shutil

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from tmlib.lda import OnlineVB
from tmlib.lda import MLCGS
from tmlib.lda import MLFW
from tmlib.lda import MLOPE
from tmlib.lda import OnlineCGS
from tmlib.lda import OnlineCVB0
from tmlib.lda import OnlineFW
from tmlib.lda import OnlineOPE
from tmlib.lda import StreamingFW
from tmlib.lda import StreamingOPE
from tmlib.lda import StreamingVB
from tmlib.datasets import DataSet

#dir_path = os.path.dirname(os.path.realpath(__file__))
#train_file = dir_path + '/ap/ap_train_raw.txt'

def learning(method_name, train_file='data/ap_train.txt', vocab_file='data/vocab.txt'):

    data = DataSet(train_file, 100, passes=5, shuffle_every=2, vocab_file=vocab_file)
    # Check method and run algorithm
    methods = ['ml-fw', 'ml-ope', 'online-fw', 'online-ope', 'streaming-fw', 'streaming-ope',
               'ml-cgs', 'online-cgs', 'online-cvb0', 'online-vb', 'streaming-vb']
    method_low = method_name.lower()
    if method_low == 'ml-fw':
        object = MLFW(data)
    elif method_low == 'ml-ope':
        object = MLOPE(data)
    elif method_low == 'online-fw':
        object = OnlineFW(data)
    elif method_low == 'online-ope':
        object = OnlineOPE(data)
    elif method_low == 'streaming-fw':
        object = StreamingFW(data)
    elif method_low == 'streaming-ope':
        object = StreamingOPE(data)
    elif method_low == 'ml-cgs':
        object = MLCGS(data)
    elif method_low == 'online-cgs':
        object = OnlineCGS(data)
    elif method_low == 'online-cvb0':
        object = OnlineCVB0(data)
    elif method_low == 'online-vb':
        object = OnlineVB(data)
    elif method_low == 'streaming-vb':
        object = StreamingVB(data)
    else:
        print '\ninput wrong method name: %s\n' % (method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s' % (method)
        exit()

    model_folder_name = 'model-' + method_low
    if not os.path.exists(model_folder_name):
        os.mkdir(model_folder_name)
    model = object.learn_model()
    model.save(os.path.join(model_folder_name, 'beta_final.txt'), file_type='text')
    model.print_top_words(10, data.vocab_file, result_file=os.path.join(model_folder_name,'top_words_final.txt'))

if __name__ == '__main__':
    try:
        learning('online-ope')
    except:
        exit(0)
    try:
        learning('online-vb')
    except:
        exit(0)
    try:
        learning('online-fw')
    except:
        exit(0)
    try:
        learning('online-cvb0')
    except:
        exit(0)
    try:
        learning('online-cgs')
    except:
        exit(0)
    try:
        learning('streaming-vb')
    except:
        exit(0)
    try:
        learning('streaming-fw')
    except:
        exit(0)
    try:
        learning('streaming-ope')
    except:
        exit(0)
    try:
        learning('ml-cgs')
    except:
        exit(0)
    try:
        learning('ml-fw')
    except:
        exit(0)
    try:
        learning('ml-ope')
    except:
        exit(0)
