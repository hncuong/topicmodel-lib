import sys, os, shutil

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from tmlib import OnlineVB
from tmlib import MLCGS
from tmlib import MLFW
from tmlib import MLOPE
from tmlib import OnlineCGS
from tmlib import OnlineCVB0
from tmlib import OnlineFW
from tmlib import OnlineOPE
from tmlib import StreamingFW
from tmlib import StreamingOPE
from tmlib import StreamingVB
from tmlib.datasets import DataSet
from tmlib import LdaModel

#dir_path = os.path.dirname(os.path.realpath(__file__))
#train_file = dir_path + '/ap/ap_train_raw.txt'

def learning(method_name, train_file='data/ap_train_raw.txt', vocab_file=None):

    data = DataSet(train_file, 100, passes=10, shuffle_every=2, vocab_file=vocab_file)
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
    model = object.learn_model()
    model.save(os.path.join(model_folder_name, 'beta_final.txt'), file_type='text')
    model.print_top_words(10, data.vocab_file, result_file=os.path.join(model_folder_name,'top_words_final.txt'))

if __name__ == '__main__':
    learning('online-ope')
    learning('online-vb')
    learning('online-fw')
    learning('online-cvb0')
    learning('online-cgs')
    learning('streaming-vb')
    learning('streaming-fw')
    learning('streaming-ope')
    learning('ml-cgs')
    learning('ml-fw')
    learning('ml-ope')
