import os, logging

from wiki_stream import WikiStream
from tmlib.datasets import utilizies
from tmlib.lda.Streaming_FW import StreamingFW
from tmlib.lda.Streaming_OPE import StreamingOPE
from tmlib.lda.Streaming_VB import StreamingVB
from tmlib.lda.ldamodel import LdaModel

def learn(method_name):
    data = WikiStream(64, 100)
    methods = ['ml-fw', 'ml-ope', 'online-fw', 'online-ope', 'streaming-fw', 'streaming-ope',
               'ml-cgs', 'online-cgs', 'online-vb', 'streaming-vb']

    method_low = method_name.lower()

    if method_low == 'streaming-fw':
        object = StreamingFW(data)
    elif method_low == 'streaming-ope':
        object = StreamingOPE(data)
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
    model.save(os.path.join(model_folder_name,'beta_final.txt'), file_type='text')
    model.print_top_words(10, data.vocab_file, result_file=os.path.join(model_folder_name,'beta_final.txt'))

if __name__ == '__main__':
    learn('streaming-vb')
    learn('streaming-fw')
    learn('streaming-ope')
