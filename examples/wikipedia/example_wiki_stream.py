import os, logging

from wiki_stream import WikiStream
from lib.datasets import utilizies
from lib.lda.Streaming_FW import StreamingFW
from lib.lda.Streaming_OPE import StreamingOPE
from lib.lda.Streaming_VB import StreamingVB
from lib.lda.ldamodel import LdaModel

def learn(method_name):
    data = WikiStream(32, 2)
    methods = ['ml-fw', 'ml-ope', 'online-fw', 'online-ope', 'streaming-fw', 'streaming-ope',
               'ml-cgs', 'online-cgs', 'online-vb', 'streaming-vb']

    method_low = method_name.lower()
    num_terms = data.get_num_terms()

    if method_low == 'streaming-fw':
        object = StreamingFW(num_terms)
    elif method_low == 'streaming-ope':
        object = StreamingOPE(num_terms)
    elif method_low == 'streaming-vb':
        object = StreamingVB(num_terms)
    else:
        print '\ninput wrong method name: %s\n' % (method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s' % (method)
        exit()
    model_folder = 'model_' + method_low
    model = object.learn_model(data, save_model_every=5, compute_sparsity_every=5,
                           save_statistic=True, save_top_words_every=5, num_top_words=10, model_folder=model_folder)
    model.save(model_folder+'/lambda_final.txt', file_type='text')
    model.normalize()
    model.print_top_words(10, data.vocab_file, result_file=model_folder+'/top_words_final.txt')

if __name__ == '__main__':
    learn('streaming-vb')
