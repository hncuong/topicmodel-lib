import os, logging

from wiki_stream import WikiStream
from tmlib.lda import StreamingFW
from tmlib.lda import StreamingOPE
from tmlib.lda import StreamingVB

def learn(method_name):
    data = WikiStream(500, 1000)
    methods = ['streaming-fw', 'streaming-ope', 'streaming-vb']

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
    model = object.learn_model(save_topic_proportions=model_folder_name+'/db.h5')
    model.save(os.path.join(model_folder_name,'model.h5'))
    model.print_top_words(10, data.vocab_file, display_result=os.path.join(model_folder_name,'top_word_final.txt'))

if __name__ == '__main__':
    learn('streaming-ope')
    #learn('streaming-fw')
    #learn('streaming-vb')
