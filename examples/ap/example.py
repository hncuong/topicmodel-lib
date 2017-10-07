import sys, os, shutil
import logging

from tmlib.datasets import DataSet
from tmlib.datasets import utilizies
from tmlib.lda import LdaModel
from tmlib.lda import OnlineVB, OnlineCVB0, OnlineOPE, OnlineFW, OnlineCGS, \
                    StreamingVB, StreamingOPE, StreamingFW, MLOPE, MLFW, MLCGS

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

class Example(object):

    def __init__(self):
        self.vocab_file = None
        self.train_file = None
        self.new_doc_file = None

    def learning(self, method_name):

        data = DataSet(self.train_file, batch_size=100, passes=5, shuffle_every=2, vocab_file=self.vocab_file)

        # get vocab_file after preprocessing
        self.vocab_file = data.vocab_file

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
            object = OnlineOPE(data, num_topics=20, alpha=0.2)
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

        self.model_folder_name = 'model-' + method_low

        if not os.path.exists(self.model_folder_name):
            os.mkdir(self.model_folder_name)
        model = object.learn_model()

        # save model: beta (or lambda)
        self.file_model = self.model_folder_name + '/beta_final.npy'
        model.save_model(self.file_model, file_type='binary')
        #model.save(self.model_folder_name+'/model.h5')
        # display topics to screen
        model.print_top_words(5, data.vocab_file, display_result='screen', distribution=True)

    def infer_new_docs(self, method_name):
        if self.train_file is None:
            logging.error("Run learning first")

        new_corpus = DataSet().load_new_documents(self.new_doc_file, self.vocab_file)

        # learned_model is a object of class LdaModel
        learned_model = LdaModel()
        dist_topics = learned_model.load_model(self.file_model)
        print(dist_topics.shape)

        methods = ['ml-fw', 'ml-ope', 'online-fw', 'online-ope', 'streaming-fw', 'streaming-ope',
                   'ml-cgs', 'online-cgs', 'online-cvb0', 'online-vb', 'streaming-vb']
        method_low = method_name.lower()

        if method_low == 'ml-fw':
            object = MLFW(lda_model=learned_model)
        elif method_low == 'ml-ope':
            object = MLOPE(lda_model=learned_model)
        elif method_low == 'online-fw':
            object = OnlineFW(lda_model=learned_model)
        elif method_low == 'online-ope':
            object = OnlineOPE(lda_model=learned_model)
        elif method_low == 'streaming-fw':
            object = StreamingFW(lda_model=learned_model)
        elif method_low == 'streaming-ope':
            object = StreamingOPE(lda_model=learned_model)
        elif method_low == 'ml-cgs':
            object = MLCGS(lda_model=learned_model)
        elif method_low == 'online-cgs':
            object = OnlineCGS(lda_model=learned_model)
        elif method_low == 'online-cvb0':
            object = OnlineCVB0(lda_model=learned_model)
        elif method_low == 'online-vb':
            object = OnlineVB(lda_model=learned_model)
        elif method_low == 'streaming-vb':
            object = StreamingVB(lda_model=learned_model)
        else:
            print '\ninput wrong method name: %s\n' % (method_name)
            print 'list of methods:'
            for method in methods:
                print '\t\t%s' % (method)
            exit()

        theta = object.infer_new_docs(new_corpus)
        utilizies.write_topic_proportions(theta, self.model_folder_name+'/topic_proportions.txt')




if __name__ == '__main__':

    example = Example()
    example.train_file = 'data/ap_train.txt'
    example.vocab_file = 'data/vocab.txt'
    example.learning('online-ope')
    example.new_doc_file = 'data/ap_infer.txt'
    example.infer_new_docs('online-ope')
