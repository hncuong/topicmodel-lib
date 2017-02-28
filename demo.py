from tmlib.lda.Online_VB import OnlineVB
from tmlib.datasets.dataset import DataSet
from tmlib.datasets import base
import sys
from tmlib.config import get_config


def main():
    file_path = '/home/cuonghn/workspace/python/topicmodel-lib/tmlib/datasets/data/ap/ap.txt'
    vocab_file = '/home/cuonghn/workspace/python/topicmodel-lib/tmlib/datasets/data/ap/vocab.txt'
    data = DataSet(file_path, 100, 2)
    num_terms = 13261

    # print data.output_format
    # print data.data_format
    # cp = data.load_mini_batch()
    # print cp.word_ids_tks[0]
    # print cp.cts_lens[0]
    # print cp.format_type

    # learning
    ovb = OnlineVB(num_terms)
    model = ovb.learn_model(data)

    # print top words
    model.print_top_words(10, vocab_file)


if __name__ == '__main__':
    # main()
    #print get_config('datasets', 'tmlib_data_home')
    file_name = 'tmlib/tests/ap/ap_infer.txt'
    corpus = base.load_batch_formatted_from_file(file_name, base.DataFormat.TERM_SEQUENCE)
    print len(corpus.word_ids_tks), corpus.cts_lens
