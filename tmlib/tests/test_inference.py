import sys, os
from tmlib.datasets import base
from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
from tmlib.lda.Online_OPE import OnlineOPE
from tmlib.lda.ldamodel import LdaModel
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
new_file_path = 'D:/dataset/ap/ap_infer_raw.txt'
vocab_file_path = 'C:/Users/Khang Truong/tmlib_data/ap_train_raw/vocab.txt'

input_format = base.check_input_format(new_file_path)
if input_format == base.DataFormat.RAW_TEXT:
    docs = base.get_list_docs_raw_text(new_file_path)
    vocab_dict_format = read_vocab(vocab_file_path)
    new_corpus = parse_doc_list(docs, vocab_dict_format)
else:
    fp = open(new_file_path, 'r')
    num_docs = len(fp.readlines())
    if input_format == base.DataFormat.TERM_FREQUENCY:
        new_corpus = base.load_mini_batch_term_frequency_from_term_frequency_file(fp, num_docs)
    else:
        new_corpus = base.load_mini_batch_term_frequency_from_sequence_file(fp, num_docs)
# learned_model is a object of class LdaModel
learned_model = LdaModel(0,0)

_lambda = learned_model.load('model/lambda_final.txt')
print(learned_model.model.shape)
obj_onlope = OnlineOPE(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
theta = obj_onlope[new_corpus]
print theta[1]