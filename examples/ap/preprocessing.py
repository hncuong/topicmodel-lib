import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from lib.preprocessing import preprocessing

def main(file_path):
    raw_text = preprocessing.PreProcessing(file_path)
    raw_text.process()
    raw_text.extract_vocab()
    raw_text.save_format_sq()
    raw_text.save_format_tf()
    print 'num_terms: ', raw_text.num_terms
    print 'num_docs: ', raw_text.num_docs
    return (raw_text.path_file_vocab, raw_text.path_file_tf, raw_text.path_file_sq)

if __name__ == '__main__':
    (vocab, tf, sq) = main('data/ap_infer_raw.txt')
    print 'Vocab path: ', vocab
    print 'Term-frequency file path: ', tf
    print 'Term-sequence file path: ', sq