import sys
sys.path.insert(0, '../preprocessing')
import preprocessing

def main(file_path):
    raw_text = preprocessing.PreProcessing(file_path, remove_rare_word=0)
    raw_text.process()
    raw_text.extract_vocab()
    raw_text.save_format_sq()
    raw_text.save_format_tf()
    print raw_text.num_terms
    print raw_text.num_docs
    return (raw_text.path_file_vocab, raw_text.path_file_tf, raw_text.path_file_sq)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python test_preprocessing.py <file_raw_text>'
    file_raw_text = sys.argv[1]
    (vocab, tf, sq) = main(file_raw_text)