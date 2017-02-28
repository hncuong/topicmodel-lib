2.2. Online-CVB0
================

Refer to the detail guide of `Online VB`_ or in tutorial `here`_

.. _Online VB: online_vb.rst
.. _here: ../tutorials/ap_tutorial.rst#learning


All of steps are quite similar with `Online-VB`_. See class `Online-CVB0`_ to set the necessary parameters

**Notes**: We shouldn't use this method for the stream data which is downloaded from Wikipedia. Because this method requires the number of tokens of corpus

Learning
````````

  ::
   
    from tmlib.lda.Online_CVB0 import OnlineCVB0
    from tmlib.datasets.dataset import DataSet

    # Assume that file isn't raw text
    training_data = DataSet(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_tokens = training_data.get_num_tokens()
    num_terms = training_data.get_num_terms()
    obj_onlcvb0 = OnlineCVB0(num_tokens, num_terms)
    obj_model = obj_onlcvb0.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::
  
    from tmlib.datasets import base
    from tmlib.lda.Online_CVB0 import OnlineCVB0
    import numpy as np

    input_format = base.check_input_format(new_file_path)
    if input_format == base.DataFormat.RAW_TEXT:
        docs = base.load_batch_raw_text(new_file_path)
        vocab_dict_format = base.read_vocab(vocab_file_path)
        new_corpus = base.parse_doc_list(docs, vocab_dict_format)
    else:
        new_corpus = base.load_batch_formatted_from_file(new_file_path)
    # learned_model is a object of class LdaModel
	num_terms = len(open(vocab_file_path, 'r').readlines())
	sq_corpus = base.convert_corpus_format(new_corpus, base.DataFormat.TERM_SEQUENCE)
    num_tokens = 0
    for N in sq_corpus.cts_lens:
        num_tokens += N
    obj_onlcvb0 = OnlineCVB0(num_tokens, num_terms, lda_model=learned_model)
    theta = obj_onlcvb0.infer_new_docs(new_corpus)
	
.. _Online-CVB0: ../api/api_lda.rst
.. _Online-VB: online_vb.rst
