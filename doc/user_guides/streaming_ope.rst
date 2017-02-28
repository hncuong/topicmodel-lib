2.7. Streaming-OPE
==================

Refer to the detail guide of `Online VB`_ or in tutorial `here`_

.. _Online VB: online_vb.rst
.. _here: ../tutorials/ap_tutorial.rst#learning


Learning
````````

  ::
   
    from tmlib.lda.Streaming_OPE import StreamingOPE
    from tmlib.datasets.dataset import DataSet

    # Assume that file isn't raw text
    training_data = DataSet(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_strope = StreamingOPE(num_terms)
    obj_model = obj_strope.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.lda.Streaming_OPE import StreamingOPE
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
    obj_strope = StreamingOPE(num_terms, lda_model=learned_model)
    theta = obj_strope.infer_new_docs(new_corpus)
