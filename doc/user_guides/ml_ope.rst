2.10. ML-OPE
===============

Refer to the detail guide of `Online VB`_ or in tutorial `here`_

.. _Online VB: online_vb.rst
.. _here: ../tutorials/ap_tutorial.rst#learning


Learning
````````

::
   
    from tmlib.lda.ML_OPE import MLOPE
    from tmlib.datasets.dataset import DataSet

    # Assume that file isn't raw text
    training_data = DataSet(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_mlope = MLOPE(num_terms)
    obj_model = obj_mlope.learn_model(training_data)

Inference for new corpus
````````````````````````

::

    from tmlib.datasets import base
    from tmlib.lda.ML_OPE import MLOPE
    import numpy as np

    input_format = base.check_input_format(new_file_path)
    if input_format == base.DataFormat.RAW_TEXT:
        docs = base.load_batch_raw_text(new_file_path)
        vocab_dict_format = base.read_vocab(vocab_file_path)
        new_corpus = base.parse_doc_list(docs, vocab_dict_format)
    else:
        new_corpus = base.load_batch_formatted_from_file(new_file_path)
    # learned_model is a object of class LdaModel
    obj_mlope = MLOPE(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    theta = obj_mlope.infer_new_docs(new_corpus)
