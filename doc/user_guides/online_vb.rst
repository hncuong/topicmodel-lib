2.1. Online-VB
==============

Learning model from training set
````````````````````````````````````
  
Path of training file is *training_file_path*. If file is formatted (tf or sq), we need the vocabulary file *vocab_file_path*

First, import 2 class: `OnlineVB`_ and `DataSet`_

  ::
  
    from tmlib.lda.Online_VB import OnlineVB
    from tmlib.datasets.dataset import DataSet

Create object of class DataSet to work with training data

  ::

    # if training file is raw text
    training_data = DataSet(traing_file_path, batch_size=100, passes=5, shuffle_every=2)

Or 

  ::

    # if training file is formatted
    training_data = DataSet(training_file_path, batch_size=100, passes=5, shuffle_every=2, vocab_file=vocab_file_path)

Create object of class OnlineVB to implement learning model

  ::
  
   # get number words in file vocabulary
   num_terms = training_data.get_num_terms()

   # default settings
    obj_onlvb = OnlineVB(num_terms)

or change settings as follow:
 
  ::

    obj_onlvb = OnlineVB(num_terms, num_topics=50, alpha=0.02, eta=0.02, kappa=0.8, conv_infer=0.001, iter_infer=60)

Learning model by call function learn_model() of object OnlineVB

  ::

    obj_model = obj_onlvb.learn_model(training_data)

This returned result is a object of class `LdaModel`_ . The obj_model.model is value of :math:`\lambda` learned from training_data. We also save model and statistics in learning by set paramaters for method *learn_model*. See class `LdaLearning`_ for detail

.. _LdaModel: ../api/api_lda.rst
.. _LdaLearning: ../api/api_lda.rst
.. _OnlineVB: ../api/api_lda.rst
.. _Dataset: ../api/api_dataset.rst

Inference for new documents
```````````````````````````````

With the learned model, we need inference for new corpus with path file is *new_file_path*

  ::

    from tmlib.datasets import base
    from tmlib.lda.Online_VB import OnlineVB
    import numpy as np

    input_format = base.check_input_format(new_file_path)
    if input_format == base.DataFormat.RAW_TEXT:
        docs = base.get_list_docs_raw_text(new_file_path)
        vocab_dict_format = base.read_vocab(vocab_file_path)
        new_corpus = base.parse_doc_list(docs, vocab_dict_format)
    else:
        fp = open(new_file_path, 'r')
        num_docs = len(fp.readlines())
        if input_format == base.DataFormat.TERM_FREQUENCY:
            new_corpus, end_file = base.load_mini_batch_term_frequency_from_term_frequency_file(fp, num_docs)
        else:
            new_corpus, end_file = base.load_mini_batch_term_frequency_from_sequence_file(fp, num_docs)
    # learned_model is a object of class LdaModel
	num_terms = len(open(vocab_file_path, 'r').readlines())
    obj_onlvb = OnlineVB(num_terms, lda_model=learned_model)
    theta = obj_onlvb.infer_new_docs(new_corpus)

**Note:** 

- learned_model may be loaded from file which is saved after learning phase. See section 3 to know how to load or save a model.
- We also continually learn model by call function *learn_model*. For example: obj_onlvb.learn_model(training_data).