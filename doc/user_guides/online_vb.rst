2.1. Online-VB
==============

Learning model from training set
````````````````````````````````````
  
Path of training file is *training_file_path*. If file is formatted (tf or sq), we need the vocabulary file *vocab_file_path*

First, import 2 class: `OnlineVB`_ and `DataSet`_

::
  
    from tmlib.lda.Online_VB import OnlineVB
    from tmlib.datasets.dataset import DataSet

Loading data:

::

    # if training file is raw text
    training_data = DataSet(traing_file_path, batch_size=100, passes=5, shuffle_every=2)

Or if training file is formatted (term-frequency or term-sequence), we need one more parameter: file vocabulary *vocab_file_path*

::

    # if training file is formatted
    training_data = DataSet(training_file_path, batch_size=100, passes=5, shuffle_every=2, vocab_file=vocab_file_path)

The *batch_size* controls how many documents are processed at a time in the training algorithm. Increasing batch_size will speed up training, at least as long as the chunk of documents easily fit into memory.

*passes* controls how often we train the model on the entire corpus. It also controls number of iterations in the algorithm. 

*shuffle_every* controls the stochastic property in the algorithm. Because the corpus is saved in a file, and the mini-batches are read sequentially from that file. So to have the stochastic method, after some passes, we need to shuffle all of documents in corpus. Here if we set shuffle_every=2, it means after pass over corpus 2 time, we'll shuffle corpus one time

Next, we'll create object of class OnlineVB to implement learning model

::
  
   # get number words in file vocabulary
   num_terms = training_data.get_num_terms()

   # default settings
   obj_onlvb = OnlineVB(num_terms)

   # or change settings as follow:
   # obj_onlvb = OnlineVB(num_terms, num_topics=50, alpha=0.02, eta=0.02, kappa=0.8, conv_infer=0.001, iter_infer=60)

Learning model by call function learn_model() of object OnlineVB

::

    obj_model = obj_onlvb.learn_model(training_data)
    
There are some parameters in method learn_model we need to attend: *save_model_every*, *compute_sparsity_every*, *save_statistic*, *save_top_words_every*, *num_top_words*, *model_folder* (folder we save the parameters). It means how often we save: the model (:math:`\lamda`), the statistics of method such as time for inference, time for learning in each iteration or document sparsity or top words of each topic in the learning process. In the call function above, this parameters won't be saved by default (*save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*). (See class `LdaLearning`_ for detail)

The returned result is a object of class `LdaModel`_ . The obj_model.model is value of :math:`\lambda` learned from training_data. 

.. _LdaModel: ../api/api_lda.rst
.. _LdaLearning: ../api/api_lda.rst
.. _OnlineVB: ../api/api_lda.rst
.. _Dataset: ../api/api_dataset.rst

Inference for new documents
```````````````````````````````

With the learned model, we need inference for new corpus with path file is *new_file_path*. Remember that in this part, we definitely need the file vocabulary used in training phase named *vocab_file_path*. 

::

    from tmlib.datasets import base
    from tmlib.lda.Online_VB import OnlineVB
    import numpy as np
    
First, we need check the format of data

::

    input_format = base.check_input_format(new_file_path)
    
If format of data is raw text, we need to preprocess it

::

    if input_format == base.DataFormat.RAW_TEXT:
        # load all of documents to memory with string format
        docs = base.load_batch_raw_text(new_file_path)
	# read file vocabulary and save in a dictionary structure of python
        vocab_dict_format = base.read_vocab(vocab_file_path)
	# preprocessing corpus for inference
        new_corpus = base.parse_doc_list(docs, vocab_dict_format)

If the corpus is formatted:

::

    else:
        new_corpus = base.load_batch_formatted_from_file(new_file_path)
    # learned_model is a object of class LdaModel
    # loading the model which is learned in training phase from file 
    learned_model = LdaModel(0, 0)
    learned_model.load(<path to file model-lambda learned>)
    # get number of unique terms 
    num_terms = len(open(vocab_file_path, 'r').readlines())
    # calculate topic mixtures
    obj_onlvb = OnlineVB(num_terms, lda_model=learned_model)
    theta = obj_onlvb.infer_new_docs(new_corpus)
    # we can write topic mixtures to a file
    base.write_topic_mixtures(theta, 'topic_mixtures.txt')

**Note:** 

- learned_model may be loaded from file which is saved after learning phase. See section 3 to know how to load or save a model.
- We also continually learn model by call function *learn_model*. For example: obj_onlvb.learn_model(training_data).
