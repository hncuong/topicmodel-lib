.. -*- coding: utf-8 -*-

===========
User guide
===========

This document contains a description of all stochastic algorithms for learning LDA, and it also contains some tutorials about how to use the useful functions or methods in the library for many purposes. To understand clearly this document, user need to read `quick start`_ document and `lda model`_ first.

.. _quick start: ./quick_start.rst
.. _lda model: ./LatentDirichletAllocation.rst

.. Contents::


---------------------------
1. Working with data input
---------------------------

This section includes some tutorials for process data input of model (documents - corpus). This corpus maybe supplied by user, or available copus from `wikipedia`_ website (refer to `paper`_ and `source code`_). The library will support preprocessing, converting format of input for specific learning method.

.. _wikipedia: https://en.wikipedia.org/wiki/Main_Page
.. _paper: https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf
.. _source code: https://github.com/blei-lab/onlineldavb

1.1. Preprocessing
==================

- This work will be implemented when data format of input is raw text. Topic models take documents that contain words as input. We still have to determine what "words" we're going to use and how to extract them from the format raw text. Recall that most topic models treat documents as a bag-of-words, so we can stop caring about the order of the tokens within the text and concentrate on how many times a particular word appears in the text. So, we need to convert the raw format to term-sequence or term-frequency as mentioned in the `quick start`_ section. To understand in detail about technique of preprocessing, please read preprocessing [1]_ document. 

- File raw text also need a specific format type so that we can recognize it. The format of file   as follow:

  - Corpus includes many documents, all of that are saved into a file. 
  - Each document is represented as follow

    .. image:: images/format.PNG
   
- This is tutorial for how to preprocess a file raw text:

  ::
    
    from tmlib.preprocessing.preprocessing import PreProcessing

    object = PreProcessing(file_path)                  
    object.process()                  # run algorithm of preprocessing step
    object.extract_vocab()            # extract to the vocabulary of corpus
    object.save_format_sq()           # save the new format is term-sequence format
    object.save_format_tf()           # save the format is term-frequency format
    # display path of file vocabulary, file term-sequence, file term-frequency
    print(object.path_file_vocab, object.path_file_sq, object.path_file_tf)

  The result files is automatically saved in a folder named "tmlib_data" in the user data home. User can change the position by set value parameters in functions such as extract_vocab(), save_format_sq() or save_format_tf(). User can also change the setting parameters of preprocessing algorithm by set value when create object. More detailed, refer to the `api preprocessing`_ document.

.. _api preprocessing: ./api/api_preprocessing.rst

1.2. Loading a "mini-batch" from corpus
=======================================

- This is a extension of the basic stochastic inference [2]_, the use of multiple samples (“minibatches”) to improve the algorithm’s stability. Previously, stochastic variational inference algorithms sample only one observation (one document) at a time from corpus. Many stochastic optimization algorithms benefit from “minibatches,” that is, several examples at a time (Bottou and Bousquet, 2008; Liang et al., 2009; Mairal et al., 2010). 
- There are two reasons to use minibatches. 
  
  - First, to amortize any computational expenses associated with updating the global parameters across more data points; for example, if the expected sufficient statistics of β are expensive to compute, using minibatches allows us to incur that expense less frequently. 
  - Second, it may help the algorithm to find better local optima. Stochastic variational inference is guaranteed to converge to a local optimum but taking large steps on the basis of very few data points may lead to a poor one. Using more of the data per update can help the algorithm (refer to [2]_)

- Thus, if users want to load a minibatch which the size is **batch_size** from the file corpus has path is **file_path**, there are two choices:
  
  - Sample randomly **batch_size** documents from file at each iterator
  - Shuffle (arrange randomly) all of documents in file. After that, we'll load minibatches from beginning to end of file in order (pass over data one time). We can do this several time and then, shuffle file again and repeat loading. 

- The library provides a class named "Dataset" to implement the second choice:
 
  - Loading a minibatch which have format is term-frequency
    
    ::

      from tmlib.datasets import dataset
    
      data = dataset.Dataset(file_path, batch_size)
      minibatch = data.load_mini_batch()  # The format is term-frequency by default
     
    By default in above, number of passing over data is 1. We can change it by set:

    ::  
   
      data = dataset.Dataset(file_path, batch_size, passes=4, shuffle_every=2)
    
    This means 4 times of passing over data and after every 2 times, file is shuffled again. Assume that size of corpus is 5000 documents, batch_size = 100, then number of iterators is: 5000/100*4 = 2000. We can check the last iterator by using method *check_end_of_data()*.
  - output format is term-sequence

    ::

      from tmlib.datasets import dataset
      from tmlib.datasets.base import DataFormat

      data = dataset.Dataset(file_path, batch_size)
      data.set_output_format(DataFormat.TERM_SEQUENCE)
      minibatch = data.load_mini_batch()
      
- However, we can also implement the first choice as follow:

  - Define a function *sample()* with 2 parameters is: file which is formatted (tf or sq) and format of output (minibatch)
 
    ::

      from tmlib.datasets import base
      from tmlib.datasets.base import DataFormat
  
      def sample(file_formatted_path, batch_size, output_format):
          work_file = base.shuffle_formatted_data_file(file_formatted_path)
          fp = open(work_file, 'r')
          if output_format == DataFormat.TERM_FREQUENCY:
              return base.load_mini_batch_term_frequency_from_term_frequency_file(fp, batch_size)
          else:
              return base.load_mini_batch_term_sequence_from_sequence_file(fp, batch_size) 

  - Loading a minibatch which has term-frequency format

    ::
  
      input_format = base.check_input_format(file_path)
      if input_format == DataFormat.RAW_TEXT:
          vocab_file, tf_file, sq_file = base.pre_process(file_path)
          work_file_path = tf_file
      else:
          work_file_path = base.reformat_file_to_term_frequency(file_path)
      # at each iterator, repeat this statement
      minibatch = sample(work_file_path, batch_size, DataFormat.TERM_FREQUENCY)

  - Loading a minibatch which has term-sequence format

    ::
  
      input_format = base.check_input_format(file_path)
      if input_format == DataFormat.RAW_TEXT:
          vocab_file, tf_file, sq_file = base.pre_process(file_path)
          work_file_path = sq_file
      else:
          work_file_path = base.reformat_file_to_term_sequence(file_path)
      # at each iterator, repeat this statement
      minibatch = sample(work_file_path, batch_size, DataFormat.TERM_FREQUENCY)

- Note: if minibatch is term-frequency format, the returned result is tuple *(wordids, wordcts)*.

  - The first, *wordids*, says what vocabulary tokens are present in each document. wordids[i][j] gives the jth unique token present in document i. (Don't count on these tokens being in any particular order.)
  - The second, *wordcts*, says how many times each vocabulary token is present. wordcts[i][j] is the number of times that the token given by wordids[i][j] appears in document i.

- if minibatch is term-sequence format, the result is pair *(wordtks, lens)*, where

  - *wordtks* is a list of documents, each document is represented as a sequence of tokens
  - *lens* is list of number of tokens in each document.

1.3. Loading a minibatch from Wikipedia website
===============================================
- This is a simulation of stream data (the data observations are arriving in a continuous stream). So, we can't pass over all of data. At each iterator, we'll download and analyze a bunch of random Wikipedia
- With size of batch is **batch_size** and number of iterators is **num_batches**:

  ::
  
    from tmlib.datasets.wiki_stream import WikiStream
    from tmlib.datasets.base import DataFormat

    data = WikiStream(batch_size, num_batches)
    minibatch = data.load_mini_batch() # the format is term frequency by default

- To load minibatch with term-sequence format, add method *set_output_format* before *load_mini_batch()*
  
  ::
    
    data.set_output_format(DataFormat.TERM_SEQUENCE)

---------------------------------------------------------
2. Stochastic methods for learning LDA from large corpora
---------------------------------------------------------

.. _OPE: https://arxiv.org/abs/1512.03308
.. _FW: https://arxiv.org/abs/1512.03300

Our library is deployed with 5 inference methods: VB [3]_, CVB0 [4]_, CGS [5]_, `OPE`_, `FW`_ and apply with online scheme, stream scheme or regularized online learning

- Online learning includes methods: Online-VB [2]_, Online-CVB0 [6]_, Online-CGS [5]_, Online-OPE, Online-FW
- Stream learning includes: Streaming-VB [7]_, Streaming-OPE, Streaming-FW
- Regularized online learning with CGS, FW, OPE: ML-CGS, ML-FW, ML-OPE.

**Default setting parameters**: 

- *Model paramters*: num_topics = 100; :math:`\alpha` = 0.01; :math:`\eta` = 0.01. Such a choice of (:math:`\alpha`, :math:`\eta`) has been observed to work well in many previous studies
- *Inference parameters*: at most 50 iterations were allowed for OPE and VB to do inference. We terminated VB if the relative improvement of the lower bound on likelihood is not better than 10−4. 50 samples were used in CGS for which the first 25 were discarded and the remaining were used to approximate the posterior distribution. 50 iterations were used to do inference in CVB0, in which the first 25 iterations were burned in. Those number of samples/iterations are often enough to get a good inference solution according to [5]_ , [6]_
- *Learning parameters*: :math:`\kappa` = 0.9; :math:`\tau` = 1; in addition to, :math:`\beta` (or :math:`\lambda`) is instructed randomly with a specific distribution, depend on each method learning.
- User can change value of setting parameters above when object of method is created. (see example below)

2.1. Online-VB
==============

Learning model from training set
````````````````````````````````````
  
Path of training file is *training_file_path*. If file is formatted (tf or sq), we need the vocabulary file *vocab_file_path*

First, import 2 class: `OnlineVB`_ and `Dataset`_

  ::
  
    from tmlib.lda.Online_VB import OnlineVB
    from tmlib.datasets.dataset import Dataset

Create object of class Dataset to work with training data

  ::

    # if training file is raw text
    training_data = Dataset(traing_file_path, batch_size=100, passes=5, shuffle_every=2)

Or 

  ::

    # if training file is formatted
    training_data = Dataset(training_file_path, batch_size=100, passes=5, shuffle_every=2, vocab_file=vocab_file_path)

Create object of class OnlineVB to implement learning model

  ::

   # default settings
    obj_onlvb = OnlineVB(training_data.get_num_terms())

or change settings as follow:
 
  ::

    obj_onlvb = OnlineVB(training_data.get_num_terms(), num_topics=50, alpha=0.02, eta=0.02, kappa=0.8, conv_infer=0.001, iter_infer=60)

Learning model by call function learn_model() of object OnlineVB

  ::

    obj_model = obj_onlvb.learn_model(training_data)

This returned result is a object of class `LdaModel`_ . The obj_model.model is value of :math:`\lambda` learned from training_data. 

.. _LdaModel: ./api/api_lda.rst
.. _OnlineVB: ./api/api_lda.rst
.. _Dataset: ./api/api_dataset.rst

Inference for new documents
```````````````````````````````

With the learned model, we need inference for new corpus with path file is *new_file_path*

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.Online_VB import OnlineVB
    import numpy as np

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
    obj_onlvb = OnlineVB(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    gamma = obj_onlvb[new_corpus]
    gamma_norm = gamma.sum(axis=1)
    theta = gamma / gamma_norm[:, np.newaxis]

**Note:** 

- learned_model may be loaded from file which is saved after learning phase. See section 3 to know how to load or save a model.
- We also learn more by call obj_onlvb.learn_model(training_data,...). It's up to user

2.2. Online-CVB0
================
All of steps are quite similar with Online-VB. See class `OnlineCVB0`_ to set the necessary parameters

.. _OnlineCVB0: ./api/api_lda.rst

Learning
````````

  ::
   
    from tmlib.lda.Online_CVB0 import OnlineCVB0
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_tokens = training_data.get_num_tokens()
    num_terms = training_data.get_num_terms()
    obj_onlcvb0 = OnlineCVB0(num_tokens, num_terms)
    obj_model = obj_onlcvb0.learn_model(training_data)

Inference for new corpus
````````````````````````

2.3. Online-CGS
===============

Learning
````````

  ::
   
    from tmlib.lda.Online_CGS import OnlineCGS
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_onlcgs = OnlineCGS(num_terms)
    obj_model = obj_onlcgs.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.Online_CGS import OnlineCGS
    import numpy as np

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
    obj_onlcgs = OnlineCGS(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    gamma = obj_onlcgs[new_corpus]
    gamma_norm = gamma.sum(axis=1)
    theta = gamma / gamma_norm[:, np.newaxis]

2.4. Online-OPE
===============

Learning
````````

  ::
   
    from tmlib.lda.Online_OPE import OnlineOPE
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_onlope = OnlineOPE(num_terms)
    obj_model = obj_onlope.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.Online_OPE import OnlineOPE
    import numpy as np

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
    obj_onlope = OnlineOPE(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    theta = obj_onlope[new_corpus]

2.5. Online-FW
===============

Learning
````````

  ::
   
    from tmlib.lda.Online_FW import OnlineFW
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_onlfw = OnlineFW(num_terms)
    obj_model = obj_onlfw.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.Online_FW import OnlineFW
    import numpy as np

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
    obj_onlfw = OnlineFW(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    theta = obj_onlfw[new_corpus]

2.6. Streaming-VB
=================

Learning
````````

  ::
   
    from tmlib.lda.Streaming_VB import StreamingVB
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_strvb = StreamingVB(num_terms)
    obj_model = obj_strvb.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.Streaming_VB import StreamingVB
    import numpy as np

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
    obj_strvb = StreamingVB(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    gamma = obj_strvb[new_corpus]
    gamma_norm = gamma.sum(axis=1)
    theta = gamma / gamma_norm[:, np.newaxis]

2.7. Streaming-OPE
==================

Learning
````````

  ::
   
    from tmlib.lda.Streaming_OPE import StreamingOPE
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_strope = StreamingOPE(num_terms)
    obj_model = obj_strope.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.Streaming_OPE import StreamingOPE
    import numpy as np

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
    obj_strope = StreamingOPE(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    theta = obj_strope[new_corpus]

2.8. Streaming-FW
==================

Learning
````````

  ::
   
    from tmlib.lda.Streaming_FW import StreamingFW
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_strfw = StreamingFW(num_terms)
    obj_model = obj_strfw.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.Streaming_FW import StreamingFW
    import numpy as np

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
    obj_strfw = StreamingFW(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    theta = obj_strfw[new_corpus]

2.9. ML-CGS
===============

Learning
````````

  ::
   
    from tmlib.lda.ML_CGS import MLCGS
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_mlcgs = MLCGS(num_terms)
    obj_model = obj_mlcgs.learn_model(training_data)

With ML-methods, model returned is :math:`\beta`.

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.ML_CGS import MLCGS
    import numpy as np

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
    obj_mlcgs = MLCGS(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    gamma = obj_mlcgs[new_corpus]
    gamma_norm = gamma.sum(axis=1)
    theta = gamma / gamma_norm[:, np.newaxis]

2.10. ML-OPE
===============

Learning
````````

  ::
   
    from tmlib.lda.ML_OPE import MLOPE
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_mlope = MLOPE(num_terms)
    obj_model = obj_mlope.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.ML_OPE import MLOPE
    import numpy as np

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
    obj_mlope = MLOPE(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    theta = obj_mlope[new_corpus]


2.11. ML-FW
===============

Learning
````````

  ::
   
    from tmlib.lda.ML_FW import MLFW
    from tmlib.datasets.dataset import Dataset

    # Assume that file isn't raw text
    training_data = Dataset(training_file_path, batch_size=100, vocab_file=vocab_file_path)
    num_terms = training_data.get_num_terms()
    obj_mlfw = MLFW(num_terms)
    obj_model = obj_mlfw.learn_model(training_data)

Inference for new corpus
````````````````````````

  ::

    from tmlib.datasets import base
    from tmlib.datasets.wiki_stream import parse_doc_list, read_vocab
    from tmlib.lda.ML_FW import MLFW
    import numpy as np

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
    obj_mlfw = MLFW(len(open(vocab_file_path, 'r').readlines()), lda_model=learned_model)
    theta = obj_mlfw[new_corpus]

----------------------------
3. How to save or load model
----------------------------

Save model (:math:`\lambda` or :math:`\beta`)
==============================================
After learning model **obj_model** as above. We can save this result as follow:

  ::
    
    obj_model.save(file_name, file_type='text')

The result model is saved in file named *file_name* with format file is text. The default format is binary file if we remove the file_type parameter. 

Moreover, we can save the model and some statistics like the study time, topic mixtures :math:`\theta`, the sparsity of document [8]_ in the running process of the algorithm so that we can have necessary comparison and assessment. For example with VB method:

  ::

    obj_onlvb = OnlineVB(training_data, save_model_every=2, compute_sparsity_every=2, save_statistic=True, save_top_words_every=2, num_top_words=20, model_folder='model_vb')

This means after 2 iterators, the model, time of E-step, M-step and document sparsity is saved into files. All of this files is in the folder 'model_vb' named by user.

Load model from a file
======================
Assume that :math:`\lambda` or :math:`\beta` is saved in a file has path *model_file_path*. Loading is supported with 2 type of file: text (.txt) and binary (.npy). 

  ::

    from tmlib.lda.ldamodel import LdaModel

    obj_model = LdaModel(num_terms, num_topics)
    obj_model.load(model_file_path)

The num_terms and num_topics are 2 parameters which are determined by user. For example, if we combine this section with tutorial learning, we can set

    num_terms = training_data.get_num_terms()
    num_topics = obj_onlvb.num_topics      # for example with Online-VB method

Save top words of topics
========================

Display to the screen

  :: 
    
    # print 10 topics, top 20 words which have the highest probability will be displayed in each topic
    obj_model.print_top_words(20, vocab_file_path, show_topics=10)

Save into a file

  ::

    obj_model.print_top_words(20, vocab_file_path, show_topics=30, result_file='topics.txt')

.. [1] Care and Feeding of Topic Models: Problems, Diagnostics, and Improvements Jordan Boyd-Graber, David Mimno, and David Newman. In Handbook of Mixed Membership Models and Their Applications, CRC/Chapman Hall, 2014.
.. [2] M.D. Hoffman, D.M. Blei, C. Wang, and J. Paisley, "Stochastic variational inference," The Journal of Machine Learning Research, vol. 14, no. 1, pp. 1303–1347, 2013.
.. [3] D.M. Blei, A.Y. Ng, and M.I. Jordan, "Latent dirichlet allocation," Journal of Machine Learning Research, vol. 3, no. 3, pp. 993–1022, 2003.
.. [4] Asuncion, M. Welling, P. Smyth, and Y. Teh, "On smoothing and inference for topic models," in Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, 2009, pp. 27–34.
.. [5] D.Mimno, M. D. Hoffman, and D. M. Blei, "Sparse stochastic inference for latent dirichlet allocation," in Proceedings of the 29th Annual International Conference on Machine Learning, 2012.
.. [6] James Foulds, Levi Boyles, Christopher DuBois, Padhraic Smyth, and Max Welling. Stochastic collapsed variational bayesian inference for latent dirichlet allocation. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 446–454. ACM, 2013.
.. [7] Tamara Broderick, Nicholas Boyd, Andre Wibisono, Ashia C Wilson, and Michael Jordan. Streaming variational bayes. In Advances in Neural Information Processing Systems, pages 1727–1735, 2013.
.. [8] Khoat Than and Tu Bao Ho, “Fully sparse topic models”. European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD), Bristol, UK. Vol. 7523 of Lecture Notes in Computer Science, Springer, pages 490-505, 2012.
