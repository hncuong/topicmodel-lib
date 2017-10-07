==============
Module tmlib.datasets
==============

This module includes some classes and utility functions which help us work with the dataset

-----------------------------------------------------
class DataSet
-----------------------------------------------------

This is the main class storing the information about your corpus such as: number of documents, size of vocabulary set, etc. You also can load the mini-batches data to implement your learning algorithm. 

::

  tmlib.datasets.DataSet(data_path=None, batch_size=None, passes=1, shuffle_every=None, vocab_file=None)

Parameters
===========

- **data_path**: string,

  Path of file input (corpus)
- **batch_size**: int
  
  size of mini-batch in each sampling from corpus. 

- **passes**: int, default: 1

  passes controls how often we train the model on the entire corpus. Another word for passes might be "epochs" (in training neural network). iterations is somewhat technical, but essentially it controls how often we repeat a particular loop over each document. It is important to set the number of "passes" and "iterations" high enough.
  
  For example, if you set passes = 5, assume that batch_size = 100 and size of corpus is 10000 then number of training iterations is 10000/100*5 = 5000

- **shuffle_every**: int,

  This parameter help us shuffle the samples (documents) in corpus at each pass (epoch)

  If you set shuffle_every=2, it means after passing over corpus 2 times, corpus will be shuffled

- **vocab_file**: string, default: None
  
  File vocabulary of corpus
  
  If corpus is raw text format, file vocabulary is non-necessary. Otherwise, if corpus is tf or sq format, user must set it

Attributes
==========

- **batch_size**: int
- **vocab_file**: string,
- **num_docs**: int,

  Return number of document in corpus
- **data_path**: string, 

  path of file corpus which is the term-frequency or term-sequence format
- **data_format**: attribute of class ``DataFormat``

  The class ``DataFormat`` stores name of formats data: DataFormat.RAW_TEXT, DataFormat.TERM_SEQUENCE or DataFormat.TERM_FREQUENCY
- **output_format**: attribute of class ``DataFormat``, default is DataFormat.TERM_FREQUENCY
  
  format of mini-batch. User change the format by use method *set_output_format*

- **passes**: int
- **shuffle_every**: int

- **work_path**: string

  This path is different from data_path. If corpus is shuffled then work_path is path of the shuffled file, not the original file

Methods
=======

- __init__(*data_path=None, batch_size=None, passes=1, shuffle_every=None, vocab_file=None*)
- **load_mini_batch** ()

  loading a mini-batch from corpus with specific format (controlled by **output_format**)
  Return: object class ``Corpus`` storing mini-batch
  
- **load_new_document** (path_file, vocab_file=None)

  You can load new document from ``path_file``. If format of file is raw text, you need add ``vocab_file``
  Return: object ``Corpus``

- **check_end_of_data** ()

  To check out that whether we visit to the last mini-batch or not.
  
  Return True if the last mini-batch is loaded and the training is done
  
- **set_output_format** (output_format)

  set format for the loaded mini-batch

  - **Parameters**: output_format (DataFormat.TERM_SEQUENCE or DataFormat.TERM_FREQUENCY)


- **get_total_docs** ()

  Return number of documents which have been analyzed until the present

- **get_num_tokens** ()

  Return number of tokens in corpus 

- **get_num_terms** ()

  Return number of unique terms in corpus (size of vocabulary set)
  
Example
=======

- Load mini-batch with term-frequency format

::

  from tmlib.datasets import DataSet
    
  #AP corpus in folder examples/ap/data
  data = DataSet(data_path='data/ap_train_raw.txt', batch_size=100, passes=4, shuffle_every=2)
  minibatch = data.load_mini_batch()  # The format is term-frequency by default
  
- Load mini-batch with term-sequence format

::

  from tmlib.datasets import DataSet
  from tmlib.datasets.utilities import DataFormat
    
  #AP corpus in folder examples/ap/data
  data = DataSet(data_path='data/ap_train_raw.txt', batch_size=100, passes=4, shuffle_every=2)
  data.set_output_format(DataFormat.TERM_SEQUENCE)
  minibatch = data.load_mini_batch()

In these examples, we set **passes=4** and **shuffle_every=2**, it means: 4 times of passing over data and after every 2 times, corpus is shuffled again. Assume that size of corpus is 5000 documents, batch_size = 100, then number of iterators is: 5000/100*4 = 2000. We can check the last iterator by using method *check_end_of_data()*.

-----------------------------------------------------
class DataFormat
-----------------------------------------------------

This is class which contains 3 `data-format`_ types of library is: raw text, term_sequence, term-frequency

.. _data-format: ./quick_start.rst

::

  tmlib.datasets.utilities.DataFormat

Static Attributes
=================

- **RAW_TEXT**: string, value is 'txt'
- **TERM_FREQUENCY**: string, value is 'tf'
- **TERM_SEQUENCE**: string, value is 'sq'

Example
=======
This example allows checking data format for: corpus *examples/ap/ap_train_raw.txt*

::

  from tmlib.datasets.utilities import DataFormat, check_input_format

  input_format = check_input_format('examples/ap/ap_train_raw.txt')
  print(input_format)
  if input_format == DataFormat.RAW_TEXT:
      print('Corpus is raw text')
  elif input_format == DataFormat.TERM_SEQUENCE:
      print('Corpus is term-sequence format')
  else:
      print(Corpus is term-frequency format')
        
**Output**:

::

  txt
  Corpus is raw text
  
--------------------------------
class Corpus
--------------------------------

This class is used to store the corpus with 2 formats: term-frequency and term-sequence

::

  tmlib.datasets.utilities.Corpus(format_type)

Parameters
==========

- **format_type**: DataFormat.TERM_SEQUENCE or DataFormat.TERM_FREQUENCY

Attributes
==========

- **format_type**: format of corpus
- **word_ids_tks**: list of list,

  Each element in this list is a list which include the words of a document in corpus (words is unique terms if format is term-frequency and is list of tokens if format is term-sequence)

- **cts_lens**: list

  if format is term-frequency, each element in list is a list frequency of unique terms in respectly document of corpus. If format is term-sequence, each element in list is the number of tokens in document (number of tokens in each doc).

Methods
=======

- **append_doc** (ids_tks, cts_lens)

  Add a document to corpus. If format of this document is term-frequency, this method will append list of unique terms to **word_ids_tks** and append list of frequency to **cts_lens**. If format is term-sequence, the list of tokens and number of tokens will be appended respectly

  - **Parameters**: ids_tks and cts_lens is format (tf or sq) of added document
    
    **ids_tks**: list of unique terms (term-frequency format) or list of tokens (term-sequence format)
    **cts_lens**: list of frequency of unique terms (term-frequency format) or number tokens in document (term-sequence format)

-----------------
Utility functions
-----------------

These functions below are in module ``tmlib.datasets.utilities``

get_data_home
=============

::

  tmlib.datasets.utilities.get_data_home(data_home=None)

This folder is used by some large dataset loaders to avoid downloading the data several times.

By default the data dir is set to a folder named 'tmlib_data' in the user home folder. We can change it by change value of data_home parameter
The '~' symbol is expanded to the user home folder.

If the folder does not already exist, it is automatically created.

- **Return**: path of the tmlib data dir.

>>> from tmlib.datasets import utilities
>>> print 100.get_data_home()
/home/kde/tmlib_data

clear_data_home
===============

::

  tmlib.datasets.utilities.clear_data_home(data_home=None)

Delete all the content of the data home cache. 

check_input_format
==================

::

  tmlib.datasets.utilities.check_input_format(file_path)

- Check format of input file(text formatted or raw text)
- **Parameters**: file_path (string)

  Path of file input
- **Return**: format of input (DataFormat.RAW_TEXT, DataFormat.TERM_FREQUENCY or DataFormat.TERM_SEQUENCE)

>>> from tmlib.datasets import utilities
>>> file_path = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> print utilities.check_input_format(file_path)
tf
>>> file_path = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train_raw.txt'
>>> print utilities.check_input_format(file_path)
txt

load_batch_raw_text
===================

::

  tmlib.datasets.utilities.load_batch_raw_text(file_raw_text_path)

- load all of documents and store as a list. Each element in this list is a document with raw text format (string)

- **Parameters**: file_raw_text_path (string)

  Path of file input 
  
- **Return**: list, each element in list is string type and also is text of a document

>>> from tmlib.datasets import utilities
>>> path_file_raw_text = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_infer_raw.txt'
>>> list_docs = utilities.load_batch_raw_text(path_file_raw_text)
>>> print 'number of documents: ', len(list_docs)
number of documents:  50
>>> print list_docs[8]
 Here is a summary of developments in forest and brush fires in Western states:

pre_process
===========

::

  tmlib.datasets.utilities.pre_process(file_path)

- Preprocessing for file input if format of data is raw text 
- **Paremeter**: file_path (string)

  Path of file input
- **Return**: list which respectly includes path of vocabulary file, term-frequency file, term-sequence file after preprocessing

>>> from tmlib.datasets import utilities
>>> path_file = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train_raw.txt'
>>> path_vocab, path_tf, path_sq = utilities.pre_process(path_file)
Waiting...
>>> print 'path to file vocabulary extracted: ', path_vocab
path to file vocabulary extracted:  /home/kde/tmlib_data/ap_train_raw/vocab.txt
>>> print 'path to file with term-frequency format: ', path_tf
path to file with term-frequency format:  /home/kde/tmlib_data/ap_train_raw/ap_train_raw.tf
>>> print 'path to file with term-sequence format: ', path_sq
path to file with term-sequence format:  /home/kde/tmlib_data/ap_train_raw/ap_train_raw.sq

load_batch_formatted_from_file
==============================

::

  tmlib.datasets.utilities.load_batch_formatted_from_file(data_path, output_format=DataFormat.TERM_FREQUENCY)

- load all of documents in file which is formatted as term-frequency format or term-sequence format and return a corpus with format is **output_format**
- **Parameters**:

  - **data_path**: path of file data input which is formatted
  - **output_format**: format data of output, default: term-frequence format
  
- **Return**: object corpus which is the data input for learning 

>>> path_file_tf = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> corpus_tf = utilities.load_batch_formatted_from_file(path_file_tf)
>>> print 'Unique terms in the 9th documents: ', corpus_tf.word_ids_tks[8]
Unique terms in the 9th documents:  [5829 4040 2891   14 1783  381 2693]
>>> print 'Frequency of unique terms in the 9th documents: ', corpus_tf.cts_lens[8]
Frequency of unique terms in the 9th documents:  [1 1 1 1 1 1 1]
>>> corpus_sq = utilities.load_batch_formatted_from_file(path_file_tf, output_format=utilities.DataFormat.TERM_SEQUENCE)
>>> print 'List of tokens in the 9th documents: ', corpus_sq.word_ids_tks[8]
List of tokens in the 9th documents:  [5829 4040 2891   14 1783  381 2693]
>>> print 'Number of tokens in the 9th document: ', corpus_sq.cts_lens[8]
Number of tokens in the 9th document:  7


reformat_file_to_term_sequence
==============================

::

  tmlib.datasets.utilities.reformat_file_to_term_sequence(file_path)

- convert the formatted file input (tf or sq) to file with format term-sequence
- **Parameter**: file_path (string)

  Path of file input
- **Return**: path of file which is formatted to term-sequence

>>> from tmlib.datasets import utilities
>>> path_file_tf = tmlib
>>> path_file_sq = utilities.reformat_file_to_term_sequence(path_file_tf)
>>> print 'path to file term-sequence: ', path_file_sq
path to file term-sequence:  /home/kde/tmlib_data/ap_train/ap_train.sq


reformat_file_to_term_frequency
===============================

::

  tmlib.datasets.utilities.reformat_file_to_term_sequence(file_path)

- convert the formatted file input (tf or sq) to file with format term-frequency
- **Parameter**: file_path (string)

  Path of file input
- **Return**: path of file which is formatted to term-frequency

>>> from tmlib.datasets import utilities
>>> path_file = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> path_file_tf = utilities.reformat_file_to_term_sequence(path_file)
>>> print 'path to file term-frequency: ', path_file_tf
path to file term-frequency:  /home/kde/tmlib_data/ap_train/ap_train.tf


convert_corpus_format
=====================

::

  tmlib.datasets.utilities.convert_corpus_format(corpus, data_format)

- convert corpus (object of class ``Corpus``) to desired format
- **Parameters**:

  - **corpus**: object of class Corpus, 
  - **data_format**: format type desired (DataFormat.TERM_SEQUENCE or DataFormat.TERM_FREQUENCY)

- **Return**: object corpus with desired format

>>> from tmlib.datasets import utilities
>>> path_file_tf = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> corpus = utilities.load_batch_formatted_from_file(path_file_tf)
>>> corpus_sq = utilities.convert_corpus_format(corpus, utilities.DataFormat.TERM_SEQUENCE)
>>> print 'Unique terms in the 22th documents: ', corpus.word_ids_tks[21]
Unique terms in the 22th documents:  [  32  396  246   87  824 3259  316  285]
>>> print 'Frequency of unique terms in the 22th documents: ', corpus.cts_lens[21]
Frequency of unique terms in the 22th documents:  [1 1 1 2 1 1 2 1]
>>> print 'List of tokens in the 22th documents: ', corpus_sq.word_ids_tks[21]
List of tokens in the 22th documents:  [32, 396, 246, 87, 87, 824, 3259, 316, 316, 285]
>>> print 'Number of tokens in the 22th document: ', corpus_sq.cts_lens[21]
Number of tokens in the 22th document:  10


compute_sparsity
================

::

  tmlib.datasets.utilities.compute_sparsity(doc_tp, num_docs, num_topics, _type)

- Compute document sparsity.
- **Parameters**:

  - **doc_tp**: numpy.array, 2-dimention, the estimated topic mixtures of all documents in corpus
  - **num_docs**: int, the number of documents in corpus
  - **num_topics**: int, is the number of requested latent topics to be extracted from the training corpus.
  - **_type**: string, if the value is 'z', the topic mixtures is estimated by the sampling method as CGS or CVB0, so we have the individual caculation for this. Otherwise, if the value of it isn't 'z', this is for the methods as: VB, OPE or FW
  
- **Return**: float, sparsity of documents

>>> import numpy as np
>>> from tmlib.datasets import utilities
>>> theta = np.array([[0.1, 0.3, 0.2, 0.2, 0.1, 0.1], [0.02, 0.05, 0.03, 0.5, 0.2, 0.2]], dtype='float32')
>>> utilities.compute_sparsity(theta, theta.shape[0], theta.shape[1], _type='t')
1.0


write_topic_proportions 
=======================

tmlib.datasets.utilities.write_topic_proportions(theta, file_name)

- save topic mixtures (theta) to a file
- **Parameters**:

  - **theta**: numpy.array, 2-dimention
  - **file_name**: name (path) of file which is written
