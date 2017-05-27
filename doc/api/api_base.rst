.. -*- coding: utf-8 -*-

=================================
tmlib.datasets.base
=================================
This module includes some base classes and base functions which support for working with data input (corpus)

.. Contents::


-----------------------------------------------------
class tmlib.datasets.base.DataFormat
-----------------------------------------------------

This is class which contains 3 `data-format`_ types of library is: raw text, term_sequence, term-frequency

.. _data-format: ../quick_start.rst#data-input-format

Static Attributes
=================

- **RAW_TEXT**: string, value is 'txt'
- **TERM_FREQUENCY**: string, value is 'tf'
- **TERM_SEQUENCE**: string, value is 'sq'

Example
=======
This example allows checking data format for: corpus *examples/ap/ap_train_raw.txt*

::

    from tmlib.datasets.base import DataFormat, check_input_format

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
class tmlib.datasets.base.Corpus
--------------------------------

tmlib.datasets.base. **Corpus** (*format_type*)

This class is used to store the corpus with 2 formats: term-frequency and term-sequence

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

--------------------------------------
class tmlib.datasets.base.DataIterator
--------------------------------------

tmlib.datasets.base.DataIterator()

Abstract class is used for working with data input in learning LDA. Inherited by 2 class: `tmlib.datasets.dataset.Dataset`_ and `tmlib.datasets.wiki_stream.WikiStream`_

.. _tmlib.datasets.dataset.Dataset: ./api_dataset.rst
.. _tmlib.datasets.wiki_stream.WikiStream: ./api_wiki.rst

Attributes
==========

- **mini_batch_no**: int, 

  Numerical order of mini-batch which used for learning at the present
- **end_of_data**: boolean,

  To check the mini-batch is the last mini-batch or not

Methods
=======

- **load_mini_batch** ()

  return a mini-batch sampled from corpus

- **check_end_of_data** ():
  
  return value of **end_of_data** variable

---------------------------
Function base.get_data_home
---------------------------

tmlib.datasets.base. **get_data_home** (data_home=None)

This folder is used by some large dataset loaders to avoid downloading the data several times.

By default the data dir is set to a folder named 'tmlib_data' in the user home folder. We can change it by change value of data_home parameter
The '~' symbol is expanded to the user home folder.

If the folder does not already exist, it is automatically created.

- **Return**: path of the tmlib data dir.

>>> from tmlib.datasets import base
>>> print base.get_data_home()
/home/kde/tmlib_data

-----------------------------
Function base.clear_data_home
-----------------------------

tmlib.datasets.base. **clear_data_home** (data_home=None)

Delete all the content of the data home cache. 

--------------------------------
Function base.check_input_format
--------------------------------

tmlib.datasets.base.check_input_format(*file_path*)

- Check format of input file(text formatted or raw text)
- **Parameters**: file_path (string)

  Path of file input
- **Return**: format of input (DataFormat.RAW_TEXT, DataFormat.TERM_FREQUENCY or DataFormat.TERM_SEQUENCE)

>>> from tmlib.datasets import base
>>> file_path = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> print base.check_input_format(file_path)
tf
>>> file_path = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train_raw.txt'
>>> print base.check_input_format(file_path)
txt

-----------------------------------------
Function base.load_batch_raw_text
-----------------------------------------

tmlib.datasets.base.load_batch_raw_text(*file_raw_text_path*)

- load all of documents and store as a list. Each element in this list is a document with raw text format (string)

- **Parameters**: file_raw_text_path (string)

  Path of file input 
  
- **Return**: list, each element in list is string type and also is text of a document

>>> from tmlib.datasets import base
>>> path_file_raw_text = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_infer_raw.txt'
>>> list_docs = base.load_batch_raw_text(path_file_raw_text)
>>> print 'number of documents: ', len(list_docs)
number of documents:  50
>>> print list_docs[8]
 Here is a summary of developments in forest and brush fires in Western states:

------------------------------------
Function base.pre_process
------------------------------------

tmlib.datasets.base.pre_process(*file_path*)

- Preprocessing for file input if format of data is raw text 
- **Paremeter**: file_path (string)

  Path of file input
- **Return**: list which respectly includes path of vocabulary file, term-frequency file, term-sequence file after preprocessing

>>> from tmlib.datasets import base
>>> path_file = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train_raw.txt'
>>> path_vocab, path_tf, path_sq = base.pre_process(path_file)
Waiting...
>>> print 'path to file vocabulary extracted: ', path_vocab
path to file vocabulary extracted:  /home/kde/tmlib_data/ap_train_raw/vocab.txt
>>> print 'path to file with term-frequency format: ', path_tf
path to file with term-frequency format:  /home/kde/tmlib_data/ap_train_raw/ap_train_raw.tf
>>> print 'path to file with term-sequence format: ', path_sq
path to file with term-sequence format:  /home/kde/tmlib_data/ap_train_raw/ap_train_raw.sq

--------------------------------------------
Function base.load_batch_formatted_from_file
--------------------------------------------

tmlib.datasets.base.load_batch_formatted_from_file(*data_path, output_format=DataFormat.TERM_FREQUENCY*)

- load all of documents in file which is formatted as term-frequency format or term-sequence format and return a corpus with format is **output_format**
- **Parameters**:

  - **data_path**: path of file data input which is formatted
  - **output_format**: format data of output, default: term-frequence format
  
- **Return**: object corpus which is the data input for learning 

>>> path_file_tf = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> corpus_tf = base.load_batch_formatted_from_file(path_file_tf)
>>> print 'Unique terms in the 9th documents: ', corpus_tf.word_ids_tks[8]
Unique terms in the 9th documents:  [5829 4040 2891   14 1783  381 2693]
>>> print 'Frequency of unique terms in the 9th documents: ', corpus_tf.cts_lens[8]
Frequency of unique terms in the 9th documents:  [1 1 1 1 1 1 1]
>>> corpus_sq = base.load_batch_formatted_from_file(path_file_tf, output_format=base.DataFormat.TERM_SEQUENCE)
>>> print 'List of tokens in the 9th documents: ', corpus_sq.word_ids_tks[8]
List of tokens in the 9th documents:  [5829 4040 2891   14 1783  381 2693]
>>> print 'Number of tokens in the 9th document: ', corpus_sq.cts_lens[8]
Number of tokens in the 9th document:  7


-------------------------------------------------------
Function base.reformat_file_to_term_sequence
-------------------------------------------------------

tmlib.datasets.base.reformat_file_to_term_sequence(*file_path*)

- convert the formatted file input (tf or sq) to file with format term-sequence
- **Parameter**: file_path (string)

  Path of file input
- **Return**: path of file which is formatted to term-sequence

>>> from tmlib.datasets import base
>>> path_file_tf = tmlib
>>> path_file_sq = base.reformat_file_to_term_sequence(path_file_tf)
>>> print 'path to file term-sequence: ', path_file_sq
path to file term-sequence:  /home/kde/tmlib_data/ap_train/ap_train.sq

--------------------------------------------------------
Function base.reformat_file_to_term_frequency
--------------------------------------------------------

tmlib.datasets.base.reformat_file_to_term_sequence(*file_path*)

- convert the formatted file input (tf or sq) to file with format term-frequency
- **Parameter**: file_path (string)

  Path of file input
- **Return**: path of file which is formatted to term-frequency

>>> from tmlib.datasets import base
>>> path_file = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> path_file_tf = base.reformat_file_to_term_sequence(path_file)
>>> print 'path to file term-frequency: ', path_file_tf
path to file term-frequency:  /home/kde/tmlib_data/ap_train/ap_train.tf

-----------------------------------
Function base.convert_corpus_format
-----------------------------------

tmlib.datasets.base.convert_corpus_format(*corpus, data_format*)

- convert corpus (object of class Corpus) to desired format
- **Parameters**:

  - **corpus**: object of class Corpus, 
  - **data_format**: format type desired (DataFormat.TERM_SEQUENCE or DataFormat.TERM_FREQUENCY)

- **Return**: object corpus with desired format

>>> from tmlib.datasets import base
>>> path_file_tf = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> corpus = base.load_batch_formatted_from_file(path_file_tf)
>>> corpus_sq = base.convert_corpus_format(corpus, base.DataFormat.TERM_SEQUENCE)
>>> print 'Unique terms in the 22th documents: ', corpus.word_ids_tks[21]
Unique terms in the 22th documents:  [  32  396  246   87  824 3259  316  285]
>>> print 'Frequency of unique terms in the 22th documents: ', corpus.cts_lens[21]
Frequency of unique terms in the 22th documents:  [1 1 1 2 1 1 2 1]
>>> print 'List of tokens in the 22th documents: ', corpus_sq.word_ids_tks[21]
List of tokens in the 22th documents:  [32, 396, 246, 87, 87, 824, 3259, 316, 316, 285]
>>> print 'Number of tokens in the 22th document: ', corpus_sq.cts_lens[21]
Number of tokens in the 22th document:  10


--------------------------------------------------------------
Function base.load_mini_batch_term_sequence_from_sequence_file
--------------------------------------------------------------

tmlib.datasets.base.load_mini_batch_term_sequence_from_sequence_file(*fp, batch_size*)

- loading a mini-batch with size: **batch_size** from a file which has the file pointer **fp**. This file includes the documents with term-sequence format and the loaded mini-batch is also term-sequence format
- **Parameter**:

  - **fp**: file pointer of file term-sequence format
  - **batch_size**: int, size of mini-batch
- **Return**: *(minibatch, end_file)*. *minibatch* is object of class Corpus with term-sequence format and *end_file* is boolean variable which check that file pointer is at end of file or not

>>> from tmlib.datasets import base
>>> path_file_tf = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> path_file_sq = base.reformat_file_to_term_sequence(path_file_tf)
>>> fp = open(path_file_sq)
>>> mini_batch, end_file = base.load_mini_batch_term_sequence_from_sequence_file(fp, 1500)
>>> print 'Format of mini_batch: ', mini_batch.format_type
Format of mini_batch:  sq
>>> print 'List of tokens in the first document of mini_batch: \n', mini_batch.word_ids_tks[0]
List of tokens in the first document of mini_batch: 
[    0  6144  3586  3586     3     4  1541     8    10  3927    12    12
    12    12    12    12    12  4621   527  9232  1112  1112    20  2587
  6172 10269 10269    37    42  3117  1582  1585  1585  1585   435  9268
  9268  9268   571   571    60    61    63    63    64    64  5185    11
  4683   590   590  1103  1103   592  5718  1623  1623  1624  1624  1624
  1624    89    89  6234  8802  1638   103   600  9404   106  3691   720
  2672   113  2165  5751   123   123   123  1148   128   128  1670  1670
  4231  1167   144   147   149   149   149   149   149   149   149  3735
  3735  5272  5272  1732   673   673  5282    27  1700  9893  9893   166
   167   173   174  2224  2248   372   372   186  4284  4284  4284  3450
  3450   117   117   203  2244  5320   201  4215  9932  9932   207   207
   208   208   208   208   208  8914  7898   733   733  1760  1744   744
   234  1259  1259  4287  7254   249  8311  5884  5884   298   254   767
   767  2304  4876   270   557   786   789   789  2331   287  5409   290
  5923  2854  1834  1834  1834   303  3888  3888  3888  3888   817   817
  9523   334  1333   311   311  1855  1417   325  1870  1870  1870  1870
  1870  1870  1870  1361  1362  6995   342   343   344   857  5469  5469
   351   351   351   351   351  1377  2402   487   884   885   890  4477
  3455  1410  5099  4489   395  2570   152   404  1429  1430  3992   416
  3491  2033  3499   429  3502  5040   433   433  1971  1971  1971  1971
   437  9667  9667   322  7119  8656  1102   985   989  1840  2529   997
  2022  2022  4071  2536 10219  1517  1009   221  3059   500   511]
>>> print 'Number of tokens in the first document of mini_batch: \n', mini_batch.cts_lens[0]
Number of tokens in the first document of mini_batch: 
263


--------------------------------------------------------------------
Function base.load_mini_batch_term_sequence_from_term_frequency_file
--------------------------------------------------------------------

tmlib.datasets.base.load_mini_batch_term_sequence_from_term_frequency_file(*fp, batch_size*)

- loading a mini-batch with size: **batch_size** from a file which has the file pointer **fp**. This file includes the documents with term-frequency format and the returned mini-batch is term-sequence format
- **Parameter**:

  - **fp**: file pointer of file term-frequency format
  - **batch_size**: int, size of mini-batch
- **Return**: *(minibatch, end_file)*. *minibatch* is object of class Corpus with term-sequence format and *end_file* is boolean variable which check that file pointer is at end of file or not

>>> from tmlib.datasets import base
>>> path_file_tf = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> fp = open(path_file_tf)
>>> mini_batch1, end_file = base.load_mini_batch_term_sequence_from_term_frequency_file(fp, 1500)
>>> print 'Format of mini_batch1: ', mini_batch1.format_type
Format of mini_batch1:  sq
>>> print 'Size of mini_batch1: ', len(mini_batch1.word_ids_tks)
Size of mini_batch1:  1500
>>> print 'End file: ', end_file
End file:  False
>>> mini_batch2, end_file = base.load_mini_batch_term_sequence_from_term_frequency_file(fp, 1500)
>>> print 'Size of mini_batch2: ', len(mini_batch2.word_ids_tks)
Size of mini_batch2:  700
>>> print 'End file: ', end_file
End file:  True


---------------------------------------------------------------
Function base.load_mini_batch_term_frequency_from_sequence_file
---------------------------------------------------------------

tmlib.datasets.base.load_mini_batch_term_frequency_from_sequence_file(*fp, batch_size*)

- loading a mini-batch with size: **batch_size** from a file which has the file pointer **fp**. This file includes the documents with term-sequence format and the returned mini-batch is term-frequency format
- **Parameter**:

  - **fp**: file pointer of file term-sequence format
  - **batch_size**: int, size of mini-batch
- **Return**: *(minibatch, end_file)*. *minibatch* is object of class Corpus with term-frequency format and *end_file* is boolean variable which check that file pointer is at end of file or not

>>> from tmlib.datasets import base
>>> path_file_tf = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> path_file_sq = base.reformat_file_to_term_sequence(path_file_tf)
>>> fp = open(path_file_sq)
>>> mini_batch, end_file = base.load_mini_batch_term_frequency_from_sequence_file(fp, 1500)
>>> print 'Format of mini_batch: ', mini_batch.format_type
Format of mini_batch:  tf
>>> print 'Unique terms in the first document of mini_batch: \n', mini_batch.word_ids_tks[0]
Unique terms in the first document of mini_batch: 
[    0  6144  3586     3     4  1541     8    10    11    12  4621   527
  9232  1112    20  2587  6172 10269    37    42  3117  1582  1585  1971
  9268   571    60    61    63    64  5185   343  4683   590  1103   592
  5718  1623  1624    89  6234  8802  1638   103   600  9404   106  3691
   890  2672   113  2165  4215   123  1148   128  1670  4231  1167   144
   147   149  3735  5272  1732  1744  4489   673  5282    27  1700  9893
   166   167  5751   173   174  2224  2248   884   186  4284   117  2244
  5320   201   203  9932   207   720  8914  7898   733  1760   208   744
   234  1259  4287   249  8311  7254  1834   254   767  2304  4876   270
   557   786   789  2331   287  5409   290  5923  2854   298   303  3888
   817  9523  1333   311  1855   322   325  1102   334  1361  1362  6995
   342  3927   344   857  5469   351  1377  2402  4071   372   885  3450
  4477  3455  1410 10219  1417   395  2570   152   404  1429  1430  3992
   416  3491  1009  3499   429  3502  5040   433   435   437  9667  7119
  8656  1870   985   989  1840  2529   997  2022   487  2536  5884  5099
  1517  2033   221  3059   500   511]
>>> print 'Frequency of unique terms in the first document of mini_batch: \n', mini_batch.cts_lens[0]
Frequency of unique terms in the first document of mini_batch: 
[1 1 2 1 1 1 1 1 1 7 1 1 1 2 1 1 1 2 1 1 1 1 3 4 3 2 1 1 2 2 1 1 1 2 2 1 1
 2 4 2 1 1 1 1 1 1 1 1 1 1 1 1 1 3 1 2 2 1 1 1 1 7 2 2 1 1 1 2 1 1 1 2 1 1
 1 1 1 1 1 1 1 3 2 1 1 1 1 2 2 1 1 1 2 1 5 1 1 2 1 1 1 1 3 1 2 1 1 1 1 1 2
 1 1 1 1 1 1 1 1 4 2 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 2 5 1 1 1 2 1 2 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 2 1 1 7 1 1 1 1 1 2 1 1 2 1 1 1 1 1 1
 1]

---------------------------------------------------------------------
Function base.load_mini_batch_term_frequency_from_term_frequency_file
---------------------------------------------------------------------

- loading a mini-batch with size: **batch_size** from a file which has the file pointer **fp**. This file includes the documents with term-frequency format and the returned mini-batch is also term-frequency format
- **Parameter**:

  - **fp**: file pointer of file term-frequency format
  - **batch_size**: int, size of mini-batch
- **Return**: *(minibatch, end_file)*. *minibatch* is object of class Corpus with term-frequency format and *end_file* is boolean variable which check that file pointer is at end of file or not

>>> from tmlib.datasets import base
>>> path_file_tf = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> fp = open(path_file_tf)
>>> mini_batch, end_file = base.load_mini_batch_term_frequency_from_term_frequency_file(fp, 1500)
>>> print 'Format of mini_batch: ', mini_batch.format_type
Format of mini_batch:  tf
>>> print 'Unique terms in the first document of mini_batch: \n', mini_batch.word_ids_tks[0]
Unique terms in the first document of mini_batch: 
[    0  6144  3586     3     4  1541     8    10  3927    12  4621   527
  9232  1112    20  2587  6172 10269    37    42  3117  1582  1585   435
  9268   571    60    61    63    64  5185    11  4683   590  1103   592
  5718  1623  1624    89  6234  8802  1638   103   600  9404   106  3691
   720  2672   113  2165  5751   123  1148   128  1670  4231  1167   144
   147   149  3735  5272  1732   673  5282    27  1700  9893   166   167
   173   174  2224  2248   372   186  4284  3450   117   203  2244  5320
   201  4215  9932   207   208  8914  7898   733  1760  1744   744   234
  1259  4287  7254   249  8311  5884   298   254   767  2304  4876   270
   557   786   789  2331   287  5409   290  5923  2854  1834   303  3888
   817  9523   334  1333   311  1855  1417   325  1870  1361  1362  6995
   342   343   344   857  5469   351  1377  2402   487   884   885   890
  4477  3455  1410  5099  4489   395  2570   152   404  1429  1430  3992
   416  3491  2033  3499   429  3502  5040   433  1971   437  9667   322
  7119  8656  1102   985   989  1840  2529   997  2022  4071  2536 10219
  1517  1009   221  3059   500   511]
>>> print 'Frequency of unique terms in the first document of mini_batch: \n', mini_batch.cts_lens[0]
Frequency of unique terms in the first document of mini_batch: 
[1 1 2 1 1 1 1 1 1 7 1 1 1 2 1 1 1 2 1 1 1 1 3 1 3 2 1 1 2 2 1 1 1 2 2 1 1
 2 4 2 1 1 1 1 1 1 1 1 1 1 1 1 1 3 1 2 2 1 1 1 1 7 2 2 1 2 1 1 1 2 1 1 1 1
 1 1 2 1 3 2 2 1 1 1 1 1 2 2 5 1 1 2 1 1 1 1 2 1 1 1 1 2 1 1 2 1 1 1 1 1 2
 1 1 1 1 1 1 3 1 4 2 1 1 1 2 1 1 1 7 1 1 1 1 1 1 1 2 5 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 4 1 2 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1
 1]


-----------------------------------------------------------
Function shuffle_formatted_data_file
-----------------------------------------------------------

tmlib.datasets.base.shuffle_formatted_data_file(*data_path, batch_size*)

- Random permutation of all documents in file input. Because the learning methods are stochastic, so this fuction help sample randomly mini-batch from corpus. And after shuffling, the documents with new position will be written to a new file.
- **Parameter**:

  - **data_path**: file input which is formatted (tf or sq)
  - **batch_size**: the necessary parameter for the shuffling algorithm designed by us

- **Return**: path of new file which is shuffled

>>> from tmlib.datasets import base
>>> path_file = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt'
>>> path_file_shuffled = base.shuffle_formatted_data_file(path_file, 500)
>>> print 'Path to file shuffled: ', path_file_shuffled
Path to file shuffled:  /home/kde/Desktop/topicmodel-lib/examples/ap/ap_train.txt.shuffled

------------------------------
Function base.compute_sparsity
------------------------------

tmlib.datasets.base.compute_sparsity(*doc_tp, num_docs, num_topics, _type*)

- Compute document sparsity.
- **Parameters**:

  - **doc_tp**: numpy.array, 2-dimention, the estimated topic mixtures of all documents in corpus
  - **num_docs**: int, the number of documents in corpus
  - **num_topics**: int, is the number of requested latent topics to be extracted from the training corpus.
  - **_type**: string, if the value is 'z', the topic mixtures is estimated by the sampling method as CGS or CVB0, so we have the individual caculation for this. Otherwise, if the value of it isn't 'z', this is for the methods as: VB, OPE or FW
  
- **Return**: float, sparsity of documents

>>> import numpy as np
>>> from tmlib.datasets import base
>>> theta = np.array([[0.1, 0.3, 0.2, 0.2, 0.1, 0.1], [0.02, 0.05, 0.03, 0.5, 0.2, 0.2]], dtype='float32')
>>> base.compute_sparsity(theta, theta.shape[0], theta.shape[1], _type='t')
1.0

----------------------------------
Function base.write_topic_mixtures 
----------------------------------

tmlib.datasets.base.write_topic_mixtures(*theta, file_name*)

- save topic mixtures (theta) to a file
- **Parameters**:

  - **theta**: numpy.array, 2-dimention
  - **file_name**: name (path) of file which is written

-------------------------------
Function base.read_vocab
-------------------------------

tmlib.datasets.base. **read_vocab** (*path_vocab*)

- Read file vocabulary and store it with dictionary type of python (for example: the word 'hello' is the 2nd word (index = 2) in file vocabulary, this function return a object named *dict*, we have: dict['hello'] = 2), it's used as input parameter for function **parse_doc_list**

- **Parameters**:
  
  - **path_vocab**: path of file vocabulary 
  
>>> from tmlib.datasets import base
>>> path_vocab = '/home/kde/Desktop/topicmodel-lib/examples/ap/vocab.txt'
>>> list_unique_terms = open(path_vocab).readlines()
>>> list_unique_terms[10].strip()
'years'
>>> dict_vocab = base.read_vocab(path_vocab)
>>> dict_vocab['years']
10
>>> list_unique_terms[1021].strip()
'laws'
>>> dict_vocab['laws']
1021

-----------------------------------
Function base.parse_doc_list
-----------------------------------

tmlib.datasets.base. **parse_doc_list** (*docs, vocab_dict*)

- **Parameters**:

  - **docs**: list of document. Each document is represented as a string of words (same as raw text)
  - **vocab_dict**: vocabulary is represented as dictionary format described above

- **Return**: object of class Corpus

>>> path_vocab = '/home/kde/Desktop/topicmodel-lib/examples/ap/vocab.txt'
>>> dict_vocab = base.read_vocab(path_vocab)
>>> path_file_raw_text = '/home/kde/Desktop/topicmodel-lib/examples/ap/ap_infer_raw.txt'
>>> list_docs = base.load_batch_raw_text(path_file_raw_text)
>>> print 'The 9th document has the content: "%s"' %list_docs[8]
The 9th document has the content: " Here is a summary of developments in forest and brush fires in Western states:"
>>> corpus_tf = base.parse_doc_list(list_docs, dict_vocab)
>>> list_unique_terms = open(path_vocab).readlines()
>>> list_terms_in_doc_9th = list()
>>> for index in corpus_tf.word_ids_tks[8]: \
...     list_terms_in_doc_9th.append(list_unique_terms[index].strip())
... 
>>> print 'List of unique terms in the 9th document: ', '\nindexs: ', corpus_tf.word_ids_tks[8], '\nwords: ', list_terms_in_doc_9th 
List of unique terms in the 9th document:  
indexs:  [5829, 4040, 2891, 14, 1783, 381, 2693] 
words:  ['summary', 'brush', 'fires', 'states', 'forest', 'western', 'developments']
>>> print 'Frequency of unique terms: ', corpus_tf.cts_lens[8]
Frequency of unique terms:  [1, 1, 1, 1, 1, 1, 1]
