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

-----------------------------------------
Function base.load_batch_raw_text
-----------------------------------------

tmlib.datasets.base.load_batch_raw_text(*file_raw_text_path*)

- load all of documents and store as a list. Each element in this list is a document with raw text format (string) , this function usually is used for inference new documents

- **Parameters**: file_raw_text_path (string)

  Path of file input 
  
- **Return**: list, each element in list is string type and also is text of a document

------------------------------------
Function base.pre_process
------------------------------------

tmlib.datasets.base.pre_process(*file_path*)

- Preprocessing for file input if format of data is raw text 
- **Paremeter**: file_path (string)

  Path of file input
- **Return**: list which respectly includes path of vocabulary file, term-frequency file, term-sequence file after preprocessing

-------------------------------------------------------
Function base.reformat_file_to_term_sequence
-------------------------------------------------------

tmlib.datasets.base.reformat_file_to_term_sequence(*file_path*)

- convert the formatted file input (tf or sq) to file with format term-sequence
- **Parameter**: file_path (string)

  Path of file input
- **Return**: path of file which is formatted to term-sequence

--------------------------------------------------------
Function base.reformat_file_to_term_frequency
--------------------------------------------------------

tmlib.datasets.base.reformat_file_to_term_sequence(*file_path*)

- convert the formatted file input (tf or sq) to file with format term-frequency
- **Parameter**: file_path (string)

  Path of file input
- **Return**: path of file which is formatted to term-frequency

-----------------------------------
Function base.convert_corpus_format
-----------------------------------

tmlib.datasets.base.convert_corpus_format(*corpus, data_format*)

- convert corpus (object of class Corpus) to desired format
- **Parameters**:

  - **corpus**: object of class Corpus, 
  - **data_format**: format type desired (DataFormat.TERM_SEQUENCE or DataFormat.TERM_FREQUENCY)

- **Return**: object corpus with desired format

--------------------------------------------
Function base.load_batch_formatted_from_file
--------------------------------------------

tmlib.datasets.base.load_batch_formatted_from_file(*data_path, output_format=DataFormat.TERM_FREQUENCY*)

- load all of documents in file which is formatted as term-frequency format or term-sequence format and return a corpus with format is **output_format**
- **Parameters**:

  - **data_path**: path of file data input which is formatted
  - **output_format**: format data of output, default: term-frequence format
  
- **Return**: object corpus which is the data input for learning 


--------------------------------------------------------------
Function base.load_mini_batch_term_sequence_from_sequence_file
--------------------------------------------------------------

tmlib.datasets.base.load_mini_batch_term_sequence_from_sequence_file(*fp, batch_size*)

- loading a mini-batch with size **batch_size** from a file which has the file pointer **fp**. This file is term-sequence format and the loaded mini-batch is also term-sequence format
- **Parameter**:

  - **fp**: file pointer of file term-sequence format
  - **batch_size**: int, size of mini-batch
- **Return**: *(minibatch, end_file)*. *minibatch* is object of class Corpus with term-sequence format and *end_file* is boolean variable which check that file pointer is at end of file or not

--------------------------------------------------------------------
Function base.load_mini_batch_term_sequence_from_term_frequency_file
--------------------------------------------------------------------

tmlib.datasets.base.load_mini_batch_term_sequence_from_term_frequency_file(*fp, batch_size*)

- loading a mini-batch with size: **batch_size** from a file which has the file pointer **fp**. This file is term-sequence format and the loaded mini-batch is also term-sequence format
- **Parameter**:

  - **fp**: file pointer of file term-sequence format
  - **batch_size**: int, size of mini-batch
- **Return**: *(minibatch, end_file)*. *minibatch* is object of class Corpus with term-sequence format and *end_file* is boolean variable which check that file pointer is at end of file or not


---------------------------------------------------------------
Function base.load_mini_batch_term_frequency_from_sequence_file
---------------------------------------------------------------
Similar

---------------------------------------------------------------------
Function base.load_mini_batch_term_frequency_from_term_frequency_file
---------------------------------------------------------------------
Similar

-----------------------------------------------------------
Function shuffle_formatted_data_file
-----------------------------------------------------------

tmlib.datasets.base.shuffle_formatted_data_file(*data_path, batch_size*)

- shuffle file input and write to new file
- **Parameter**:

  - **data_path**: file input which is formatted (tf or sq)
  - **batch_size**: the necessary parameter for shuffling algorithm designed by us

- **Return**: path of new file which is shuffled

------------------------------
Function base.compute_sparsity
------------------------------

tmlib.datasets.base.compute_sparsity(*doc_tp, batch_size, num_topics, _type*)

- Compute document sparsity.
- **Parameters**:

  - **doc_tp**: 
  - **batch_size**:
  - **num_topics**:
  - **_type**: 
- **Return**: float, sparsity of documents

----------------------------------
Function base.write_topic_mixtures 
----------------------------------

tmlib.datasets.base.write_topic_mixtures(*theta, file_name*)

- write theta to a file
- **Parameters**:

  - **theta**: numpy.array, 2-dimention
  - **file_name**: name (path) of file which is written

-------------------------------
Function base.read_vocab
-------------------------------

tmlib.datasets.base. **read_vocab** (*path_vocab*)

Return a dictionary as described with **vocab** attribute above, it's used as input parameter for function **parse_doc_list**

-----------------------------------
Function base.parse_doc_list
-----------------------------------

tmlib.datasets.base. **parse_doc_list** (*docs, vocab_dict*)

- **Parameters**:

  - **docs**: list of document. Each document is represented as a string of words (same as raw text)
  - **vocab_dict**: vocabulary is represented as dictionary format described above

- **Return**: object of class Corpus
