.. -*- coding: utf-8 -*-

=================================
tmlib.datasets.wiki_stream
=================================
This module helps loading minibatches from wikipedia website for learning LDA

.. Contents::


-----------------------------------------------------
class tmlib.datasets.wiki_stream.WikiStream
-----------------------------------------------------

tmlib.datasets.wiki_stream.WikiStream(*batch_size, num_batch, save_into_file=False, path_vocab=None*)

Parameters
===========

- **batch_size**: int
  
  size of mini-batch in each sampling randomly ariticles from wikipedia website. 

- **num_batch**: int,

  This is the number of iterations for training LDA

- **save_into_file**: boolean, default: False

  Save the aritiles which is downloaded and analyzed into a file with format tf or sq

- **path_vocab**: string, default: None
  
  File vocabulary of articles on website. By default, we provided a file vocabulary (https://github.com/blei-lab/onlineldavb/blob/master/dictnostops.txt). This file has path "../tmlib/datasets/data/wikipedia/vocab.txt"

Attributes
==========

- **batch_size**: int
- **num_batch**: int

- **output_format**: default is DataFormat.TERM_FREQUENCY
  
  format of mini-batch. User change the format by use method *set_output_format*

- **vocab**: file vocabulary is save as dictionary format

  For examle: vocab['online'] = 10. That means term 'online' is at 10th position (start from 0) in vocabulary file

- **end_of_data** (inheritance attribute)

- **mini_batch_no** (inheritance attribute)

Methods
=======

- __init__(*data_path, batch_size, passes=1, shuffle_every=None, vocab_file=None*)
- **load_mini_batch** ()

  loading a mini-batch from corpus with specific format (controlled by **output_format**)

- **check_end_of_data** ()

  Return True if the last minibatch is loaded

- **set_output_format** (output_format)

  set format for the loaded mini-batch

  - **Parameters**: output_format (DataFormat.TERM_SEQUENCE or DataFormat.TERM_FREQUENCY)

-------------------------------
Function wiki_stream.read_vocab
-------------------------------

tmlib.datasets.wiki_stream. **read_vocab** (*path_vocab*)

Return a dictionary as described with **vocab** attribute above, it's used as input parameter for function **parse_doc_list**

-----------------------------------
Function wiki_stream.parse_doc_list
-----------------------------------

tmlib.datasets.wiki_stream. **parse_doc_list** (*docs, vocab_dict*)

- **Parameters**:

  - **docs**: list of document. Each document is represented as a string of words (same as raw text)
  - **vocab_dict**: vocabulary is represented as dictionary format described above

- **Return**: object of class Corpus
