.. -*- coding: utf-8 -*-

=================================
tmlib.datasets.dataset
=================================
This module helps loading minibatches from corpus for learning LDA

.. Contents::


-----------------------------------------------------
class tmlib.datasets.dataset.Dataset
-----------------------------------------------------

tmlib.datasets.dataset.Dataset(*data_path, batch_size, passes=1, shuffle_every=None, vocab_file=None*)

Parameters
===========

- **data_path**: string, not default 

  Path of file input (corpus)
- **batch_size**: int
  
  size of mini-batch in each sampling from corpus. 

- **passes**: int, default: 1

  passes controls how often we train the model on the entire corpus. Another word for passes might be "epochs". iterations is somewhat technical, but essentially it controls how often we repeat a particular loop over each document. It is important to set the number of "passes" and "iterations" high enough.
  
  For example, if you set passes = 5, assume that batch_size = 100 and size of corpus is 10000 then number of training iterations is 10000/100*5 = 5000

- **shuffle_every**: int, default: None

  Because of sampling randomly from corpus, so minibatches in each pass time must be sampled pseudo-randomly. By the way shuffling corpus, we can create the randomize of sampling minibatch.

  If you set shuffle_every=2, it means after passing over corpus 2 times, corpus will be shuffled

- **vocab_file**: string, default: None
  
  File vocabulary of corpus
  
  If corpus is raw text format, file vocabulary is non-necessary. Otherwise, if corpus is tf or sq format, user must set it

Attributes
==========

- **batch_size**: int
- **vocab_file**: string,
- **data_path**: string, path of formatted file corpus
- **data_format**: DataFormat.TERM_SEQUENCE or DataFormat.TERM_FREQUENCY
- **output_format**: default is DataFormat.TERM_FREQUENCY
  
  format of mini-batch. User change the format by use method *set_output_format*

- **passes**: int
- **shuffle_every**: int
- **pass_no**: int

  numerical order of passes

- **batch_no_in_pass**: int

  numerical order of minibatches in each pass

- **work_path**: string

  This path is different from data_path. If corpus is shuffled then work_path is path of the shuffled file, not the original file

- **fp**: file pointer of file **work_path**

- **end_of_file**: boolean,

- **end_of_data** (inheritance attribute)

- **mini_batch_no** (inheritance attribute)

Methods
=======

- __init__(*data_path, batch_size, passes=1, shuffle_every=None, vocab_file=None*)
- **load_mini_batch** ()

  loading a mini-batch from corpus with specific format (controlled by **output_format**)

- **check_end_of_data** ()

  Return True if the pass is the last pass and loading to end of file

- **set_output_format** (output_format)

  set format for the loaded mini-batch

  - **Parameters**: output_format (DataFormat.TERM_SEQUENCE or DataFormat.TERM_FREQUENCY)


- **get_total_docs** ()

  get number of documents which is analyzed

- **get_num_tokens** ()

  get number of words of corpus 

- **get_num_terms** ()

  get number of terms of vocabulary
