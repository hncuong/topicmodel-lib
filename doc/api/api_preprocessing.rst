.. -*- coding: utf-8 -*-

=================================
tmlib.preprocessing.preprocessing
=================================
This is module work with text data which needed to preprocess.

Please refer [1]_ to know more detail about preprocessing technique

.. Contents::


-----------------------------------------------------
class tmlib.preprocessing.preprocessing.PreProcessing
-----------------------------------------------------

tmlib.preprocessing.preprocessing.PreProcessing(*file_path, stemmed=False, remove_rare_word=3, remove_common_word=None*)

Parameters
===========

- **file_path**: string, not default 

  Path of file corpus which has raw text format.
- **stemmed**: boolean, default: False
  
  Appply the stemming algorithm (Porter, 1980) to preprocess text data. The algorithm is applied when parameter is set value *True*

- **remove_rare_word**: int, default: 3

  Removing rarely words in the documents. Default, words which appeared in less 3 documents will be removed

- **remove_common_word**: int, default: None

  Removing common words (which appeared in many documents). Default, words which appeared in greater a half documents of corpus will be removed.

Attributes
==========

- **path_file_vocab**: string,

  path of the vocabulary file which created after calling method *extract_vocab()*

- **path_file_tf**: string,
 
  path of file corpus with term-frequency format. This file is created after calling method *save_format_tf()*

- **path_file_sq**: string,
 
  path of file corpus with term-sequence format. This file is created after calling method *save_format_sq()*

Methods
=======

- __init__(*file_path, stemmed=False, remove_rare_word=3, remove_common_word=None*)
- **process()**

  run the preprocessing algorithms

- **extract_vocab** (folder=None)

  Extracting to the file vocabulary of corpus after preprocessing
  
  - **Parameters**: folder (string, default: None)

    The position which file vocabulary is saved. By default, file is saved in a folder with path  *<user home folder> + "tmlib_data/" + <name of file input>* 

- **save_format_sq** (folder=None)

  Extracting to the file corpus with term-sequence format

  - **Parameters**: folder (string, default: None)

    The position which file term-sequence is saved. By default, file is saved in a same folder with file vocablary created above

- **save_format_tf** (folder=None)

  Extracting to the file corpus with term-frequency format

  - **Parameters**: folder (string, default: None)

    The position which file term-frequency is saved. By default, file is saved in a same folder with file vocablary created above

.. [1] Care and Feeding of Topic Models: Problems, Diagnostics, and Improvements. Jordan Boyd Graber, David Mimno, and David Newman. In Handbook of Mixed Membership Models and Their Applications, CRC/Chapman Hall, 2014.
