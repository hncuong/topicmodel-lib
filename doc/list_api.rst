======
Module
======

`tmlib.preprocessing.preprocessing`_
------------------------------------

.. _tmlib.preprocessing.preprocessing: api/api_preprocessing.rst

- class **tmlib.preprocessing.preprocessing.PreProcessing**

`tmlib.datasets.base`_
----------------------

.. _tmlib.datasets.base: api/api_base.rst

- **Base classes**:

  - class **tmlib.datasets.base.DataFormat**

  - class **tmlib.datasets.base.Corpus**

  - class **tmlib.datasets.base.DataIterator**

- **Base functions**:

  - Function **base.get_data_home**

  - Function **base.clear_data_home**

  - Function **base.check_input_format**
  
  - Function **base.get_list_docs_raw_text**

  - Function **base.pre_process**

  - Function **base.reformat_file_to_term_sequence**

  - Function **base.reformat_file_to_term_frequency**

  - Function **base.convert_corpus_format**

  - Function **base.load_mini_batch_term_sequence_from_sequence_file**

  - Function **base.load_mini_batch_term_sequence_from_term_frequency_file**

  - Function **base.load_mini_batch_term_frequency_from_sequence_file**

  - Function **base.load_mini_batch_term_frequency_from_term_frequency_file**

  - Function **shuffle_formatted_data_file**

  - Function **base.compute_sparsity**

  - Function **base.write_topic_mixtures**
  
  - Function **base.read_vocab**

  - Function **base.parse_doc_list**
  

`tmlib.datasets.dataset`_
-------------------------

.. _tmlib.datasets.dataset: api/api_dataset.rst

- class **tmlib.datasets.dataset.Dataset**

`tmlib.datasets.wiki_stream`_
-----------------------------

.. _tmlib.datasets.wiki_stream: ..api/api_wiki.rst

- class **tmlib.datasets.wiki_stream.WikiStream**


`tmlib.lda`_
------------

.. _tmlib.lda: api/api_lda.rst

- class **tmlib.lda.ldamodel.LdaModel**

- class **tmlib.lda.ldalearning.LdaStatistics**

- class **tmlib.lda.ldalearning.LdaLearning**

- class **tmlib.lda.Online_VB.OnlineVB**

- class **tmlib.lda.Online_CVB0.OnlineCVB0**

- class **tmlib.lda.Online_CGS.OnlineCGS**

- class **tmlib.lda.Online_FW.OnlineFW**

- class **tmlib.lda.Online_OPE.OnlineOPE**

- class **tmlib.lda.Streaming_VB.StreamingVB**

- class **tmlib.lda.Streaming_FW.StreamingFW**


- class **tmlib.lda.Streaming_OPE.StreamingOPE**

- class **tmlib.lda.ML_CGS.MLCGS**

- class **tmlib.lda.ML_FW.MLFW**

- class **tmlib.lda.ML_OPE.MLOPE**

