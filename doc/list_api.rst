======
Module
======

`tmlib.preprocessing.preprocessing`_
------------------------------------

.. _tmlib.preprocessing.preprocessing: api/api_preprocessing.rst

- class `tmlib.preprocessing.preprocessing.PreProcessing`_

.. _tmlib.preprocessing.preprocessing.PreProcessing: api/api_preprocessing.rst#class-tmlib-preprocessing-preprocessing-preprocessing

`tmlib.datasets.base`_
----------------------

.. _tmlib.datasets.base: api/api_base.rst

- **Base classes**:

  - class `tmlib.datasets.base.DataFormat`_

  - class `tmlib.datasets.base.Corpus`_

  - class `tmlib.datasets.base.DataIterator`_

- **Base functions**:

  - Function `base.get_data_home`_

  - Function `base.clear_data_home`_

  - Function `base.check_input_format`_
  
  - Function `base.load_batch_raw_text`_

  - Function `base.pre_process`_

  - Function `base.reformat_file_to_term_sequence`_

  - Function `base.reformat_file_to_term_frequency`_

  - Function `base.convert_corpus_format`_
  
  - Function `base.load_batch_formatted_from_file`_

  - Function `base.load_mini_batch_term_sequence_from_sequence_file`_

  - Function `base.load_mini_batch_term_sequence_from_term_frequency_file`_

  - Function `base.load_mini_batch_term_frequency_from_sequence_file`_

  - Function `base.load_mini_batch_term_frequency_from_term_frequency_file`_

  - Function `shuffle_formatted_data_file`_

  - Function `base.compute_sparsity`_

  - Function `base.write_topic_mixtures`_
  
  - Function `base.read_vocab`_

  - Function `base.parse_doc_list`_
  
.. _tmlib.datasets.base.DataFormat: api/api_base.rst#class-tmlib-datasets-base-dataformat

.. _tmlib.datasets.base.Corpus: api/api_base.rst#class-tmlib-datasets-base-corpus

.. _tmlib.datasets.base.DataIterator: api/api_base.rst#class-tmlib-datasets-base-dataiterator

.. _base.get_data_home: api/api_base.rst#function-base-get-data-home

.. _base.clear_data_home: api/api_base.rst#function-base-clear-data-home

.. _base.check_input_format: api/api_base.rst#function-base-check-input-format
  
.. _base.load_batch_raw_text: api/api_base.rst#function-base-load-batch-raw-text

.. _base.pre_process: api/api_base.rst#function-base-pre-process

.. _base.reformat_file_to_term_sequence: api/api_base.rst#function-base-format-reformat-file-to-term-sequence

.. _base.reformat_file_to_term_frequency: api/api_base.rst#function-base-reformat-file-to-term-frequency

.. _base.convert_corpus_format: api/api_base.rst#function-base-convert-corpus-format
  
.. _base.load_batch_formatted_from_file: api/api_base.rst#function-base-load-batch-formatted-from-file

.. _base.load_mini_batch_term_sequence_from_sequence_file: api/api_base.rst#function-base-load-mini-batch-term-sequence-from-sequence-file

.. _base.load_mini_batch_term_sequence_from_term_frequency_file: api/api_base.rst#function-base-load-mini-batch-term-sequence-from-term-frequency-file

.. _base.load_mini_batch_term_frequency_from_sequence_file: api/api_base.rst#function-base-load-mini-batch-term-frequency-from-sequence-file

.. _base.load_mini_batch_term_frequency_from_term_frequency_file: api/api_base.rst#function-base-load-mini-batch-term-frequency-from-term-frequency-file

.. _base.shuffle_formatted_data_file: api/api_base.rst#function-base-shuffle-formatted-data-file

.. _base.compute_sparsity: api/api_base.rst#function-base-compute-sparsity

.. _base.write_topic_mixtures: api/api_base.rst#function-base-write-topic-mixtures
  
.. _base.read_vocab: api/api_base.rst#function-base-read-vocab

.. _base.parse_doc_list: api/api_base.rst#function-base-parse-doc-list

`tmlib.datasets.dataset`_
-------------------------

.. _tmlib.datasets.dataset: api/api_dataset.rst

- class `tmlib.datasets.dataset.DataSet`_

.. _tmlib.datasets.dataset.DataSet: api/api_dataset.rst#class-tmlib-datasets-dataset-DataSet

`tmlib.datasets.wiki_stream`_
-----------------------------

.. _tmlib.datasets.wiki_stream: api/api_wiki.rst

- class `tmlib.datasets.wiki_stream.WikiStream`_

.. _tmlib.datasets.wiki_stream.WikiStream: api/api_wiki.rst#class-tmlib-datasets-wiki-stream-wikistream


`tmlib.lda`_
------------

.. _tmlib.lda: api/api_lda.rst

- class `tmlib.lda.ldamodel.LdaModel`_

- class `tmlib.lda.ldalearning.LdaStatistics`_

- class `tmlib.lda.ldalearning.LdaLearning`_

- class `tmlib.lda.Online_VB.OnlineVB`_

- class `tmlib.lda.Online_CVB0.OnlineCVB0`_

- class `tmlib.lda.Online_CGS.OnlineCGS`_

- class `tmlib.lda.Online_FW.OnlineFW`_

- class `tmlib.lda.Online_OPE.OnlineOPE`_

- class `tmlib.lda.Streaming_VB.StreamingVB`_

- class `tmlib.lda.Streaming_FW.StreamingFW`_


- class `tmlib.lda.Streaming_OPE.StreamingOPE`_

- class `tmlib.lda.ML_CGS.MLCGS`_

- class `tmlib.lda.ML_FW.MLFW`_

- class `tmlib.lda.ML_OPE.MLOPE`_

.. _tmlib.lda.ldamodel.LdaModel: api/api_lda.rst#class-tmlib-lda-ldamodel-ldamodel

.. _tmlib.lda.ldalearning.LdaStatistics: api/api_lda.rst#class-tmlib-lda-ldalearning-ldastatistics

.. _tmlib.lda.ldalearning.LdaLearning: api/api_lda.rst#class-tmlib-lda-ldalearning-ldalearning

.. _tmlib.lda.Online_VB.OnlineVB: api/api_lda.rst#class-tmlib-lda-online-vb-onlinevb

.. _tmlib.lda.Online_CVB0.OnlineCVB0: api/api_lda.rst#class-tmlib-lda-online-cvb0-onlinecvb0

.. _tmlib.lda.Online_CGS.OnlineCGS: api/api_lda.rst#class-tmlib-lda-online-cgs-onlinecgs

.. _tmlib.lda.Online_FW.OnlineFW: api/api_lda.rst#class-tmlib-lda-online-fw-onlinefw

.. _tmlib.lda.Online_OPE.OnlineOPE: api/api_lda.rst#class-tmlib-lda-online-ope-onlineope

.. _tmlib.lda.Streaming_VB.StreamingVB: api/api_lda.rst#class-tmlib-lda-streaming-vb-streamingvb

.. _tmlib.lda.Streaming_FW.StreamingFW: api/api_lda.rst#class-tmlib-lda-streaming-fw-streamingfw


.. _tmlib.lda.Streaming_OPE.StreamingOPE: api/api_lda.rst#class-tmlib-lda-streaming-ope-streamingope

.. _tmlib.lda.ML_CGS.MLCGS: api/api_lda.rst#class-tmlib-lda-ml-cgs-mlcgs

.. _tmlib.lda.ML_FW.MLFW: api/api_lda.rst#class-tmlib-lda-ml-fw-mlfw

.. _tmlib.lda.ML_OPE.MLOPE: api/api_lda.rst#class-tmlib-lda-ml-ope-mlope


