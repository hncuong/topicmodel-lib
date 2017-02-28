==========================
1. Working with data input
==========================

This section includes some tutorials for process data input of model (documents - corpus). This corpus maybe supplied by user, or available copus from `wikipedia`_ website (refer to `paper`_ and `source code`_). The library will support preprocessing, converting format of input for specific learning method.

.. _wikipedia: https://en.wikipedia.org/wiki/Main_Page
.. _paper: https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf
.. _source code: https://github.com/blei-lab/onlineldavb

.. Contents::


------------------
1.1. Preprocessing
------------------

- This work will be implemented when data format of input is raw text. Topic models take documents that contain words as input. We still have to determine what "words" we're going to use and how to extract them from the format raw text. Recall that most topic models treat documents as a bag-of-words, so we can stop caring about the order of the tokens within the text and concentrate on how many times a particular word appears in the text. So, we need to convert the raw format to term-sequence or term-frequency as mentioned in the `quick start`_ section. To understand in detail about technique of preprocessing, please read preprocessing [1]_ document. 

- File raw text also need a specific format type so that we can recognize it. The format of file   as follow:

  - Corpus includes many documents, all of that are saved into a file. 
  - Each document is represented as follow

    .. image:: ../images/format.PNG
   
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

.. _api preprocessing: ../api/api_preprocessing.rst

---------------------------------------
1.2. Loading a "mini-batch" from corpus
---------------------------------------

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
    
      data = dataset.DataSet(file_path, batch_size)
      minibatch = data.load_mini_batch()  # The format is term-frequency by default
     
    By default in above, number of passing over data is 1. We can change it by set:

    ::  
   
      data = dataset.DataSet(file_path, batch_size, passes=4, shuffle_every=2)
    
    This means 4 times of passing over data and after every 2 times, file is shuffled again. Assume that size of corpus is 5000 documents, batch_size = 100, then number of iterators is: 5000/100*4 = 2000. We can check the last iterator by using method *check_end_of_data()*.
  - output format is term-sequence

    ::

      from tmlib.datasets import dataset
      from tmlib.datasets.base import DataFormat

      data = dataset.DataSet(file_path, batch_size)
      data.set_output_format(DataFormat.TERM_SEQUENCE)
      minibatch = data.load_mini_batch()
      
- However, we can also implement the first choice as follow:

  - Define a function *sample()* with 2 parameters is: file which is formatted (tf or sq) and format of output (minibatch)
 
    ::

      from tmlib.datasets import base
      from tmlib.datasets.base import DataFormat
  
      def sample(file_formatted_path, batch_size, output_format):
          work_file = base.shuffle_formatted_data_file(file_formatted_path, batch_size)
          fp = open(work_file, 'r')
          if output_format == DataFormat.TERM_FREQUENCY:
              minibatch = base.load_mini_batch_term_frequency_from_term_frequency_file(fp, batch_size)
          else:
              minibatch = base.load_mini_batch_term_sequence_from_sequence_file(fp, batch_size) 
		  fp.close()
		  return minibatch

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

- Note: minibatch is object of class `Corpus`_ . This mini-corpus is stored with term-frequency format or term-sequence format
  
.. _Corpus: ../api/api_base.rst

-----------------------------------------------
1.3. Loading a minibatch from Wikipedia website
-----------------------------------------------
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
