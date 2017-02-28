==========================================================
Learning LDA and inference with stream data from wikipedia
==========================================================

The purpose of this tutorial is to show you how to train the LDA model based on a specific data - stream data (inlude articles from wikipedia website) and after that, use this model to infer a new data. In this part, we'll work with stream data, so I'll use the learning method by stream scheme. Of course, the online or regularized methods are also used for this case. We will select a detailed method to guide all of you. If you want to go into detail , you can find out more these methods in `user guide`_ document.

.. _user guide: ../user_guide.rst

.. Contents::


-------
Data
-------

To design and implement for this part, we refered `source code`_ Online-VB (Hoffman, 2010). Authors fit the LDA model to 3.3 million articles from Wikipedia (actually is a large corpora), and a `vocabulary`_ is extracted from this corpus. In each training iteration, we'll crawl randomly a mini-batch articles from Wikipedia and analyze it for training. To understand more detail, you can infer `how to load a mini-batch from wikipedia`_

.. _source code: https://github.com/blei-lab/onlineldavb/blob/master/onlinewikipedia.py
.. _vocabulary: ../../tmlib/datasets/data/wikipedia/vocab.txt
.. _how to load a mini-batch from wikipedia: ../user_guides/work_data_input.rst#loading-a-minibatch-from-wikipedia-website.rst

------------------------------
Learning
------------------------------

We will demo with the learning method `Streaming VB`_

.. _Streaming VB: ../user_guides/streaming_vb.rst


First, we'll create a object used for load data

**In[1]**:

:: 
 
  from tmlib.datasets.wiki_stream import WikiStream
  
  # Create object to load mini-batch from website 
  training_data = WikiStream(64, 100)

In setting above, size of a mini-batch is 64, and number of mini-batches used for traning (or number of interations to run the learning algorithm) is 100


After create object to load data, we need set value for `parameters`_ . By `default`_, number of topics is 100, alpha=0.01, eta=0.01, tau0=0.9, kappa=1.0, conv_infer=50, iter_infer=50

**In[2]**:

::

  from tmlib.lda.Streaming_VB import StreamingVB

  # get number of unique terms
  num_terms = training_data.get_num_terms()
  #create object and setting parameters in default
  obj_strvb = StreamingVB(num_terms)
  
After that, we learn model as follow:

**In[3]**:

::

  # learn model, model and statistics are saved in folder model_vb
  model = obj_strvb.learn_model(training_data, save_model_every=5, compute_sparsity_every=5,
                                save_statistic=True, save_top_words_every=5, num_top_words=10,
                                model_folder='model_stream_vb')  

See class `LdaLearning`_ to know what the above parameters mean. The algorithm will be stopped after 100 iterations. At the 4th, 9th, 14th, ..., 99th loop, the value of :math:`\lambda`, sparsity document, time and top words of each topic are saved. The folder **model_stream_vb** inludes these files:

- model_batch4.npy, model_batch9.npy, model_batch14.npy, ... , model_batch99.npy. These files save value of :math:`\lambda`
- top_words_batch4.txt, top_words_batch9.txt, ..., top_words_batch99.txt to save top 10 words of topics
- file sparsity100.csv and time100.csv save respectly document sparsity and time (time of E-step, time M-step in each iteration)

Finally, we save the value of :math:`\lambda`, display top 10 words of topics as follow:

**In[4]**:

::

  # save lambda to a file text 
  model.save('model_stream_vb/lambda_final.txt', file_type='text')
  # Estimating beta by normalize lambda
  model.normalize()
  # Display top 10 words of 10 topic
  model.print_top_words(10, training_data.vocab_file, show_topics=10)
  # or you can show all of topics by
  # model.print_top_words(10, training_data.vocab_file)
  # or you can save to a file named top_words_final.txt
  # model.print_top_words(10, training_data.vocab_file, result_file='model_stream_vb/top_words_final.txt')

**Output**:

::

  topic 000
      new 		 0.008113
      first 		 0.004547
      time 		 0.003746
      two 		 0.003542
      york 		 0.002589
      university 		 0.002514
      school 		 0.002432
      world 		 0.002413
      three 		 0.002332
      october 		 0.002200

  topic 001
      first 		 0.001946
      two 		 0.001712
      new 		 0.001666
      time 		 0.001343
      years 		 0.001296
      university 		 0.001249
      three 		 0.001050
      states 		 0.001046
      number 		 0.001033
      world 		 0.001029

  topic 002
      first 		 0.001967
      two 		 0.001936
      time 		 0.001618
      new 		 0.001458
      city 		 0.001394
      years 		 0.001256
      university 		 0.001232
      duke 		 0.001223
      war 		 0.001202
      world 		 0.001189

  topic 003
      score 		 0.186668
      team 		 0.108287
      seed 		 0.026724
      round 		 0.009304
      mens 		 0.006177
      first 		 0.005672
      time 		 0.005346
      final 		 0.005298
      report 		 0.005259
      event 		 0.004698

  topic 004
      first 		 0.002050
      art 		 0.001949
      new 		 0.001816
      two 		 0.001546
      time 		 0.001318
      university 		 0.001036
      united 		 0.001015
      city 		 0.000984
      series 		 0.000980
      day 		 0.000946

  topic 005
      first 		 0.004525
      new 		 0.003888
      two 		 0.002278
      time 		 0.002250
      united 		 0.001957
      named 		 0.001742
      war 		 0.001675
      years 		 0.001493
      john 		 0.001473
      year 		 0.001444

  topic 006
      first 		 0.001904
      new 		 0.001838
      two 		 0.001798
      time 		 0.001594
      university 		 0.001481
      ship 		 0.001445
      group 		 0.001380
      number 		 0.001303
      united 		 0.001280
      member 		 0.001171

  topic 007
      first 		 0.003349
      new 		 0.002382
      two 		 0.002283
      time 		 0.001614
      three 		 0.001502
      art 		 0.001463
      number 		 0.001443
      life 		 0.001332
      field 		 0.001295
      known 		 0.001275

  topic 008
      new 		 0.002254
      first 		 0.002059
      two 		 0.001728
      time 		 0.001414
      world 		 0.001260
      states 		 0.001254
      air 		 0.001119
      army 		 0.001067
      city 		 0.001044
      art 		 0.001039

  topic 009
      two 		 0.003724
      first 		 0.003343
      time 		 0.002620
      new 		 0.002562
      city 		 0.002016
      august 		 0.001570
      october 		 0.001520
      game 		 0.001482
      year 		 0.001446
      january 		 0.001401

-----------------------------
Inference for new stream data
-----------------------------

Assume that a stream data arrives and we have to infer for all of documents in this block. 
First, we need load stream data and return a corpus with a specific format

**In[5]**:

::

  from tmlib.datasets import base

  # size of data is 10 documents
  data = WikiStream(10,1)
  # return corpus of 10 documents with term-frequency format
  new_corpus = data.load_mini_batch()
  
After that, execute inference for new corpus

::

  from tmlib.lda.ldamodel import LdaModel

  # create object model
  learned_model = LdaModel(0,0)
  # load value of lambda from file saved above
  learned_model.load('model_stream_vb/lambda_final.txt')
  # inference by create new object for OnlineVB
  object = StreamingVB(num_terms, lda_model=learned_model)
  theta = object.infer_new_docs(new_corpus)
  # or you can infer by using object in learning phase
  # theta = obj_strvb.infer_new_docs(new_corpus)
  base.write_topic_mixtures(theta, 'model_stream_vb/topic_mixtures.txt')


.. _parameters: ../api/api_lda.rst#class-tmlib-lda-online-vb-onlinevb
.. _default: ../user_guide.rst#stochastic-methods-for-learning-lda-from-large-corpora
.. _LdaLearning: ../api/api_lda.rst#class-tmlib-lda-ldalearning-ldalearning
