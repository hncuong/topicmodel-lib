=============
Streaming OPE
=============

Similar to `Online OPE`_, Streaming OPE uses the inference OPE [1]_ for individual document to find out the local variables :math:`\theta` (topic proportions). But, the update global variable :math:`\lambda` (variational pamameter of :math:`\beta`) is adapted to the stream environments. With the streaming learning, we don't need to know the number of documents in Corpus.

For more detail, you can see in [1]_

We also make a simulation for the stream evironment with the articles from Wikipedia website. See `simulation`_

.. _simulation: ../simulation.rst
.. _Online OPE: online_ope.rst

------------------------------------------
class StreamingOPE
------------------------------------------

::

  tmlib.lda.StreamingOPE(data=None, num_topics=100, alpha=0.01, eta=0.01, iter_infer=50, lda_model=None)

Parameters
========== 

- **data**: object ``DataSet``

  object used for loading mini-batches data to analyze 

- **num_topics**: int, default: 100

  number of topics of model.

- **alpha**: float, default: 0.01

  hyperparameter of model LDA that affect sparsity of topic proportions :math:`\theta`

- **eta** (:math:`\eta`): float, default: 0.01 

  hyperparameter of model LDA that affect sparsity of topics :math:`\beta`

- **iter_infer**: int, default: 50.

  Number of iterations of FW algorithm to do inference step

- **lda_model**: object of class ``LdaModel``.

  If this is None value, a new object ``LdaModel`` will be created. If not, it will be the model learned previously

Attributes
==========

- **num_terms**: int,

  size of the vocabulary set of the training corpus

- **num_topics**: int, 

- **alpha** (:math:`\alpha`): float, 

- **eta** (:math:`\eta`): float,  

- **INF_MAX_ITER**: int,

  Number of iterations of FW algorithm to do inference step

- **lda_model**: object of class ``LdaModel``

Methods
=======

- __init__ (*data=None, num_topics=100, alpha=0.01, eta=0.01, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  First does an E step on the mini-batch given in wordids and wordcts, then uses the result of that E step to update the topics in M step.

  **Parameters**:

  - **wordids**: A list whose each element is an array (terms), corresponding to a document. Each element of the array is index of a unique term, which appears in the document, in the vocabulary.
  - **wordcts**: A list whose each element is an array (frequency), corresponding to a document. Each element of the array says how many time the corresponding term in wordids appears in the document.
    
  **Return**: tuple (time of E-step, time of M-step, theta): time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch. 

- **e_step** (*wordids, wordcts*)

  Does e step

  **Return**: Returns topic mixtures theta.

- **m_step** (*wordids, wordcts, theta*)

  Does M-step

- **learn_model** (*save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=10, model_folder=None, save_topic_proportions=None*)

  This used for learning model and to save model, statistics of model. 

  **Parameters**:

    - **save_model_every**: int, default: 0. If it is set to 2, it means at iterators: 0, 2, 4, 6, ..., model will is save into a file. If setting default, model won't be saved.

    - **compute_sparsity_every**: int, default: 0. Compute sparsity and store in attribute **statistics**. The word "every" here means as same as **save_model_every**

    - **save_statistic**: boolean, default: False. Saving statistics or not. The statistics here is the time of E-step, time of M-step, sparsity of document in corpus

    - **save_top_words_every**: int, default: 0. Used for saving top words of topics (highest probability). Number words displayed is **num_top_words** parameter.

    - **num_top_words**: int, default: 20. By default, the number of words displayed is 10.

    - **model_folder**: string, default: None. The place which model file, statistics file are saved.

    - **save_topic_proportions**: string, default: None. This used to save topic proportions :math:`\theta` of each document in training corpus. The value of it is path of file ``.h5``  

  **Return**: the learned model (object of class LdaModel)

- **infer_new_docs** (*new_corpus*)

  This used to do inference for new documents. **new_corpus** is object ``Corpus``. This method return topic proportions :math:`\theta` for each document in new corpus
  
-------
Example
-------

  ::

    from tmlib.lda import StreamingOPE
    from tmlib.datasets import DataSet

    # data preparation
    data = DataSet(data_path='data/ap_train_raw.txt', batch_size=100, passes=5, shuffle_every=2)
    # learning and save the model, statistics in folder 'models-streaming-ope'
    streaming_ope = StreamingOPE(data=data, num_topics=20, alpha=0.2)
    model = streaming_ope.learn_model(save_model_every=1, compute_sparsity_every=1, save_statistic=True, save_top_words_every=1, num_top_words=10, model_folder='models-streaming-ope')
    

    # inference for new documents
    vocab_file = data.vocab_file
    # create object ``Corpus`` to store new documents
    new_corpus = data.load_new_documents('data/ap_infer_raw.txt', vocab_file=vocab_file)
    theta = streaming_ope.infer_new_docs(new_corpus)
  
.. [1]  Khoat Than, Tung Doan, “Guaranteed inference in topic models". [Online]. Available at: https://arxiv.org/abs/1512.03308
