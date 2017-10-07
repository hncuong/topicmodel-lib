============
Streaming VB
============

Similar to `Online VB`_, Streaming VB uses the inference VB [1]_ for individual document to find out the local variables :math:`\gamma` (variational parameter of topic proportions :math:`\theta`) and :math:`\phi` (variational parameter of topic indicators **z**). But, the update global variable :math:`\lambda` (variational pamameter of :math:`\beta`) is adapted to the stream environments. With the streaming learning, we don't need to know the number of documents in Corpus.

For more detail, you can see in [2]_

We also make a simulation for the stream evironment with the articles from Wikipedia website. See `simulation`_

.. _simulation: ../simulation.rst
.. _Online VB: online_vb.rst

----------------------------------------
class StreamingVB
----------------------------------------

::

  tmlib.lda.StreamingVB(data=None, num_topics=100, alpha=0.01, eta=0.01, conv_infer=0.0001, iter_infer=50, lda_model=None)

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

- **conv_infer**: float, default: 0.0001

  The relative improvement of the lower bound on likelihood of VB inference. If If bound hasn't changed much, the inference will be stopped

- **iter_infer**: int, default: 50.

  number of iterations to do inference step 

- **lda_model**: object of class ``LdaModel``.

  If this is None value, a new object ``LdaModel`` will be created. If not, it will be the model learned previously

Attributes
==========

- **num_terms**: int,

  size of the vocabulary set of the training corpus

- **num_topics**: int, 

- **alpha**: float, 

- **eta** (:math:`\eta`): float, 

- **conv_infer**: float, 

- **iter_infer**: int,

- **lda_model**: object of class LdaModel

- **_Elogbeta**: float,

  This is expectation of random variable :math:`\beta` (topics of model).

- **_expElogbeta**: float, this is equal exp(**_Elogbeta**)

Methods
=======

- __init__ (*data=None, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, conv_infer=0.0001, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  Excute the learning algorithm, includes: inference for individual document and update :math:`\lambda`. 2 parameters *wordids*, *wordcts* represent for term-frequency data of mini-batch. It is the value of 2 attribute **word_ids_tks** and **cts_lens** in class `Corpus`_

.. _Corpus: ../datasets.rst

  **Return**: tuple (time of E-step, time of M-step, gamma). gamma (:math:`\gamma`) is variational parameter of :math:`\theta`

- **e_step** (*wordids, wordcts*)

  Do inference for indivial document (E-step)

  **Return**: tuple (gamma, sstats), where, sstats is the sufficient statistics for the M-step

- **update_lambda** (*batch_size, sstats*)

  Update :math:`\lambda` by stochastic way. 

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

  This used to do inference for new documents. **new_corpus** is object ``Corpus``. This method return :math:`\gamma`
  
-------
Example
-------

  ::

    from tmlib.lda import StreamingVB
    from tmlib.datasets import DataSet

    # data preparation
    data = DataSet(data_path='data/ap_train_raw.txt', batch_size=100, passes=5, shuffle_every=2)
    # learning and save the model, statistics in folder 'models-streaming-vb'
    streaming_vb = StreamingVB(data=data, num_topics=20, alpha=0.2)
    model = streaming_vb.learn_model(save_model_every=1, compute_sparsity_every=1, save_statistic=True, save_top_words_every=1, num_top_words=10, model_folder='models-streaming-vb')
    

    # inference for new documents
    vocab_file = data.vocab_file
    # create object ``Corpus`` to store new documents
    new_corpus = data.load_new_documents('data/ap_infer_raw.txt', vocab_file=vocab_file)
    gamma = streaming_vb.infer_new_docs(new_corpus)

.. [1] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,” Journal of Machine Learning Research, vol. 3, no. 3, pp. 993–1022, 2003.
.. [2] Tamara Broderick, Nicholas Boyd, Andre Wibisono, Ashia C Wilson, and Michael Jordan. Streaming variational bayes. In Advances in Neural Information Processing Systems, pages 1727{1735, 2013.