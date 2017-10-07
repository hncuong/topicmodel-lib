======
ML-CGS
======

You can see that `Online CGS`_ is also a hybrid algorithm. It infers to topic indicators **z** at each token in individual document by Gibb sampling. After that, it defines a approximatee sufficient statistics to update global variable :math:`\lambda`. By borrowing idea from `ML-FW`_ and `ML-OPE`_, ML-CGS will estimate directly topics :math:`\beta` instead of :math:`\lambda`

First, ML-CGS will estimate :math:`\theta` from S sampled topic indicators :math:`{z}^{1,2,...,S}` in each mini-batch [1]_

And then, we can define a sufficient statistics :math:`\hat{\beta}` to update :math:`\beta` following [2]_ 

.. _Online CGS: ./online_cgs.rst
.. _ML-FW: ./online_fw.rst
.. _ML-OPE: ./online_ope.rst 

------------------------------------
class tmlib.lda.MLCGS
------------------------------------

::

  tmlib.lda.MLCGS(data=None, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, burn_in=25, samples=25, lda_model=None)

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

- **tau0** (:math:`\tau_{0}`): float, default: 1.0

  In the update :math:`\lambda` step, a parameter used is step-size :math:`\rho` (it is similar to the learning rate in gradient descent optimization). The step-size changes after each training iteration t

  .. math::

     \rho_t = (t + \tau_0)^{-\kappa}

  And in this, the `delay` tau0 (:math:`\tau_{0}`) >= 0 down-weights early iterations

- **kappa** (:math:`\kappa`): float, default: 0.9

  kappa (:math:`\kappa`) :math:`\in` (0.5, 1] is the forgetting rate which controls how quickly old information is forgotten

- **burn_in**: int, default: 25

  Topic indicator at each token in indivisual document is sampled many times. But at the first several iterations, the samples will be discarded. The parameter **burn_in** is number of the first iterations that we discard the samples

- **samples**: int, default: 25

  After burn-in sweeps, we begin saving sampled topic indicators and we have saved S samples :math:`{z}^{1,...,S}` (by default, S = 25)

- **lda_model**: object of class ``LdaModel``.

  If this is None value, a new object ``LdaModel`` will be created. If not, it will be the model learned previously

Attributes
==========

- **num_terms**: int,

  size of the vocabulary set of the training corpus

- **num_topics**: int, 

- **alpha**: float, 

- **eta** (:math:`\eta`): float, 

- **tau0** (:math:`\tau_{0}`): float, 

- **kappa** (:math:`\kappa`): float, 

- **burn_in**: int, 

- **samples**: int,

- **lda_model**: object of class LdaModel

Methods
=======

- __init__ (*data=None, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, burn_in=25, samples=25, lda_model=None*)

- **static_online** (*wordtks, lengths*)

  Excute the learning algorithm, includes: inference for individual document and update :math:`\lambda`. 2 parameters *wordtks*, *lengths* represent for term-sequence data of mini-batch. It is the value of 2 attribute **word_ids_tks** and **cts_lens** in class `Corpus`_

.. _Corpus: ../datasets.rst

  **Return**: tuple (time of E-step, time of M-step, statistic_theta). statistic_theta is a statistic estimated from sampled topic indicators :math:`{z}^{1,...,S}`. It plays a similar role with :math:`\gamma` in VB 

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

  This used to do inference for new documents. **new_corpus** is object ``Corpus``. This method return a statistic which used for estimating topic proportions :math:`\theta`

-------
Example
-------

  ::

    from tmlib.lda import MLCGS
    from tmlib.datasets import DataSet

    # data preparation
    data = DataSet(data_path='data/ap_train_raw.txt', batch_size=100, passes=5, shuffle_every=2)
    # learning and save the model, statistics in folder 'models-ml-cgs'
    ml_cgs = MLCGS(data=data, num_topics=20, alpha=0.2)
    model = ml_cgs.learn_model(save_model_every=1, compute_sparsity_every=1, save_statistic=True, save_top_words_every=1, num_top_words=10, model_folder='models-ml-cgs')
    

    # inference for new documents
    vocab_file = data.vocab_file
    # create object ``Corpus`` to store new documents
    new_corpus = data.load_new_documents('data/ap_infer_raw.txt', vocab_file=vocab_file)
    statistic_theta = ml_cgs.infer_new_docs(new_corpus)

.. [1] D. Mimno, M. D. Hoffman, and D. M. Blei, “Sparse stochastic inference for latent dirichlet allocation,” in Proceedings of the 29th Annual International Conference on Machine Learning, 2012.
.. [2] K. Than and T. B. Ho, “Fully sparse topic models,” in Machine Learning and Knowledge Discovery in Databases, ser. Lecture Notes in Computer Science, P. Flach, T. De Bie, and N. Cristianini, Eds. Springer, 2012, vol. 7523, pp. 490–505.
