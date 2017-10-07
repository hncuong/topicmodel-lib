==========
Online CGS
==========

Originally, Collapsed Gibbs Sampling (CGS) was proposed by [1]_ for learning LDA from data. It recently has been successfully adapted to posterior inference for individual documents by [2]_. It tries to estimate **P** (z|d, :math:`\alpha`, :math:`\eta`) by iteratively resampling the topic indicator at each token in document d from the conditional distribution over that position given the remaining topic indicator variables (:math:`z^{−i}`):

.. math::

   P(z_i = k | z^{-i}) \propto (\alpha + \sum_{t \neq i} I(z_t = k)) * exp[\psi(\lambda_{kz_i} - \psi(\sum_t \lambda_{kt})].

Note that this adaptation makes the inference more local, i.e., posterior inference for a document does not need to modify any global variable. This property is similar with VB, but very different with CVB and CVB0

Online CGS [2]_ includes: inference to find out topic indicator (z) at each token in document and update global variable (variational parameter :math:`\lambda`) after that. 

---------------
class OnlineCGS
---------------

::

  tmlib.lda.OnlineCGS(data=None, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, burn_in=25, samples=25, lda_model=None)

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

- **_Elogbeta**: float,

  This is expectation of random variable :math:`\beta` (topics of model).

- **_expElogbeta**: float, this is equal exp(**_Elogbeta**)

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

    from tmlib.lda import OnlineCGS
    from tmlib.datasets import DataSet

    # data preparation
    data = DataSet(data_path='data/ap_train_raw.txt', batch_size=100, passes=5, shuffle_every=2)
    # learning and save the model, statistics in folder 'models-online-cgs'
    onl_cgs = OnlineCGS(data=data, num_topics=20, alpha=0.2)
    model = onl_cgs.learn_model(save_model_every=1, compute_sparsity_every=1, save_statistic=True, save_top_words_every=1, num_top_words=10, model_folder='models-online-cgs')
    

    # inference for new documents
    vocab_file = data.vocab_file
    # create object ``Corpus`` to store new documents
    new_corpus = data.load_new_documents('data/ap_infer_raw.txt', vocab_file=vocab_file)
    statistic_theta = onl_cgs.infer_new_docs(new_corpus)

.. [1] T. Griffiths and M. Steyvers, “Finding scientific topics,” Proceedings of the National Academy of Sciences of the United States of America, vol.101, no. Suppl 1, p. 5228, 2004.
.. [2] D. Mimno, M. D. Hoffman, and D. M. Blei, “Sparse stochastic inference for latent dirichlet allocation,” in Proceedings of the 29th Annual International Conference on Machine Learning, 2012.
