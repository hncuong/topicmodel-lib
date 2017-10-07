=========
Online VB
=========

Online VB stand for Online Variational Bayes which is proposed by Hoffman, 2010 [1]_. The learning problem of LDA is to estimate full joint distribution **P** (**z**, :math:`\theta`, :math:`\beta` | C) given a corpus C. This problem is intractable and to sovle this, VB [2]_ approximate that distribution by a distribution Q

.. math::

   Q(z, \theta, \beta) = \prod_{d \in C} Q(z_d | \phi_d) \prod_{d \in C} Q(\theta_d | \gamma_d) \prod_k Q(\beta_k | \lambda_k)

(k is index of topic)

and now, the learning problem is reduced to estimation the variational parameters {:math:`\phi`, :math:`\gamma`, :math:`\lambda`}

The Online VB using `stochastic variational inference` includes 2 steps:

- Inference for each document in corpus C to find out :math:`\phi_{d}`, :math:`\gamma_{d}`
- Update global variable :math:`\lambda` by online fashion

.. _introduction: ../quick_start.rst

----------------------------------
class OnlineVB
----------------------------------

::

  tmlib.lda.OnlineVB (data=None, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, conv_infer=0.0001, iter_infer=50, lda_model=None)

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

- **tau0** (:math:`\tau_{0}`): float, 

- **kappa** (:math:`\kappa`): float, 

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

    from tmlib.lda import OnlineVB
    from tmlib.datasets import DataSet

    # data preparation
    data = DataSet(data_path='data/ap_train_raw.txt', batch_size=100, passes=5, shuffle_every=2)
    # learning and save the model, statistics in folder 'models-online-vb'
    onl_vb = OnlineVB(data=data, num_topics=20, alpha=0.2)
    model = onl_vb.learn_model(save_model_every=1, compute_sparsity_every=1, save_statistic=True, save_top_words_every=1, num_top_words=10, model_folder='models-online-vb')
    

    # inference for new documents
    vocab_file = data.vocab_file
    # create object ``Corpus`` to store new documents
    new_corpus = data.load_new_documents('data/ap_infer_raw.txt', vocab_file=vocab_file)
    gamma = onl_vb.infer_new_docs(new_corpus)

.. [1] M.D. Hoffman, D.M. Blei, C. Wang, and J. Paisley, "Stochastic variational inference," The Journal of Machine Learning Research, vol. 14, no. 1, pp. 1303–1347, 2013.
.. [2] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,” Journal of Machine Learning Research, vol. 3, no. 3, pp. 993–1022, 2003.
