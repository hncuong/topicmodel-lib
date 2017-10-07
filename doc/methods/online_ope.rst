==========
Online OPE
==========

Similar to FW, OPE [1] is a inference method allowing us estimate directly topic proportions :math:`\theta` for individual document. The problem of posterior inference for each document d, given a model {:math:`\beta`, :math:`\alpha`}, is to estimate the full joint distribution **P** (:math:`z_d`, :math:`\theta_d`, d | :math:`\beta`, :math:`\alpha`). Direct estimation of this distribution is intractable. Hence existing approaches uses different schemes. VB, CVB0 try to estimate the distribution by maximizing a lower bound of the likelihood **P** (d | :math:`\beta`, :math:`\alpha`), whereas CGS tries to estimate **P** (:math:`z_d` | d, :math:`\beta`, :math:`\alpha`).

OPE will estimate :math:`\theta` by maximize the posterior distribution **P** (:math:`\theta`, d | :math:`\beta`, :math:`\alpha`):

.. math::

   \theta^* = argmax_{\theta \in \Delta_K} P(\theta, d | \beta, \alpha)

where :math:`\theta` is a vector K-dimention (K is number of topics) and :math:`\theta \in \Delta_K`, it means:

.. math::

   \left\{\begin{array}\quad \theta_k > 0, \quad k = 1, ..., K  \\ \theta_1 + \theta_2 + ... + \theta_K = 1  \end{array}\right.

The objective function sounds like FW, but OPE hasn't a constraint about :math:`\alpha` like FW and so, the optimization algorithm of OPE is also different from FW

The update :math:`\lambda` (variational parameter of :math:`\beta`) is designed followed by online scheme
   
------------------------------------
class OnlineOPE
------------------------------------

::

  tmlib.lda.OnlineOPE(data=None, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None)

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

- **iter_infer**: int, default: 50.

  Number of iterations of FW algorithm to do inference step

- **lda_model**: object of class ``LdaModel``.

  If this is None value, a new object ``LdaModel`` will be created. If not, it will be the model learned previously

Attributes
==========

- **num_docs**: int,
  
  Number of documents in the corpus.

- **num_terms**: int,

  size of the vocabulary set of the training corpus

- **num_topics**: int, 

- **alpha** (:math:`\alpha`): float, 

- **eta** (:math:`\eta`): float, 

- **tau0** (:math:`\tau_{0}`): float, 

- **kappa** (:math:`\kappa`): float, 

- **INF_MAX_ITER**: int,

  Number of iterations of FW algorithm to do inference step

- **lda_model**: object of class ``LdaModel``

Methods
=======

- __init__ (*data=None, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

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

    from tmlib.lda import OnlineOPE
    from tmlib.datasets import DataSet

    # data preparation
    data = DataSet(data_path='data/ap_train_raw.txt', batch_size=100, passes=5, shuffle_every=2)
    # learning and save the model, statistics in folder 'models-online-ope'
    onl_ope = OnlineOPE(data=data, num_topics=20, alpha=0.2)
    model = streaming_ope.learn_model(save_model_every=1, compute_sparsity_every=1, save_statistic=True, save_top_words_every=1, num_top_words=10, model_folder='models-online-ope')
    

    # inference for new documents
    vocab_file = data.vocab_file
    # create object ``Corpus`` to store new documents
    new_corpus = data.load_new_documents('data/ap_infer_raw.txt', vocab_file=vocab_file)
    theta = onl_ope.infer_new_docs(new_corpus)
  
[1]  Khoat Than, Tung Doan, “Guaranteed inference in topic models". [Online]. Available at: https://arxiv.org/abs/1512.03308

