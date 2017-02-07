.. -*- coding: utf-8 -*-

====================================
tmlib.lda: Methods for learnning LDA
====================================


.. Contents::


-----------------------------------------------------
class tmlib.lda.ldamodel.LdaModel
-----------------------------------------------------

tmlib.lda.ldamodel.LdaModel(*num_terms, num_topics, random_type=0*)

This class works with model (:math:`\lambda` or :math:`\beta`): save, load, display words of topics...

Parameters
===========

- **num_terms**: int
  
  number of words in vocabulary file
- **num_topics**: int,

  number of topics

- **random_type**: int, default: 0

  Initialize randomly array of :math:`\lambda` (or :math:`\beta`) (size num_topics x num_terms). If random_type = 0, model is initialized with uniform distribution. Otherwise, initialized with gamma distribution

Attributes
==========

- **num_terms**: int
- **num_topics**: int

- **model**: array 2 dimentions (num_topics x num_terms)
  
  :math:`\lambda` or :math:`\beta`


Methods
=======

- __init__(*num_terms, num_topics, random_type=0*)
- **normalize** ()

  Used for estimating :math:`\beta` from :math:`\lambda`. This function is usually used for regularized methods  

- **print_top_words** (self, num_words, vocab_file, show_topics=None, result_file=None)

  Display words of topics on the screen or save into file

  - **Parameters**: 

    - **num_words**: int, 
    
      number of words of each topic is displayed
    - **vocab_file**: string, 
    
      path of file vocabulary
    - **show_topics**: int, default: None

      number of topics is displayed. By default, all of topics are displayed
    - **result_file**: string, default: None

      path of file to save words into. By default, if result_file=None, words of topics are displayed on screen

- **load** (model_file)

  loading the learned model (:math:`\lambda` or :math:`\beta`) from file named *model_file*
  
- **save** (model_file, file_type='binary')

  saving model into a file named model_file. By default, the type of file is binary. We can change type of file to text by set file_type='txt'

-----------------------------------------
class tmlib.lda.ldalearning.LdaStatistics
-----------------------------------------

tmlib.lda.ldalearning.LdaStatistics()

This class is used for saving statistics of model such as: time of E-step, time of M-step, time of inference, or sparsity document in each iteration

Attributes
==============

- **e_step_time**: list,

  list of time of E-step
- **m_step_time**: list,

  list of time of M-step
- **iter_time**: list,

  list of time of each iteration. Time of each iteration = time E-step + time M-step in each iteration

- **sparsity_record**: list,

  store the computed sparsities in some iterations

Methods
========

- __init__()

- **record_time** (time_e, time_m)

  append a time record to lists: e_step_time, m_step_time, iter_time

  **time_e**: time of E-step

  **time_m**: time of M-step

- **reset_time_record** ()

  reset all of lists to empty

- **record_sparsity** (sparsity)

  append a sparsity record to list sparsity_record

- **reset_sparsity_record** ()

- **save_time** (file_name, reset=False)

  Save time records into a file
 
  *file_name**: name of the saved file

  **reset**: if reset = True then reseting all list of time to empty

- **save_sparsity** (file_name, reset=False)

  Save sparsity records into a file named *file_name*

---------------------------------------
class tmlib.lda.ldalearning.LdaLearning
---------------------------------------

tmlib.lda.ldalearning. **LdaLearning** (*num_terms, num_topics, lda_model=None*)

This class is used for learning LDA. This is the super-class of all learning methods.

Parameters
==========

- **num_terms**: int

  number of words in vocabulary file

- **num_topics**: int

  number of topics of model

- **lda_model**: object of class LdaModel, default: None

  This parameter is used for storing the learned model after each iteration. If it is set None value, sub-class of this class must initialize it. If it is the learned model, it can be updated in this learning time. 

Attributes
==========

- **num_terms**: int

- **num_topics**: int

- **lda_model**: object of class LdaModel

- **statistics**: object of class LearningStatistics

  Used for storing the statistics of model in learning process

Methods
=======

- __init__ (*num_terms, num_topics, lda_model=None*)

- **static_online** (*word_ids_tks, cts_lens*)

  This function implements learning algorithms. It is a abstract method

  **word_ids_tks** and **cts_lens**: see attributes of class `Corpus`_

.. _Corpus: ./api_base.rst

- __getitem__ (*docs*)

  This is also abstract method. This used for inference new documents. 

  **docs**: object of class Corpus, store new documents used for inference

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*) 

  This used for learning model and to save model, statistics of model. 

  **Parameters**:

    - **data**: object of class Dataset, used to supply minibatches data for learning

    - **save_model_every**: int, default: 0. If it is set to 2, it means at iterators: 0, 2, 4, 6, ..., model will is save into a file. If setting default, model won't be saved.

    - **compute_sparsity_every**: int, default: 0. Compute sparsity and store in attribute **statistics**. The word "every" here means as same as **save_model_every**

    - **save_statistic**: boolean, default: False. Saving statistics or not

    - **save_top_words_every**: int, default: 0. Used for saving top words of topics (highest probability). Number words displayed is **num_top_words** parameter.

    - **num_top_words**: int, default: 20. By default, the number of words displayed is 20.

    - **model_folder**: string, default: "model". The place which model file, statistics file are saved. By default, all of this values will be saved in folder "model"

  **Return**: the learned model (object of class LdaModel)

----------------------------------
class tmlib.lda.Online_VB.OnlineVB
----------------------------------

tmlib.lda.Online_VB. **OnlineVB** (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, conv_infer=0.0001, iter_infer=50, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by Online-VB method.

Parameters
==========

- **num_terms**: int,

  number words of vocabulary file

- **num_topics**: int, default: 100

  number of topics of model.

- **alpha**: float, default: 0.01

  parameter :math:`\alpha` of model LDA

- **eta** (:math:`\eta`): float, default: 0.01 

- **tau0** (:math:`\tau_{0}`): float, default: 1.0

- **kappa** (:math:`\kappa`): float, default: 0.9

- **conv_infer**: float, default: 0.0001

  The relative improvement of the lower bound on likelihood of VB inference. If If bound hasn't changed much, the inference will be stopped

- **iter_infer**: int, default: 50.

  number of iterations to do inference

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_terms**: int,

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

- __init__ (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, conv_infer=0.0001, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  Excute the learning algorithm, includes: inference for individual document and update :math:`\lambda`. 2 parameters *wordids*, *wordcts* represent for term-frequency format of mini-batch

  **Return**: tuple (time of E-step, time of M-step, gamma). gamma (:math:`\gamma`) is variational parameter of :math:`\theta`

- **e_step** (*wordids, wordcts*)

  Do inference for indivial document (E-step)

  **Return**: tuple (gamma, sstats), where, sstats is the sufficient statistics for the M-step

- **update_lambda** (*batch_size, sstats*)

  Update :math:`\lambda` by stochastic way. 

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning

--------------------------------------
class tmlib.lda.Online_CVB0.OnlineCVB0
--------------------------------------

tmlib.lda.Online_CVB0. **OnlineCVB0** (*num_tokens, num_terms, num_topics=100, alpha=0.01, eta=0.01, tau_phi=1.0, kappa_phi=0.9, s_phi=1.0, tau_theta=10.0, kappa_theta=0.9, s_theta=1.0, burn_in=25, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by Online-CVB0 method.

Parameters
==========

- **num_tokens**: int,

  number tokens of corpus

- **num_terms**: int,

  number words of vocabulary file

- **num_topics**: int, default: 100

  number of topics of model.

- **alpha**: float, default: 0.01

  parameter :math:`\alpha` of model LDA

- **eta** (:math:`\eta`): float, default: 0.01 

- **tau_phi** : float, default: 1.0

- **kappa_phi** : float, default: 0.9

- **s_phi**: float, default: 1.0

- **tau_theta**: float, default: 10.0

- **kappa_theta**: float, default: 0.9

- **s_theta**: float, default: 1.0

- **burn_in**: int, default: 25

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

Same as parameters above

Methods
=======

- __init__ (*num_tokens, num_terms, num_topics=100, alpha=0.01, eta=0.01, tau_phi=1.0, kappa_phi=0.9, s_phi=1.0, tau_theta=10.0, kappa_theta=0.9, s_theta=1.0, burn_in=25, lda_model=None*)

- **static_online** (*wordtks, lengths*)

  Excute the learning algorithm, includes: inference for individual document and update :math:`\lambda`. 2 parameters *wordtks*, *lengths* represent for term-sequence format of mini-batch

  **Return**: tuple (time of E-step, time of M-step, N_theta). 

- **e_step** (*wordids, wordcts*)

  Do inference for indivial document (E-step)

  **Return**: tuple (N_phi, N_Z, N_theta)

- **m_step** (*batch_size, N_phi, N_Z*)

  Update :math:`\lambda` by stochastic way. 

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning

------------------------------------
class tmlib.lda.Online_CGS.OnlineCGS
------------------------------------

tmlib.lda.Online_CGS. **OnlineCGS** (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, burn_in=25, samples=25, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by Online-CGS method.

Parameters
==========

- **num_terms**: int,

  number words of vocabulary file

- **num_topics**: int, default: 100

  number of topics of model.

- **alpha**: float, default: 0.01

  parameter :math:`\alpha` of model LDA

- **eta** (:math:`\eta`): float, default: 0.01 

- **tau0** (:math:`\tau_{0}`): float, default: 1.0

- **kappa** (:math:`\kappa`): float, default: 0.9

- **burn_in**: int, default: 25

- **samples**: int, default: 25

  50 samples were used in CGS for which the first 25 (burn_in) were discarded and the remaining (samples) were used to approximate the posterior distribution

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_terms**: int,

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

- __init__ (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, burn_in=25, samples=25, lda_model=None*)

- **static_online** (*wordtks, lengths*)

  Excute the learning algorithm, includes: inference for individual document and update :math:`\lambda`. 2 parameters *wordtks*, *lengths* represent for term-sequence format of mini-batch

  **Return**: tuple (time of E-step, time of M-step, theta). 
- **sample_z** (*wordids, wordcts*)

  Do inference for indivial document (E-step) by sampling

  **Return**: tuple (Nkw_mean, Ndk_mean, z)

- **update_lambda** (*batch_size, sstats*)

  Update :math:`\lambda` by stochastic way. Parameter sstats is Nkw_mean in ouput of function sample_z

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning

----------------------------------
class tmlib.lda.Online_FW.OnlineFW
----------------------------------

tmlib.lda.Online_FW. **OnlineFW** (*num_terms, num_topics=100, eta=0.01, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by Online-FW method.

Parameters
========== 

- **num_terms**: int,

  Number of unique terms in the corpus (length of the vocabulary)

- **num_topics**: int, default: 100

  Number of topics shared by the whole corpus.

- **eta** (:math:`\eta`): float, default: 0.01 
  
  Hyperparameter for prior on topics beta.

- **tau0** (:math:`\tau_{0}`): float, default: 1.0

  A (positive) learning parameter that downweights early iterations.

- **kappa** (:math:`\kappa`): float, default: 0.9

  Learning rate: exponential decay rate should be between (0.5, 1.0] to guarantee asymptotic convergence.

- **iter_infer**: int, default: 50.

  Number of iterations of FW algorithm.

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_docs**: int,
  
  Number of documents in the corpus.

- **num_terms**: int,

- **num_topics**: int, 

- **eta** (:math:`\eta`): float, 

- **tau0** (:math:`\tau_{0}`): float, 

- **kappa** (:math:`\kappa`): float, 

- **INF_MAX_ITER**: int,

- **lda_model**: object of class LdaModel

Methods
=======

- __init__ (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  First does an E step on the mini-batch given in wordids and wordcts, then uses the result of that E step to update the topics in M step.

  **Parameters**:

  - **wordids**: A list whose each element is an array (terms), corresponding to a document. Each element of the array is index of a unique term, which appears in the document, in the vocabulary.
  - **wordcts**: A list whose each element is an array (frequency), corresponding to a document. Each element of the array says how many time the corresponding term in wordids appears in the document.
    
  **Return**: tuple (time of E-step, time of M-step, theta): time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch. 

- **e_step** (*wordids, wordcts*)

  Does e step
  
  Note that, FW can provides sparse solution (theta:topic mixture) when doing inference for each documents. It means that the theta have few non-zero elements whose indexes are stored in list of lists 'index'.

  **Return**: tuple (theta, index): topic mixtures and their nonzero elements' indexes of all documents in the mini-batch.

- **infer_doc** (*ids, cts*):

  Does inference for a document using Frank Wolfe algorithm.
        
  **Parameters**

  - ids: an element of wordids, corresponding to a document.
  - cts: an element of wordcts, corresponding to a document.

  **Returns**: inferred theta and list of indexes of non-zero elements of the theta.

- **m_step** (*wordids, wordcts, theta, index*)

  Does M-step

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning

------------------------------------
class tmlib.lda.Online_OPE.OnlineOPE
------------------------------------

tmlib.lda.Online_OPE. **OnlineOPE** (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by Online-OPE method.

Parameters
========== 

- **num_terms**: int,

  Number of unique terms in the corpus (length of the vocabulary)

- **num_topics**: int, default: 100

  Number of topics shared by the whole corpus.

- **alpha**: float, default: 0.01

   Hyperparameter for prior on topic mixture theta.

- **eta** (:math:`\eta`): float, default: 0.01 
  
  Hyperparameter for prior on topics beta.

- **tau0** (:math:`\tau_{0}`): float, default: 1.0

  A (positive) learning parameter that downweights early iterations.

- **kappa** (:math:`\kappa`): float, default: 0.9

  Learning rate: exponential decay rate should be between (0.5, 1.0] to guarantee asymptotic convergence.

- **iter_infer**: int, default: 50.

  Number of iterations of FW algorithm.

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_docs**: int,
  
  Number of documents in the corpus.

- **num_terms**: int,

- **num_topics**: int, 

- **alpha**: float, 

- **eta** (:math:`\eta`): float, 

- **tau0** (:math:`\tau_{0}`): float, 

- **kappa** (:math:`\kappa`): float, 

- **INF_MAX_ITER**: int,

- **lda_model**: object of class LdaModel

Methods
=======

- __init__ (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  First does an E step on the mini-batch given in wordids and wordcts, then uses the result of that E step to update the topics in M step.

  **Parameters**:

  - **wordids**: A list whose each element is an array (terms), corresponding to a document. Each element of the array is index of a unique term, which appears in the document, in the vocabulary.
  - **wordcts**: A list whose each element is an array (frequency), corresponding to a document. Each element of the array says how many time the corresponding term in wordids appears in the document.
    
  **Return**: tuple (time of E-step, time of M-step, theta): time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch. 

- **e_step** (*wordids, wordcts*)

  Does e step

  **Return**: Returns topic mixtures theta.

- **infer_doc** (*ids, cts*):

  Does inference for a document using Online MAP Estimation algorithm.
        
  **Parameters**

  - ids: an element of wordids, corresponding to a document.
  - cts: an element of wordcts, corresponding to a document.

  **Returns**: inferred theta 

- **m_step** (*wordids, wordcts, theta*)

  Does M-step

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning


----------------------------------------
class tmlib.lda.Streaming_VB.StreamingVB
----------------------------------------

tmlib.lda.Streaming_VB. **StreamingVB** (*num_terms, num_topics=100, alpha=0.01, eta=0.01, conv_infer=0.0001, iter_infer=50, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by Streaming-VB method.

Parameters
==========

- **num_terms**: int,

  number words of vocabulary file

- **num_topics**: int, default: 100

  Number of topics shared by the whole corpus.

- **alpha**: float, default: 0.01

  parameter :math:`\alpha` of model LDA

- **eta** (:math:`\eta`): float, default: 0.01 

- **conv_infer**: float, default: 0.0001

  The relative improvement of the lower bound on likelihood of VB inference. If If bound hasn't changed much, the inference will be stopped

- **iter_infer**: int, default: 50.

  number of iterations to do inference

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_terms**: int,

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

- __init__ (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, conv_infer=0.0001, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  Excute the learning algorithm, includes: inference for individual document and update :math:`\lambda`. 2 parameters *wordids*, *wordcts* represent for term-frequency format of mini-batch

  **Return**: tuple (time of E-step, time of M-step, gamma). gamma (:math:`\gamma`) is variational parameter of :math:`\theta`

- **e_step** (*wordids, wordcts*)

  Do inference for indivial document (E-step)

  **Return**: tuple (gamma, sstats), where, sstats is the sufficient statistics for the M-step

- **update_lambda** (*sstats*)

  Update :math:`\lambda` by stochastic way. Specificly, using stream learning

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning

----------------------------------------
class tmlib.lda.Streaming_FW.StreamingFW
----------------------------------------

tmlib.lda.Streaming_FW. **StreamingFW** (*num_terms, num_topics=100, eta=0.01, iter_infer=50, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by Streaming-FW method.

Parameters
==========

- **num_terms**: int,

  Number of unique terms in the corpus (length of the vocabulary).

- **num_topics**: int, default: 100

  Number of topics shared by the whole corpus.

- **eta** (:math:`\eta`): float, default: 0.01 

  Hyperparameter for prior on topics beta.

- **iter_infer**: int, default: 50.

  Number of iterations of FW algorithm.

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_terms**: int,

- **num_topics**: int, 

- **eta** (:math:`\eta`): float, 

- **iter_infer**: int,

- **lda_model**: object of class LdaModel


Methods
=======

- __init__ (*num_terms, num_topics=100, eta=0.01, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  First does an E step on the mini-batch given in wordids and wordcts, then uses the result of that E step to update the topics in M step.
		
  **Parameters**:

  - **wordids**: A list whose each element is an array (terms), corresponding to a document. Each element of the array is index of a unique term, which appears in the document, in the vocabulary.

  - **wordcts**: A list whose each element is an array (frequency), corresponding to a document. Each element of the array says how many time the corresponding term in wordids appears in the document.

  **Returns**: time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch.  

- **e_step** (*wordids, wordcts*)

  Does e step 
		
  Returns topic mixtures and their nonzero elements' indexes of all documents in the mini-batch.
        
  Note that, FW can provides sparse solution (theta:topic mixture) when doing inference for each documents. It means that the theta have few non-zero elements whose indexes are stored in list of lists 'index'.

- **infer_doc** (*ids, cts*):

  Does inference for a document using Frank Wolfe algorithm.
        
  **Parameters**

  - ids: an element of wordids, corresponding to a document.
  - cts: an element of wordcts, corresponding to a document.

  **Returns**: inferred theta and list of indexes of non-zero elements of the theta.

- **m_step** (*wordids, wordcts, theta, index*)

  Does M-step

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning

------------------------------------------
class tmlib.lda.Streaming_OPE.StreamingOPE
------------------------------------------

tmlib.lda.Streaming_OPE. **StreamingOPE** (*num_terms, num_topics=100, alpha=0.01, eta=0.01, iter_infer=50, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by Streaming-OPE method.

Parameters
========== 

- **num_terms**: int,

  Number of unique terms in the corpus (length of the vocabulary)

- **num_topics**: int, default: 100

  Number of topics shared by the whole corpus.

- **alpha**: float, default: 0.01

   Hyperparameter for prior on topic mixture theta.

- **eta** (:math:`\eta`): float, default: 0.01 
  
  Hyperparameter for prior on topics beta.

- **iter_infer**: int, default: 50.

  Number of iterations of FW algorithm.

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_docs**: int,
  
  Number of documents in the corpus.

- **num_terms**: int,

- **num_topics**: int, 

- **alpha**: float, 

- **eta** (:math:`\eta`): float, 

- **INF_MAX_ITER**: int,

- **lda_model**: object of class LdaModel

Methods
=======

- __init__ (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  First does an E step on the mini-batch given in wordids and wordcts, then uses the result of that E step to update the topics in M step.

  **Parameters**:

  - **wordids**: A list whose each element is an array (terms), corresponding to a document. Each element of the array is index of a unique term, which appears in the document, in the vocabulary.
  - **wordcts**: A list whose each element is an array (frequency), corresponding to a document. Each element of the array says how many time the corresponding term in wordids appears in the document.
    
  **Return**: tuple (time of E-step, time of M-step, theta): time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch. 

- **e_step** (*wordids, wordcts*)

  Does e step

  **Return**: Returns topic mixtures theta.

- **infer_doc** (*ids, cts*):

  Does inference for a document using Online MAP Estimation algorithm.
        
  **Parameters**

  - ids: an element of wordids, corresponding to a document.
  - cts: an element of wordcts, corresponding to a document.

  **Returns**: inferred theta 

- **m_step** (*wordids, wordcts, theta*)

  Does M-step

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning


------------------------------------
class tmlib.lda.ML_CGS.MLCGS
------------------------------------

tmlib.lda.ML_CGS. **MLCGS** (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, burn_in=25, samples=25, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by ML-CGS method.

Parameters
==========

- **num_terms**: int,

  number words of vocabulary file

- **num_topics**: int, default: 100

  number of topics of model.

- **alpha**: float, default: 0.01

  parameter :math:`\alpha` of model LDA

- **eta** (:math:`\eta`): float, default: 0.01 

- **tau0** (:math:`\tau_{0}`): float, default: 1.0

- **kappa** (:math:`\kappa`): float, default: 0.9

- **burn_in**: int, default: 25

- **samples**: int, default: 25

  50 samples were used in CGS for which the first 25 (burn_in) were discarded and the remaining (samples) were used to approximate the posterior distribution

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_terms**: int,

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

- __init__ (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, burn_in=25, samples=25, lda_model=None*)

- **static_online** (*wordtks, lengths*)

  Excute the learning algorithm, includes: inference for individual document and update :math:`\lambda`. 2 parameters *wordtks*, *lengths* represent for term-sequence format of mini-batch

  **Return**: tuple (time of E-step, time of M-step, theta). 
- **sample_z** (*wordids, wordcts*)

  Does E-step

  **Return**: tuple (Ndk_mean, z)

- **update_lambda** (*wordtks, lengths, Ndk_mean*)

  Update :math:`\lambda` by regularized online learning. :math:`\lambda` here is :math:`\beta`

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning

----------------------------------------
class tmlib.lda.ML_FW.MLFW
----------------------------------------

tmlib.lda.ML_FW. **MLFW** (*num_terms, num_topics=100, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by ML_FW method.

Parameters
==========

- **num_terms**: int,

  Number of unique terms in the corpus (length of the vocabulary).

- **num_topics**: int, default: 100

  Number of topics shared by the whole corpus.

- **tau0**: float, default: 1.0

  A (positive) learning parameter that downweights early iterations.
   
- **kappa**: float, default: 0.9

  Learning rate: exponential decay rate should be between (0.5, 1.0] to guarantee asymptotic convergence.

- **iter_infer**: int, default: 50.

  Number of iterations of FW algorithm.

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Note that if you pass the same set of all documents in the corpus every time and set kappa=0 this class can also be used to do batch FW.

Attributes
==========

- **num_terms**: int,

- **num_topics**: int, 

- **tau0** (:math:`\tau_{0}`): float,

- **kappa** (:math:`\kappa`): float,

- **iter_infer**: int,

- **lda_model**: object of class LdaModel


Methods
=======

- __init__ (*num_terms, num_topics=100, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  First does an E step on the mini-batch given in wordids and wordcts, then uses the result of that E step to update the topics in M step.
		
  **Parameters**:

  - **wordids**: A list whose each element is an array (terms), corresponding to a document. Each element of the array is index of a unique term, which appears in the document, in the vocabulary.

  - **wordcts**: A list whose each element is an array (frequency), corresponding to a document. Each element of the array says how many time the corresponding term in wordids appears in the document.

  **Returns**: time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch.  

- **e_step** (*wordids, wordcts*)

  Does e step 
		
  Returns topic mixtures and their nonzero elements' indexes of all documents in the mini-batch.
        
  Note that, FW can provides sparse solution (theta:topic mixture) when doing inference for each documents. It means that the theta have few non-zero elements whose indexes are stored in list of lists 'index'.

- **infer_doc** (*ids, cts*):

  Does inference for a document using Frank Wolfe algorithm.
        
  **Parameters**

  - ids: an element of wordids, corresponding to a document.
  - cts: an element of wordcts, corresponding to a document.

  **Returns**: inferred theta and list of indexes of non-zero elements of the theta.

- **sparse_m_step** (*wordids, wordcts, theta, index*)

  Does m step: update global variables beta, exploiting sparseness of the solutions returned by Frank-Wolfe algorithm from e step as well as that of wordids and wordcts lists.

- **m_step** (*batch_size, wordids, wordcts, theta, index*)

  Does m step: update global variables beta without considering the sparseness.

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning

------------------------------------------
class tmlib.lda.ML_OPE.MLOPE
------------------------------------------

tmlib.lda.ML_OPE. **MLOPE** (*num_terms, num_topics=100, alpha=0.01, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

This class inherits super-class LdaLearning. This used for learning LDA by Online-OPE method.

Parameters
========== 

- **num_terms**: int,

  Number of unique terms in the corpus (length of the vocabulary)

- **num_topics**: int, default: 100

  Number of topics shared by the whole corpus.

- **alpha**: float, default: 0.01

   Hyperparameter for prior on topic mixture theta.

- **tau0** (:math:`\tau_{0}`): float, default: 1.0

  A (positive) learning parameter that downweights early iterations.

- **kappa** (:math:`\kappa`): float, default: 0.9

  Learning rate: exponential decay rate should be between (0.5, 1.0] to guarantee asymptotic convergence.

- **iter_infer**: int, default: 50.

  Number of iterations of FW algorithm.

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_docs**: int,
  
  Number of documents in the corpus.

- **num_terms**: int,

- **num_topics**: int, 

- **alpha**: float, 

- **tau0** (:math:`\tau_{0}`): float, 

- **kappa** (:math:`\kappa`): float, 

- **INF_MAX_ITER**: int,

- **lda_model**: object of class LdaModel

Methods
=======

- __init__ (*num_terms, num_topics=100, alpha=0.01, tau0=1.0, kappa=0.9, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  First does an E step on the mini-batch given in wordids and wordcts, then uses the result of that E step to update the topics in M step.

  **Parameters**:

  - **wordids**: A list whose each element is an array (terms), corresponding to a document. Each element of the array is index of a unique term, which appears in the document, in the vocabulary.
  - **wordcts**: A list whose each element is an array (frequency), corresponding to a document. Each element of the array says how many time the corresponding term in wordids appears in the document.
    
  **Return**: tuple (time of E-step, time of M-step, theta): time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch. 

- **e_step** (*wordids, wordcts*)

  Does e step

  **Return**: Returns topic mixtures theta.

- **infer_doc** (*ids, cts*):

  Does inference for a document using Online MAP Estimation algorithm.
        
  **Parameters**

  - ids: an element of wordids, corresponding to a document.
  - cts: an element of wordcts, corresponding to a document.

  **Returns**: inferred theta 

- **m_step** (*wordids, wordcts, theta*)

  Does M-step: update global variables beta.

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  see class LdaLearning

- __getitem__(docs)

  see class LdaLearning

