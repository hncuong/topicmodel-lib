.. -*- coding: utf-8 -*-

===========
User guide
===========

This document contains a description of all stochastic algorithms for learning LDA, and it also contains some tutorials about how to use the useful functions or methods in the library for many purposes. To understand clearly this document, user need to read `quick start`_ document and `lda model`_ first.

.. _quick start: ./quick_start.rst
.. _lda model: ./LatentDirichletAllocation.rst


-----------------------------
`1. Working with data input`_
-----------------------------

This section includes some tutorials for process data input of model (documents - corpus). This corpus maybe supplied by user, or available copus from `wikipedia`_ website (refer to `paper`_ and `source code`_). The library will support preprocessing, converting format of input for specific learning method.

.. _wikipedia: https://en.wikipedia.org/wiki/Main_Page
.. _paper: https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf
.. _source code: https://github.com/blei-lab/onlineldavb

`1.1. Preprocessing`_
=====================


`1.2. Loading a "mini-batch" from corpus`_
==========================================



`1.3. Loading a minibatch from Wikipedia website`_
==================================================

.. _1. Working with data input: user_guides/work_data_input.rst
.. _1.1. Preprocessing: user_guides/work_data_input.rst#preprocessing
.. _1.2. Loading a "mini-batch" from corpus: user_guides/work_data_input.rst#loading-a-mini-batch-from-corpus
.. _1.2. Loading a "mini-batch" from Wikipedia website: user_guides/work_data_input.rst#loading-a-mini-batch-from-wikipedia-website


---------------------------------------------------------
2. Stochastic methods for learning LDA from large corpora
---------------------------------------------------------

.. _OPE: https://arxiv.org/abs/1512.03308
.. _FW: https://arxiv.org/abs/1512.03300

Our library is deployed with 5 inference methods: VB [3]_, CVB0 [4]_, CGS [5]_, `OPE`_, `FW`_ and apply with online scheme, stream scheme or regularized online learning

- Online learning includes methods: Online-VB [2]_, Online-CVB0 [6]_, Online-CGS [5]_, Online-OPE, Online-FW
- Stream learning includes: Streaming-VB [7]_, Streaming-OPE, Streaming-FW
- Regularized online learning with CGS, FW, OPE: ML-CGS, ML-FW, ML-OPE.

**Default setting parameters**: 

- *Model paramters*: num_topics = 100; :math:`\alpha` = 0.01; :math:`\eta` = 0.01. Such a choice of (:math:`\alpha`, :math:`\eta`) has been observed to work well in many previous studies
- *Inference parameters*: at most 50 iterations were allowed for OPE and VB to do inference. We terminated VB if the relative improvement of the lower bound on likelihood is not better than 10−4. 50 samples were used in CGS for which the first 25 were discarded and the remaining were used to approximate the posterior distribution. 50 iterations were used to do inference in CVB0, in which the first 25 iterations were burned in. Those number of samples/iterations are often enough to get a good inference solution according to [5]_ , [6]_
- *Learning parameters*: :math:`\kappa` = 0.9; :math:`\tau` = 1; in addition to, :math:`\beta` (or :math:`\lambda`) is instructed randomly with a specific distribution, depend on each method learning.
- User can change value of setting parameters above when object of method is created. (see example below)

`2.1. Online-VB`_
=================

`2.2. Online-CVB0`_
===================

`2.3. Online-CGS`_
==================

`2.4. Online-OPE`_
==================

`2.5. Online-FW`_
=================

`2.6. Streaming-VB`_
====================

`2.7. Streaming-OPE`_
=====================

`2.8. Streaming-FW`_
====================

`2.9. ML-CGS`_
===============

`2.10. ML-OPE`_
===============

`2.11. ML-FW`_
===============

.. _2.1. Online-VB: learning_tutors/online_vb.rst
.. _2.2. Online-CVB0: learning_tutors/online_cvb0.rst
.. _2.3. Online-CGS: learning_tutors/online_cgs.rst
.. _2.4. Online-OPE: learning_tutors/online_ope.rst
.. _2.5. Online-FW: learning_tutors/online_fw.rst
.. _2.6. Streaming-VB: learning_tutors/streaming_vb.rst
.. _2.7. Streaming-OPE: learning_tutors/streaming_ope.rst
.. _2.8. Streaming-FW: learning_tutors/streaming_fw.rst
.. _2.9. ML-CGS: learning_tutors/ml_cgs.rst
.. _2.10. ML-OPE: learning_tutors/ml_ope.rst
.. _2.11. ML-FW: learning_tutors/ml_fw.rst


--------------------------------
`3. How to save or load model`_
--------------------------------


`Save model`_ (:math:`\lambda` or :math:`\beta`)
================================================


`Load model from a file`_
=========================



`Save top words of topics`_
===========================

.. _3. How to save or load model: user_guides/load_save_model.rst
.. _Save model: user_guides/load_save_model.rst#save-model-lambda-or-beta
.. _Load model from a file: user_guides/load_save_model.rst#load-model-from-a-file
.. _Save top words of topics: user_guides/load_save_model.rst#save-top-words-of-topics 


.. [1] Care and Feeding of Topic Models: Problems, Diagnostics, and Improvements Jordan Boyd-Graber, David Mimno, and David Newman. In Handbook of Mixed Membership Models and Their Applications, CRC/Chapman Hall, 2014.
.. [2] M.D. Hoffman, D.M. Blei, C. Wang, and J. Paisley, "Stochastic variational inference," The Journal of Machine Learning Research, vol. 14, no. 1, pp. 1303–1347, 2013.
.. [3] D.M. Blei, A.Y. Ng, and M.I. Jordan, "Latent dirichlet allocation," Journal of Machine Learning Research, vol. 3, no. 3, pp. 993–1022, 2003.
.. [4] Asuncion, M. Welling, P. Smyth, and Y. Teh, "On smoothing and inference for topic models," in Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, 2009, pp. 27–34.
.. [5] D.Mimno, M. D. Hoffman, and D. M. Blei, "Sparse stochastic inference for latent dirichlet allocation," in Proceedings of the 29th Annual International Conference on Machine Learning, 2012.
.. [6] James Foulds, Levi Boyles, Christopher DuBois, Padhraic Smyth, and Max Welling. Stochastic collapsed variational bayesian inference for latent dirichlet allocation. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 446–454. ACM, 2013.
.. [7] Tamara Broderick, Nicholas Boyd, Andre Wibisono, Ashia C Wilson, and Michael Jordan. Streaming variational bayes. In Advances in Neural Information Processing Systems, pages 1727–1735, 2013.
.. [8] Khoat Than and Tu Bao Ho, “Fully sparse topic models”. European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD), Bristol, UK. Vol. 7523 of Lecture Notes in Computer Science, Springer, pages 490-505, 2012.
