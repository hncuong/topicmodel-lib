.. -*- coding: utf-8 -*-

===========
Quick-Start
===========
A very short introduction into topic models and how to solve them using topicmodel-lib. This document also introduces some basic concepts and conventions.

.. Contents::


---------------------------
Topic models
---------------------------
Topic models are probabilistic models of document collections that use latent variables to encode recurring patterns of word use (Blei, 2012). Topic modeling algorithms are inference algorithms; they uncover a set of patterns that pervade a collection and represent each document according to how it exhibits them. These patterns tend to be thematically coherent, which is why the models are called "topic models." Topic models are used for both descriptive tasks, such as to build thematic navigators of large collections of documents, and for predictive tasks, such as to aid document classification. Topic models have been extended and applied in many domains

`Latent Dirichlet Allocation (LDA)`_ is the simplest topic model, LDA is a generative probabilistic model for collections of discrete data such as text corpora.

.. _Latent Dirichlet Allocation (LDA): ./LatentDirichletAllocation.rst

Large-scale learning
====================
Modern data analysis requires computation with massive data. These problems illustrate some of the challenges to modern data analysis. Our data are complex and high-dimensional; we have assumptions to make - from science, intuition, or other data analyses - that involve structures we believe exist in the data but that we cannot directly observe; and finally our data sets are large, possibly even arriving in a never-ending stream. We deploy this library to computing with graphical models that is appropriate for massive data sets, data that might not fit in memory or even be stored locally. This is an efficient tool for learning LDA at large scales


Learning models for LDA
========================
To learn LDA at large-scale, a good and efficient approach is stochastic inference [1]_. The learning process includes 2 main steps:

- Inference for individual document: infer to find out the **local variables**: topic proportion :math:`\theta` and topic indices **z** (estimate directly or estimate their distribution :math:`\gamma`, :math:`\phi` - "variational parameters")
- Update **global variable** in a stochastic way to find out directly topics :math:`\beta` (regularized online learning) or its distribution by estimating :math:`\lambda` (online, stream). Global variable here maybe :math:`\beta` or :math:`\lambda` depend on each stochastic methods.

Indeed, this phase is as same as training step in machine learning. 

Posterior Inference
===============================================
Actually, this is the first step in learning phase above, posterior inference is the core step when designing efficient algorithms for learning topic models from large-scale data. It also plays a important role in many tasks, such as understanding individual texts, dimensionality reduction, and prediction

With given model {:math:`\alpha`, :math:`\eta`, :math:`\beta` (or :math:`\lambda`)}, we can infer for new document to find out topic proportion :math:`\theta` and **z** if necessary.

There exists some inference algorithms such as: Variational Bayesian (VB), Collapsed variational Bayes (CVB), Fast collapsed variational Bayes (CVB0), Collapsed Gibbs sampling (CGS), Frank Wolfe (FW) and Online Maximum a Posterior Estimation (OPE) (refer `user guide`_ document for detail)

.. _user guide: ./user_guide.rst

---------------------------------------------------------
Data input format
---------------------------------------------------------

Corpus
======
A corpus is a collection of digital documents. This collection is the input to topicmodel-lib from which it will infer the structure of the documents, their topics, topic proportions, etc. The latent structure inferred from the corpus can later be used to assign topics to new documents which were not present in the training corpus. For this reason, we also refer to this collection as the training corpus. No human intervention (such as tagging the documents by hand) is required - the topic classification is unsupervised.

Data Format
===========

Our framework is support for 3 input format:

- Corpus with raw text format:
  
  ::

    raw_corpus = ["Human machine interface for lab abc computer applications",
                  "A survey of user opinion of computer system response time",
                  "The EPS user interface management system",
                  "System and human system engineering testing of EPS",              
                  "Relation of user perceived response time to error measurement",
                  "The generation of random binary unordered trees",
                  "The intersection graph of paths in trees",
                  "Graph minors IV Widths of trees and well quasi ordering",
                  "Graph minors A survey"]

  More detail for format of raw text corpus in a file, see `user guide`_ document 

- Term-frequency format (tf):

  The implementations only support reading data type in LDA. Please refer to the following site for instructions: http://www.cs.columbia.edu/~blei/lda-c/
  Under LDA, the words of each document are assumed exchangeable.  Thus, each document is succinctly represented as a sparse vector of word counts. The data is a file where each line is of the form:

     [N] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

  where [N] is the number of unique terms in the document, and the [count] associated with each term is how many times that term appeared in the document.  Note that [term_i] is an integer which indexes the term (index of that term in file vocabulary); it is not a string.

  For example, with corpus as raw_corpus above and file vocabulary is:

     ::

       0. "human"
       1. "machine"
       2. "interface"
       3. "lab"
       4. "abc"
       5. "computer"
       6. "applications"
       7. "survey"
       8. "user"
       9. "opinion"
       10. "system"
       11. "response"
       12. "time"
       13. "eps"
       14. "management"
       15. "engineering"
       16. "testing"
       17. "relation"
       18. "perceived"
       19. "error"
       20. "measurement"
       21. "generation"
       22. "random"
       23. "binary"
       24. "unordered"
       25. "trees"
       26. "intersection"
       27. "graph"
       28. "paths"
       29. "minors"
       30. "widths"
       31. "quasi"
       32. "ordering"

  The tf format of corpus will be:
     
     ::

       7 0:1 1:1 2:1 3:1 4:1 5:1 6:1 
       7 7:1 8:1 9:1 5:1 10:1 11:1 12:1 
       5 13:1 8:1 2:1 14:1 10:1 
       5 10:2 0:1 15:1 16:1 13:1 
       7 17:1 8:1 18:1 11:1 12:1 19:1 20:1 
       5 21:1 22:1 23:1 24:1 25:1 
       4 26:1 27:1 28:1 25:1 
       6 27:1 29:1 30:1 25:1 31:1 32:1 
       3 27:1 29:1 7:1 

- Term-sequence format (sq):

  Each document is represented by a sequence of token as follow
    
      [token_1] [token_2] ....

  [token_i] also is index of it in vocabulary file, not a string. 
  The sq format of corpus above will be:

     ::

       0 1 2 3 4 5 6 
       7 8 9 5 10 11 12 
       13 8 2 14 10 
       10 0 10 15 16 13 
       17 8 18 11 12 19 20 
       21 22 23 24 25 
       26 27 28 25 
       27 29 30 25 31 32 
       27 29 7 

.. [1] M.D. Hoffman, D.M. Blei, C. Wang, and J. Paisley, "Stochastic variational inference," The Journal of Machine Learning Research, vol. 14, no. 1, pp. 1303-1347, 2013.
