.. topicmodel-lib documentation master file, created by
   sphinx-quickstart on Tue Sep 12 11:00:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to topicmodel-lib's documentation!
==========================================

topicmodel-lib is a Python library for *topic modeling* - a field which provides an efficient way to discover hidden structures/semantics in massive data. Latent Dirichlet Allocation (LDA) is a popular model in this field and we focus on methods for learning LDA by online or stream scheme.

Features
--------

- Our library provides efficient algorithms for learning LDA from large-scale data. It includes the state-of-the-art learning methods at the present
- We also implement Cython code (a programming language that makes writing C extensions for the Python language as easy as Python itself) to increase speed of some algorithms
- We've also designed the visualization module to help users understand and explore the result that the model discovers after learning

Getting started in 30s
----------------------

**Training data**

Because we need to learn the model from the massive data, the loading whole of training data into memory is a bad idea. Therefore, the online/streaming learning algorithms are usually preferred in this case. Training data should be stored in a file and with a specific format. Our library supports 3 formats of training data and in here, we'll demo with `ap corpus`_

.. _ap corpus: https://github.com/hncuong/topicmodel-lib/tree/master/examples/ap/data

**Tutorial**

First, class `DataSet` provides some functions to processing the training data:

::

    >>> from tmlib.datasets import DataSet
    >>> data = DataSet('ap_train_raw.txt', batch_size=100, passes=5, shuffle_every=2)

Learning LDA by Online VB method (`Hoffman, 2010`_):

.. _Hoffman, 2010: http://www.cs.columbia.edu/~blei/papers/HoffmanBleiBach2010c.pdf

::

    >>> from tmlib.lda import OnlineVB
    >>> onlinevb = OnlineVB(data, num_topics=20)
    >>> model = onlinevb.learn_model()


You can see the topics which is discovered by Online VB:

::

    >>> model.print_top_words(5, data.vocab_file, display_result='screen')


For a more in-depth tutorial about topicmodel-lib, you can see documentation. 
In the `examples folder`_ of the repository, you will see the example code as well as training data. You can run a demo to understand how the library work

.. _examples folder: https://github.com/hncuong/topicmodel-lib/tree/master/examples

Installation
------------

**Dependencies**

To use the library, your computer must installed all of these packages first:

- Linux OS (Stable on Ubuntu)
- Python version 2 (stable on version 2.7)
- Numpy >= 1.8
- Scipy >= 0.10
- nltk (Natural Language Toolkit)
- Cython
- Pandas >= 0.20

**User Installation**

- Installing by pip

  ::

    $ sudo pip install tmlib


- Installing by download project

  After download project, you install by running file setup.py in folder topicmodel-lib as follow:

  First, build the necessary packages:

  ::

    python setup.py build_ext --inplace
    
  if you need permission to build:
  
  ::

    sudo python setup.py build_ext --inplace
    
  After that, install library into your computer:

  ::

    sudo python setup.py install

Support
-------

If you have an open-ended or a research question, you can join and contact via: 

- `Google Group`_
- `Facebook Group`_

.. _Google Group: https://groups.google.com/forum/#!forum/dslab-tmlib
.. _Facebook Group: https://www.facebook.com/groups/465441110326541/?ref=group_browse_new

License
-------

The project is licensed under the MIT license.


.. toctree::
   :maxdepth: 2
   :caption: Getting started

   quick_start

.. toctree::
   :maxdepth: 2
   :caption: API

   lda_model
   datasets
   methods/online_vb
   methods/online_cvb0
   methods/online_cgs
   methods/online_fw
   methods/online_ope
   methods/streaming_vb
   methods/streaming_fw
   methods/streaming_ope
   methods/ml_cgs
   methods/ml_fw
   methods/ml_ope
   
.. toctree::
   :maxdepth: 2
   :caption: Other features
   
   preprocessing
   visualization
   

