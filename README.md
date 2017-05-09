topicmodel-lib
================

topicmodel-lib is a Python library for *topic modelling* - a field which provide an efficient way to discover hidden structures/semantics in massive data. Latent Dirichlet Allocation (LDA) is a popular model in this field and we focus on methods for learning LDA by online or stream scheme.

Features
--------

- Our library provides efficient algorithms for learning LDA from large-scale data. It includes the state-of-the-art learning methods at the present
- We also implement Cython code (a programming language that makes writing C extensions for the Python language as easy as Python itself) to increase speed of some algorithms
- All of learning methods are designed in the same way, easy to understand, easy to use

Installation
------------

**Dependencies**

To use library, your computer must installed all of these packages first:

- Linux OS (Stable on Ubuntu)
- Python version 2 (stable on version 2.7)
- Docutils >= 0.3
- Numpy >= 1.8
- Scipy >= 0.10,
- nltk (Natural Language Toolkit)
- Cython

**User Installation**

After download library, you install by running file setup.py as follow:

First, build the necessary packages:

      topicmodel-lib$ python setup.py build_ext --inplace

  or if you need permission to build:

      topicmodel-lib$ sudo python setup.py build_ext --inplace

After that, install library into your computer:

      topicmodel-lib$ sudo python setup.py install

Documentation
-------------

- [Quick Start](doc/quick_start.rst)
- [Tutorials](doc/tutorials.rst)
- [User Guide](doc/user_guide.rst)
- [Official API documentation](doc/list_api.rst)

Support
-------
Contributors:
      VuVanTu
