==================================================
Learning LDA and inference with an example dataset
==================================================

The purpose of this tutorial is to show you how to train the LDA model on a specific data and after that, use this model to infer a new data. In this library, we designed a lot of learning methods but those methods are similar in the way of using. Therefore, we will select a detailed method to guide all of you. If you want to go into detail , you can find out more these methods in `user guide`_ document.

.. _user guide: ./user_guide.rst

.. Contents::


-------
Data
-------

We will be using AP corpus for this tutorial. Although this is a small dataset, but we still use this corpus for convenient. This corpus is also used in paper LDA model (Blei, 2003). If you're following this tutorial to practice about LDA, you can pick a other corpus that you are familiar with.

The datasets are available in folder `topicmodel-lib/examples/ap`_ . This folder contains: (you must read `data format`_ used in this library first)

- Files **ap_train_raw.txt** and **ap_infer_raw.txt** are the raw text data, file `ap_train_raw.txt`_ inludes 2000 articles in AP corpus - used for training LDA, file `ap_infer_raw.txt`_ includes 50 articles - used for inference. 
- Files `ap_train.txt`_ and `ap_infer.txt`_ are the formatted data (here is term-frequency format) and a file vocabulary `vocab.txt`_ is used for these files.

Here, we'll demo with both of group data above to help you understand clearly about 
the way of working with a specific data input.

.. _topicmodel-lib/examples/ap: ../../examples/ap
.. _data format: ../quick_start.rst#data-input-format
.. _ap_train_raw.txt: ../../examples/ap/ap_train_raw.txt
.. _ap_infer_raw.txt: ../../examples/ap/ap_infer_raw.txt
.. _ap_train.txt: ../../examples/ap/ap_train.txt
.. _ap_infer.txt: ../../examples/ap/ap_infer.txt
.. _vocab.txt: ../../examples/ap/vocab.txt

------------------------------
Learning
------------------------------

If data input is raw text, it must be pre-processed first. You can infer how to implement `preprocessing`_ step in user guide documentation. We will demo with the learning method `Online VB`_

.. _preprocessing: ../user_guides/work_data_input.rst#preprocessing
.. _Online VB: ../user_guides/online_vb.rst


First, we'll create a object used for load data

**In[1]**:

:: 
 
  from tmlib.datasets.dataset import DataSet
  
  # Create object, file raw text will be pre-processed in this statement
  training_data = DataSet('ap/ap_train_raw.txt', 100, passes=10, shuffle_every=2)
  # or if data input isn't raw text format, you need add parameter vocab_file, for example
  # training_data = DataSet('ap/ap_train.txt', 100, passes=10, shuffle_every=2, vocab_file='ap/vocab.txt')

The parameters **passes** or **shuffle_every** are described in `here`_ . By default, you can see path of file formatted and file vocabulary after preprocessing as follow:

::

  # By default, the format of this file is term-frequency
  print 'Path of file input after preprocessing: %s' %training_data.data_path
  print 'Path of file vocabulary extracted: $s' %training_data.vocab_file

The output will be:  

`Path of file input after preprocessing: /home/kde/tmlib_data/ap_train_raw/ap_train_raw.tf`

`Path of file vocabulary extracted: /home/kde/tmlib_data/ap_train_raw/vocab.txt`


After create object to load data, we need set value for `parameters`_ . By default, number of topics is 100, but with AP corpus (a small dataset) We'll set: num_topics = 20 and alpha = 0.05 and eta = 0.05. Other parameters such as: tau0, kappa, conv_infer, iter_infer, lda_model is set in `default`_

**In[2]**:

::

  from tmlib.lda.Online_VB import OnlineVB

  # get number of unique terms
  num_terms = training_data.get_num_terms()
  #create object and setting parameters
  obj_onlvb = OnlineVB(num_terms, num_topics=20, alpha=0.05, eta=0.05)
  
After that, we learn model as follow:

**In[3]**:

::

  # learn model, model and statistics are saved in folder model_vb
  model = obj_onlvb.learn_model(training_data, save_model_every=5, compute_sparsity_every=5,
                                save_statistic=True, save_top_words_every=5, num_top_words=10,
                                model_folder='model_vb')  

See class `LdaLearning`_ to know what the above parameters mean. Because `passes` = 10 and training data includes 2200 documents, size of a mini-batch is 100 documents. Thus, the algorithm will be stopped after 220 iterations. At the 4th, 9th, 14th, ..., 219th loop, the value of :math:`\lambda`, sparsity document, time and top words of each topic are saved. The folder **model_vb** inludes these files:

- model_batch4.npy, model_batch9.npy, model_batch14.npy, ... , model_batch219.npy. These files save value of :math:`\lambda`
- top_words_batch4.txt, top_words_batch9.txt, ..., top_words_batch219.txt to save top 10 words of topics
- file sparsity220.csv and time220.csv save respectly document sparsity and time (time of E-step, time M-step in each iteration)

Finally, we save the value of :math:`\lambda`, display top 10 words of topics as follow:

**In[4]**:

::

  # save lambda to a file text 
  model.save('model_vb/lambda_final.txt', file_type='text')
  # Estimating beta by normalize lambda
  model.normalize()
  # Display top 10 words of 10 topic
  model.print_top_words(10, training_data.vocab_file, show_topics=10)
  # or you can show all of topics by
  # model.print_topc_words(10, training_data.vocab_file)
  # or you can save to a file named top_words_final.txt
  # model.print_top_words(10, training_data.vocab_file, result_file='model_vb/top_words_final.txt')

**Output**:

::

  topic 000
     year 	     	 0.018302
     percent 		 0.014672
     million 		 0.010303
     billion 		 0.006932
     company 		 0.006555
     department      0.005005
     tax 		 0.004187
     workers 		 0.003893
     people 		 0.003462
     agreement 		 0.003264

  topic 001
     bush 		 0.020104
     soviet 		 0.013548
     gorbachev 		 0.010231
     president 		 0.009194
     party 		 0.008124
     reagan 		 0.005903
     states 		 0.005415
     dukakis 		 0.004877
     campaign            0.004796
     people 		 0.004327

  topic 002
     people 		 0.006942
     military 		 0.004494
     government 		 0.004401
     state 		 0.004034
     iraq 		 0.004022
     year 		 0.004007
     police 		 0.003847
     panama 		 0.003385
     president 		 0.003355
     officials 		 0.003125

  topic 003
     people 		 0.008578
     year 		 0.005679
     government 		 0.004368
     state 		 0.004326
     police 		 0.004302
     years 		 0.004258
     mrs 		 0.003757
     time 		 0.003624
     president 		 0.003452
     house 		 0.003087

  topic 004
     dukakis 		 0.006863
     state 		 0.004504
     bush 		 0.004324
     democratic 		 0.003690
     year 		 0.003658
     campaign 		 0.003655
     made 		 0.003281
     people 		 0.003174
     years 		 0.003081
     poll 		 0.002968

  topic 005
     year 		 0.007710
     years 		 0.004193
     time 		 0.003738
     federal 		 0.003703
     company 		 0.003479
     state 		 0.003247
     people 		 0.003153
     plant 		 0.003085
     million 		 0.002967
     service 		 0.002868

  topic 006
     year 		 0.005328
     aids 		 0.005280
     percent 		 0.005178
     government 		 0.004817
     united 		 0.004670
     years 		 0.004621
     study 		 0.004392
     children 		 0.003958
     people 		 0.003840
     states 		 0.003833

  topic 007
     market 		 0.016671
     stock 		 0.013789
     dollar 		 0.013394
     prices 		 0.012173
     trading 		 0.009714
     cents 		 0.009282
     late 		 0.008945
     lower 		 0.008769
     exchange 		 0.008498
     york 		 0.008292

  topic 008
     years 		 0.005193
     people 		 0.004822
     year 		 0.004732
     president 		 0.003828
     time 		 0.003778
     government 		 0.003334
     officials 		 0.003080
     states 		 0.003017
     state 		 0.002958
     department 		 0.002893

  topic 009
     police 		 0.010539
     people 		 0.006032
     hospital 		 0.005449
     year 		 0.004424
     city 		 0.004294
     care 		 0.004002
     health 		 0.003962
     state 		 0.003774
     years 		 0.003538
     officials 		 0.003317

------------------------
Inference for new corpus
------------------------

We'll use the learned model to infer for corpus **ap_infer_raw.txt**. If format of data is raw text, it need to be preprocessed with the vocabulary file is extracted from training corpus above.

First, we need load data and return a corpus with specific format

**In[5]**:

::

  from tmlib.datasets import base

  data_path = 'ap/ap_infer_raw.txt'
  vocab_path = training_data.vocab_file
  # or you can assign directly if you know exactly position of file vocab, example
  # vocab_path = '/home/kde/tmlib_data/ap_train_raw/vocab.txt' if training file is ap_train_raw.txt or
  # vocab_path = 'ap/vocab.txt' if training file is ap_train.txt
  # check format of data
  input_format = base.check_input_format(data_path)
  if input_format == base.DataFormat.RAW_TEXT:
      # get list documents which are still raw text
      docs = base.load_batch_raw_text(data_path) 
      # load vocab and save with dictionary type of python 
      vocab_dict = base.read_vocab(vocab_path) # vocab_dict[term] = index
      # parse raw corpus to obtain the term-frequency format
      new_corpus = base.parse_doc_list(docs, vocab_dict)
  else:
      # if data is formatted, it is loaded and return corpus with term-freqency format in default
      new_corpus = base.load_batch_formatted_from_file(data_path) 
  
After that, execute inference for new corpus

::

  from tmlib.lda.ldamodel import LdaModel

  # create object model
  learned_model = LdaModel(0,0)
  # load value of lambda from file saved above
  learned_model.load('model_vb/lambda_final.txt')
  # inference by create new object for OnlineVB
  object = OnlineVB(num_terms, num_topics=20, alpha=0.05, eta=0.05, lda_model=learned_model)
  theta = object.infer_new_docs(new_corpus)
  # or you can infer by using object in learning phase
  # theta = obj_onlvb.infer_new_docs(new_corpus)
  base.write_topic_mixtures(theta, 'model_vb/topic_mixtures.txt')

**Output**:

::

  0.04855 0.05653 0.04423 0.05101 0.06032 0.06141 0.04341 0.05127 0.03749 0.04974   0.04644 0.03937 0.06163 0.04959 0.05229 0.04412 0.03936 0.07344 0.05019 0.03962
  0.05997 0.03375 0.05260 0.03929 0.04541 0.04531 0.05012 0.04784 0.02725 0.05416 0.03658 0.05041 0.05371 0.05137 0.05131 0.06556 0.06572 0.05870 0.04749 0.06345
  0.04046 0.05575 0.05613 0.05741 0.06087 0.04108 0.04663 0.05648 0.06102 0.05899 0.05117 0.05081 0.03895 0.04344 0.03452 0.04075 0.06002 0.05300 0.04759 0.04492
  0.05188 0.05720 0.05281 0.05049 0.04958 0.06348 0.06122 0.04709 0.03503 0.05076 0.04387 0.04457 0.05280 0.05577 0.05846 0.05004 0.03257 0.06078 0.04199 0.03961
  0.04668 0.06816 0.03686 0.06530 0.04859 0.04283 0.05323 0.05883 0.03720 0.05094  0.03035 0.04391 0.04971 0.05431 0.04680 0.06657 0.03382 0.05960 0.04796 0.05834
  0.05019 0.05668 0.04648 0.05900 0.03309 0.04761 0.03571 0.06495 0.06176 0.04657 0.05624 0.04677 0.04467 0.03575 0.04533 0.05397 0.05050 0.07205 0.04320 0.04947
  0.04384 0.04650 0.04305 0.05963 0.04536 0.06730 0.05199 0.03680 0.06364 0.05896 0.05809 0.04742 0.02810 0.04630 0.05672 0.03781 0.05806 0.04964 0.04915 0.05164
  0.05685 0.04112 0.06084 0.05382 0.06332 0.04710 0.06174 0.06620 0.04840 0.05370 0.04778 0.04909 0.04997 0.03806 0.04520 0.04086 0.03693 0.05186 0.03723 0.04992
  0.02535 0.02534 0.05836 0.04131 0.05822 0.04790 0.08300 0.07034 0.01391 0.01695 0.06571 0.03094 0.09627 0.04557 0.08031 0.06771 0.04167 0.03240 0.06228 0.03649
  0.05480 0.05244 0.03906 0.04824 0.03144 0.03797 0.03989 0.05175 0.04597 0.05587 0.06080 0.04574 0.04413 0.05904 0.04795 0.05280 0.06031 0.05920 0.04478 0.06782
  0.05871 0.04751 0.05916 0.03434 0.05407 0.05073 0.04154 0.04013 0.04618 0.06254 0.06337 0.04932 0.05721 0.06697 0.06181 0.06417 0.03155 0.04034 0.04088 0.02946
  0.04286 0.04187 0.04426 0.04888 0.04855 0.05688 0.06906 0.04099 0.05568 0.03943   0.06292 0.04908 0.06567 0.03792 0.05226 0.05485 0.04789 0.04561 0.06194 0.03342
  0.04749 0.04982 0.04550 0.04244 0.05487 0.05228 0.05689 0.05751 0.04590 0.04556 0.03818 0.03898 0.05071 0.04064 0.06108 0.05337 0.05939 0.04545 0.05051 0.06341
  0.04691 0.03059 0.03860 0.05256 0.04207 0.04233 0.04897 0.04930 0.04861 0.05655 0.04875 0.05382 0.04862 0.05924 0.03481 0.06436 0.07502 0.06051 0.07115 0.02723
  0.03612 0.05713 0.05239 0.04916 0.05616 0.05865 0.03381 0.04875 0.03743 0.05923 0.06432 0.05125 0.05207 0.04929 0.05661 0.05106 0.04829 0.04847 0.04461 0.04519
  0.04125 0.03342 0.05460 0.04359 0.05520 0.04115 0.05008 0.07303 0.05348 0.04705 0.04484 0.04680 0.04079 0.05068 0.04832 0.07016 0.06002 0.03659 0.06770 0.04126
  0.04606 0.05633 0.04979 0.03408 0.04267 0.05732 0.05482 0.06208 0.06391 0.05695 0.05391 0.04358 0.05679 0.05024 0.05834 0.05090 0.04362 0.04088 0.03876 0.03895
  0.05411 0.04949 0.05692 0.05868 0.04912 0.05981 0.03936 0.05109 0.04797 0.04225 0.04944 0.04549 0.04079 0.05708 0.06257 0.06069 0.03424 0.04231 0.05097 0.04761
  0.06284 0.06094 0.03648 0.05575 0.04673 0.05057 0.05416 0.04808 0.05258 0.05002 0.05488 0.03429 0.04865 0.05740 0.05125 0.05031 0.06656 0.03392 0.04235 0.04223
  0.06581 0.04898 0.06289 0.05704 0.04200 0.04421 0.04411 0.04380 0.04157 0.05180 0.03915 0.04680 0.05555 0.04733 0.05139 0.05301 0.05376 0.03843 0.05723 0.05512
  0.04262 0.06181 0.05904 0.04356 0.04492 0.03259 0.06036 0.05020 0.04119 0.04441 0.04864 0.05568 0.03615 0.03284 0.05559 0.06553 0.06558 0.04576 0.05638 0.05715
  0.05394 0.05654 0.03819 0.04678 0.03923 0.05355 0.07231 0.06859 0.05154 0.03831 0.05684 0.04186 0.05477 0.03704 0.03074 0.03663 0.06398 0.06813 0.05637 0.03467
  0.03231 0.03724 0.04716 0.06739 0.06464 0.07377 0.02288 0.03454 0.05760 0.03981 0.08134 0.04662 0.03870 0.04567 0.07471 0.04004 0.02098 0.06167 0.05185 0.06110
  0.06196 0.05868 0.04917 0.03301 0.05696 0.06749 0.05362 0.06185 0.05395 0.04239 0.03715 0.04471 0.05365 0.04497 0.04755 0.04811 0.04352 0.03606 0.05731 0.04787
  0.05106 0.04532 0.04614 0.04271 0.05626 0.05454 0.04039 0.05114 0.03677 0.04502 0.04999 0.05531 0.05126 0.06120 0.04965 0.05346 0.04621 0.05536 0.05078 0.05744
  0.04846 0.05043 0.06712 0.04888 0.03933 0.04474 0.05058 0.04468 0.04155 0.04585 0.04778 0.05339 0.04792 0.05627 0.06493 0.05459 0.05860 0.04967 0.03533 0.04990
  0.03960 0.04590 0.04912 0.07218 0.04125 0.04273 0.05521 0.05198 0.05126 0.04704 0.05824 0.06024 0.04666 0.06373 0.04543 0.03208 0.05586 0.04940 0.03918 0.05291
  0.04319 0.05252 0.05073 0.05450 0.05712 0.04390 0.03796 0.04552 0.05700 0.05640 0.05523 0.04045 0.04636 0.04916 0.04908 0.06368 0.03059 0.04960 0.05170 0.06530
  0.05728 0.04606 0.05557 0.04383 0.04968 0.05534 0.03824 0.04835 0.05675 0.04629 0.05932 0.07635 0.04682 0.04705 0.04742 0.05322 0.05274 0.03478 0.04351 0.04142
  0.04237 0.04564 0.04127 0.05528 0.04585 0.05108 0.06173 0.04598 0.05015 0.05221 0.04987 0.05646 0.03632 0.05990 0.05733 0.05725 0.04707 0.03732 0.04686 0.06004
  0.04680 0.05577 0.05644 0.04934 0.04783 0.03461 0.06147 0.04914 0.03758 0.04983 0.04709 0.04693 0.04101 0.04134 0.05732 0.05130 0.04224 0.06137 0.05227 0.07031
  0.05154 0.04771 0.04499 0.04687 0.05763 0.04348 0.06087 0.05986 0.05060 0.05471 0.05502 0.04155 0.04377 0.04471 0.06868 0.05544 0.04428 0.04969 0.03869 0.03992
  0.04097 0.05199 0.04469 0.06465 0.03482 0.03858 0.06328 0.05446 0.03943 0.04879 0.03661 0.06759 0.03924 0.06594 0.05004 0.05979 0.04849 0.04321 0.04692 0.06050
  0.05084 0.05039 0.05210 0.03791 0.05367 0.06189 0.06315 0.05878 0.04929 0.04628 0.04477 0.06008 0.05282 0.04024 0.05455 0.04368 0.04355 0.04213 0.05103 0.04287
  0.04973 0.04971 0.05817 0.05121 0.05507 0.04257 0.05511 0.05503 0.06401 0.04005 0.03639 0.05218 0.04056 0.05576 0.04497 0.04376 0.04923 0.05786 0.05136 0.04726
  0.03989 0.05958 0.05125 0.05430 0.05875 0.04631 0.04988 0.04382 0.04022 0.04871 0.05869 0.04778 0.05154 0.04568 0.06076 0.05401 0.06166 0.04527 0.03699 0.04490
  0.03175 0.05102 0.05901 0.04389 0.04965 0.04420 0.04536 0.05340 0.05534 0.05250 0.04452 0.04153 0.04996 0.04443 0.05050 0.04056 0.05129 0.07355 0.04873 0.06882
  0.04439 0.04909 0.04468 0.04552 0.05442 0.04255 0.04898 0.05140 0.04820 0.04862 0.06398 0.05472 0.04334 0.06467 0.05229 0.05435 0.04313 0.04556 0.06637 0.03372
  0.05433 0.05098 0.05076 0.06159 0.03733 0.04944 0.04354 0.05289 0.05281 0.04473 0.05485 0.06478 0.03936 0.04703 0.04916 0.07122 0.04297 0.04919 0.03563 0.04742
  0.05145 0.05954 0.05186 0.06563 0.04170 0.03042 0.04401 0.04830 0.03911 0.05273 0.05122 0.04671 0.05391 0.05047 0.05147 0.05636 0.04612 0.05497 0.04696 0.05706
  0.04295 0.04604 0.05112 0.04490 0.05057 0.04550 0.05269 0.05043 0.04828 0.06888 0.04858 0.05570 0.04479 0.04312 0.04472 0.04401 0.06402 0.05263 0.05375 0.04731
  0.05456 0.04972 0.04889 0.05264 0.05824 0.05214 0.04830 0.05617 0.03742 0.04821 0.06839 0.03970 0.03926 0.05228 0.04378 0.05051 0.05686 0.04017 0.05158 0.05118
  0.04937 0.05167 0.05159 0.04262 0.07179 0.04082 0.06060 0.03941 0.05212 0.05049 0.03544 0.04178 0.06774 0.05387 0.05970 0.04985 0.05012 0.04356 0.03653 0.05094
  0.04535 0.04814 0.05305 0.06106 0.04016 0.05326 0.05224 0.06730 0.05980 0.04973 0.04620 0.05526 0.04201 0.04333 0.04952 0.04745 0.03387 0.06711 0.04911 0.03605
  0.05062 0.05299 0.04507 0.04252 0.05661 0.04978 0.05242 0.05250 0.04808 0.05040 0.05945 0.05292 0.04745 0.04802 0.05502 0.05160 0.04564 0.04344 0.05169 0.04377
  0.05432 0.05414 0.05779 0.03690 0.05524 0.05437 0.05241 0.03971 0.04186 0.05023 0.05561 0.04629 0.04789 0.04713 0.04085 0.05330 0.04564 0.05705 0.04593 0.06333
  0.03963 0.04747 0.05688 0.05103 0.04098 0.03778 0.06956 0.04665 0.04289 0.05711 0.06029 0.04778 0.05724 0.04722 0.04566 0.04280 0.04929 0.05878 0.04657 0.05437
  0.05294 0.06364 0.05021 0.05641 0.03804 0.04314 0.05183 0.03797 0.04597 0.04781 0.04474 0.05260 0.05217 0.05392 0.06416 0.04730 0.05394 0.04954 0.03479 0.05887
  0.05013 0.04091 0.06504 0.04437 0.06210 0.03791 0.03803 0.05871 0.04651 0.05528 0.06224 0.04330 0.03696 0.05381 0.05535 0.04661 0.05118 0.05229 0.05243 0.04685
  0.06454 0.04446 0.04557 0.04729 0.05457 0.04733 0.05780 0.05545 0.03986 0.05625 0.03691 0.04922 0.05276 0.05340 0.05097 0.05201 0.04921 0.04031 0.04972 0.05237



.. _here: ../user_guides/work_data_input.rst#loading-a-mini-batch-from-corpus
.. _parameters: ../api/api_lda.rst#class-tmlib-lda-online-vb-onlinevb
.. _default: ../user_guide.rst#stochastic-methods-for-learning-lda-from-large-corpora
.. _LdaLearning: ../api/api_lda.rst#class-tmlib-lda-ldalearning-ldalearning
