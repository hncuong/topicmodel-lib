============================
3. How to save or load model
============================

..Contents::

---------------------------------------------
Save model (:math:`\lambda` or :math:`\beta`)
---------------------------------------------
After learning model **obj_model** as above. We can save this result as follow:

::
    
  obj_model.save(file_name, file_type='text')

The result model is saved in file named *file_name* with format file is text. The default format is binary file if we remove the file_type parameter. 

Moreover, we can save the model and some statistics like the study time, topic mixtures :math:`\theta`, the sparsity of document [8]_ in the running process of the algorithm so that we can have necessary comparison and assessment. For example with VB method:

::

  obj_onlvb = OnlineVB(training_data, save_model_every=2, compute_sparsity_every=2, save_statistic=True, save_top_words_every=2, num_top_words=20, model_folder='model_vb')

This means after 2 iterations, the model, time of E-step, M-step and document sparsity is saved into files. All of this files is in the folder 'model_vb' named by user.

-----------------------
Load model from a file
-----------------------

Assume that :math:`\lambda` or :math:`\beta` is saved in a file has path *model_file_path*. Loading is supported with 2 type of file: text (.txt) and binary (.npy). 

::

  from tmlib.lda.ldamodel import LdaModel

  obj_model = LdaModel(num_terms, num_topics)
  obj_model.load(model_file_path)

The num_terms and num_topics are 2 parameters which are determined by user. For example, if we combine this section with tutorial learning, we can set

  num_terms = training_data.get_num_terms()
  num_topics = obj_onlvb.num_topics      # for example with Online-VB method

------------------------
Save top words of topics
------------------------

Display to the screen

:: 
    
  # print 10 topics, top 20 words which have the highest probability will be displayed in each topic
  obj_model.print_top_words(20, vocab_file_path, show_topics=10)

Save into a file

::

  obj_model.print_top_words(20, vocab_file_path, show_topics=30, result_file='topics.txt')

