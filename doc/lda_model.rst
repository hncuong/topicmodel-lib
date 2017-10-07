=========
LDA Model
=========

--------------
class LdaModel
--------------

::

  tmlib.lda.LdaModel(num_terms=None, num_topics=None, random_type=0)

This class provides some function to help you manipulate model (:math:`\lambda` or :math:`\beta`): you can save model, load model or display words of topics...

Parameters
===========

- **num_terms**: int, default: None
  
  number of words in vocabulary file
- **num_topics**: int, default: None

  number of topics

- **random_type**: int, default: 0

  Initialize randomly array of :math:`\lambda` (or :math:`\beta`) (size num_topics x num_terms). If random_type = 0, model is initialized with uniform distribution. Otherwise, initialized with gamma distribution

Attributes
==========

- **num_terms**: int
- **num_topics**: int

- **model**: array 2 dimentions (num_topics x num_terms)
  
  :math:`\lambda` or :math:`\beta`. Depend on the learning method: :math:`\lambda` in case Online VB, Online CVB0, Online CGS, Streaming VB and :math:`\beta` for Online OPE, Online FW, Streaming OPE, Streaming FW, ML-CGS, ML-OPE, ML-FW

- **presence_score**: a measure allow us to rank the topics. If a topic has the high value of presence_score, it means that corpus is related to that topic very much


Methods
=======

- __init__(*num_terms=None, num_topics=None, random_type=0*)
- **normalize** ()

  Used for estimating :math:`\beta` from :math:`\lambda`. This function is usually used for regularized methods ML-CGS, ML-FW, ML-OPE  

- **print_top_words** (self, num_words, vocab_file, show_topics=None, display_result=None, type='word', distribution=False)

  Display words of topics on the screen or save into file

  - **Parameters**: 

    - **num_words**: int, 
    
      number of words of each topic is displayed
    - **vocab_file**: string, 
    
      path of file vocabulary
    - **show_topics**: int, default: None

      number of topics is displayed. By default, all of topics are displayed
    - **display_result**: string, default: None

      path of file to save words into, or 'screen' if you want to display topics on the screen. By default, if display_result=None, nothing happen and the method return the result as list python

    - **type**: string, default: 'word'

      You can print index of word in the vocabulary file by set type='index'. If type='word' by default, the result returned is words (type string)

    - **distribution**:

      If you only want to display words, then distribution=False. But you can display words along with term distribution of it by set distribution = True  

- **load_model** (path_file)

  loading the learned model (:math:`\lambda` or :math:`\beta`) from file named *path_file*
  
- **save_model** (path_file, file_type='binary')

  saving model into a file named path_file. By default, the type of file is binary. We can change type of file to text by set file_type='txt'

- **save** (path_file)

  you can save the object by using this function. All attributes of object LdaModel are saved in path_file and after that, you can restore object by using **load** function

- **load** (path_file)

  load the attributes from path_file to restore object LdaModel.

-------
Example
-------

:: 
    
    from tmlib.lda import OnlineOPE
    from tmlib.datasets import DataSet

    # data preparation
    data = DataSet(data_path='data/ap_train.txt', batch_size=100, passes=5, shuffle_every=2, vocab_file='data/vocab.txt')
    # learning
    onl_ope = OnlineOPE(data=data, num_topics=20, alpha=0.2)
    model = onl_vb.learn_model()
    # save model (beta or lambda)
    model.save_model('topic_distribution.npy')
    # display top 10 words of each topic
    model.print_top_words(10, data.vocab_file, display_result='screen')

Topics:

::

    topic 0: two, new, people, i, years, first, officials, time, fire, day, 
    topic 1: israel, minister, prime, vietnam, thatcher, party, opec, ministers, demjanjuk, labor, 
    topic 2: million, percent, futures, year, market, bank, analysts, new, cbs, nbc, 
    topic 3: dukakis, jackson, democratic, presidential, campaign, candidates, candidate, vote, voters, delegates, 
    topic 4: million, company, new, billion, inc, corp, board, year, court, federal, 
    topic 5: bush, united, states, president, trade, billion, house, congress, new, budget, 
    topic 6: stock, market, dollar, trading, exchange, yen, prices, late, index, rose, 
    topic 7: korean, korea, city, village, police, north, st, traffic, koreas, citys, 
    topic 8: police, people, killed, two, government, army, military, officials, three, city, 
    topic 9: south, africa, african, black, elections, party, national, war, mandela, blacks, 
    topic 10: states, united, nicaragua, noriega, drug, contras, court, coup, humphrey, manila, 
    topic 11: reagan, china, nuclear, study, b, prisoners, fitzwater, researchers, games, animals, 
    topic 12: i, new, people, years, percent, year, last, state, time, two, 
    topic 13: trial, case, prison, charges, convicted, jury, attorney, guilty, sentence, prosecutors, 
    topic 14: rain, northern, texas, inches, california, central, damage, santa, hospital, valley, 
    topic 15: soviet, government, gorbachev, union, party, president, political, two, news, people, 
    topic 16: service, offer, court, companies, firm, ruling, information, appeals, operations, services, 
    topic 17: water, care, homeless, environmental, pollution, fair, species, air, disaster, farm, 
    topic 18: percent, year, cents, oil, prices, west, german, rate, sales, price, 
    topic 19: air, plane, flight, two, iraq, soviet, force, kuwait, airport, iraqi,

If you change the last statement to:

::

  model.print_top_words(5, data.vocab_file, display_result='screen', distribution=True)

::

  topic 0: (two 1006.872532), (new 997.382525), (people 957.472761), (i 847.205429), (years 793.221432), 
  topic 1: (israel 280.810816), (minister 264.384617), (prime 236.849663), (vietnam 227.907314), (thatcher 184.359115), 
  topic 2: (million 990.763364), (percent 990.581342), (futures 676.483861), (year 554.327385), (market 487.330349), 
  topic 3: (dukakis 1443.205724), (jackson 961.280235), (democratic 666.250053), (presidential 530.063395), (campaign 374.716751), 
  topic 4: (million 2094.650569), (company 1817.952267), (new 1233.623336), (billion 1064.094612), (inc 940.197448), 
  topic 5: (bush 3039.197872), (united 2500.479748), (states 2209.172288), (president 2184.088408), (trade 1752.369137), 
  topic 6: (stock 1781.654725), (market 1612.880852), (dollar 1321.883897), (trading 1136.304384), (exchange 984.091480), 
  topic 7: (korean 315.153683), (korea 250.382257), (city 236.793215), (village 198.821685), (police 151.618394), 
  topic 8: (police 4616.893623), (people 2455.625615), (killed 1695.193209), (two 1638.251890), (government 1423.613710), 
  topic 9: (south 1367.724403), (africa 749.935889), (african 713.020710), (black 693.012008), (elections 511.215701), 
  topic 10: (states 472.340775), (united 281.539123), (nicaragua 261.313058), (noriega 209.555378), (drug 192.748472), 
  topic 11: (reagan 379.215621), (china 323.023153), (nuclear 283.032322), (study 275.714702), (b 244.181802), 
  topic 12: (i 6295.795578), (new 3049.691470), (people 2959.973828), (years 2682.041759), (percent 2517.415288), 
  topic 13: (trial 937.304453), (case 623.345300), (prison 602.181133), (charges 586.927118), (convicted 564.307367), 
  topic 14: (rain 348.371840), (northern 346.465730), (texas 323.902501), (inches 321.175532), (california 297.998834), 
  topic 15: (soviet 3327.753735), (government 1527.465015), (gorbachev 1422.083698), (union 1335.393499), (party 1296.856500), 
  topic 16: (service 540.989098), (offer 479.423750), (court 448.157831), (companies 325.420266), (firm 258.895112), 
  topic 17: (water 470.181853), (care 214.000670), (homeless 211.299046), (environmental 189.202329), (pollution 186.071190), 
  topic 18: (percent 4348.697399), (year 1170.209529), (cents 1135.822631), (oil 1093.987126), (prices 934.051475), 
  topic 19: (air 1054.797512), (plane 793.580865), (flight 762.382746), (two 597.540968), (iraq 563.872035), 
