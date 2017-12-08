

.. _sphx_glr_auto_examples_02_general_plot_show_topic_figures.py:



.. _plot1:

=========================================
 Show topic distributions
=========================================

Plot topic figures and show them.



Start with the necessary imports
--------------------------------



.. code-block:: python

    from os.path import join

    import numpy as np
    import pandas as pd
    from nilearn import plotting
    from nilearn.masking import unmask

    from gclda.model import Model
    from gclda.utils import get_resource_path







Load model
----------------------------------



.. code-block:: python

    model_file = join(get_resource_path(), 'models/Neurosynth2015Filtered2',
                      'model_200topics_2015Filtered2_10000iters.pklz')
    model = Model.load(model_file)







Get spatial probability
-----------------------



.. code-block:: python

    p_voxel_g_topic, _ = model.get_spatial_probs()







Show topic 10
-----------------------



.. code-block:: python

    topic_no = 10
    img = unmask(p_voxel_g_topic[:, topic_no], model.dataset.mask_img)

    # Get strings giving top K words and probs for the current topic
    wprobs = model.n_word_tokens_word_by_topic[:, topic_no] + model.beta
    wprobs = wprobs / np.sum(wprobs)
    word_probs = zip(*[model.dataset.word_labels, wprobs])
    df = pd.DataFrame(columns=['term', 'probability'], data=word_probs)
    df = df.sort_values(by='probability', ascending=False).reset_index(drop=True)
    print df.head(12)

    fig = plotting.plot_stat_map(img, display_mode='z')




.. image:: /auto_examples/02_general/images/sphx_glr_plot_show_topic_figures_001.png
    :align: center


.. rst-class:: sphx-glr-script-out

 Out::

    term  probability
    0   cognitive_control     0.119603
    1             demands     0.071321
    2           executive     0.062462
    3           difficult     0.033669
    4                easy     0.024810
    5   executive_control     0.023924
    6           requiring     0.019938
    7           selection     0.018609
    8              number     0.016837
    9                hard     0.015951
    10             manner     0.015508
    11           matching     0.015065


Show topic 59
------------------------



.. code-block:: python

    topic_no = 58
    img = unmask(p_voxel_g_topic[:, topic_no], model.dataset.mask_img)

    # Get strings giving top K words and probs for the current topic
    wprobs = model.n_word_tokens_word_by_topic[:, topic_no] + model.beta
    wprobs = wprobs / np.sum(wprobs)
    word_probs = zip(*[model.dataset.word_labels, wprobs])
    df = pd.DataFrame(columns=['term', 'probability'], data=word_probs)
    df = df.sort_values(by='probability', ascending=False).reset_index(drop=True)
    print df.head(12)

    fig = plotting.plot_stat_map(img, display_mode='z')




.. image:: /auto_examples/02_general/images/sphx_glr_plot_show_topic_figures_002.png
    :align: center


.. rst-class:: sphx-glr-script-out

 Out::

    term  probability
    0    somatosensory     0.135916
    1          tactile     0.108811
    2            touch     0.088289
    3      stimulation     0.048406
    4          sensory     0.039113
    5   discrimination     0.027496
    6             body     0.022850
    7     vibrotactile     0.014331
    8          touched     0.013556
    9           finger     0.013169
    10            skin     0.012395
    11       roughness     0.012008


Show topic 150
-----------------------



.. code-block:: python

    topic_no = 149
    img = unmask(p_voxel_g_topic[:, topic_no], model.dataset.mask_img)

    # Get strings giving top K words and probs for the current topic
    wprobs = model.n_word_tokens_word_by_topic[:, topic_no] + model.beta
    wprobs = wprobs / np.sum(wprobs)
    word_probs = zip(*[model.dataset.word_labels, wprobs])
    df = pd.DataFrame(columns=['term', 'probability'], data=word_probs)
    df = df.sort_values(by='probability', ascending=False).reset_index(drop=True)
    print df.head(12)

    fig = plotting.plot_stat_map(img, display_mode='z')



.. image:: /auto_examples/02_general/images/sphx_glr_plot_show_topic_figures_003.png
    :align: center


.. rst-class:: sphx-glr-script-out

 Out::

    term  probability
    0    emotional     0.120525
    1      emotion     0.064047
    2        faces     0.055130
    3    affective     0.027836
    4         fear     0.025134
    5      fearful     0.024594
    6     pictures     0.015136
    7   regulation     0.014055
    8       affect     0.013784
    9        angry     0.012704
    10      threat     0.012704
    11     ratings     0.012704


**Total running time of the script:** ( 0 minutes  41.400 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_show_topic_figures.py <plot_show_topic_figures.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_show_topic_figures.ipynb <plot_show_topic_figures.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
