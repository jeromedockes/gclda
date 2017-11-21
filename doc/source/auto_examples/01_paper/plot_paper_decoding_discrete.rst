

.. _sphx_glr_auto_examples_01_paper_plot_paper_decoding_discrete.py:



.. _pap2:

=========================================
 Discrete decoding
=========================================

Decode ROIs from Rubin et al. (2017).



Start with the necessary imports
--------------------------------



.. code-block:: python

    from os.path import join

    import nibabel as nib
    from nilearn import plotting
    from nltools.mask import create_sphere

    from gclda.model import Model
    from gclda.decode import decode_roi
    from gclda.utils import get_resource_path







Load model and initialize decoder
----------------------------------



.. code-block:: python

    model_file = join(get_resource_path(), 'models/Neurosynth2015Filtered2',
                      'model_200topics_2015Filtered2_10000iters.pklz')
    model = Model.load(model_file)
    model.display_model_summary()

    # Create mask image
    mask_data = (model.dataset.mask_img.get_data()!=0).astype(int)
    affine = model.dataset.mask_img.affine
    mask = nib.Nifti1Image(mask_data, affine)





.. rst-class:: sphx-glr-script-out

 Out::

    --- Model Summary ---
     Current State:
             Current Iteration   = 10000
             Initialization Seed = 1
             Current Log-Likely  = -11268037.3695
     Model Hyper-Parameters:
             Symmetric = True
             n_topics  = 200
             n_regions = 2
             alpha     = 0.100
             beta      = 0.010
             gamma     = 0.010
             delta     = 1.000
             roi_size  = 50.000
             dobs      = 25
     Model Training-Data Information:
             Dataset Label                 = Neurosynth2015Filtered2
             Word-Tokens (n_word_tokens)   = 520492
             Peak-Tokens (n_peak_tokens)   = 400801
             Word-Types (n_word_labels)    = 6755
             Documents (n_docs)            = 11362
             Peak-Dimensions (n_peak_dims) = 3


Temporoparietal seed
--------------------------------------



.. code-block:: python

    coords = [[-52, -56, 18]]
    radii = [6] * len(coords)

    roi_img = create_sphere(coords, radius=radii, mask=mask)
    fig = plotting.plot_roi(roi_img, display_mode='ortho',
                            cut_coords=[-52, -56, 18],
                            draw_cross=False)

    df, _ = decode_roi(model, roi_img)
    df = df.sort_values(by='Weight', ascending=False)
    print(df.head(10))




.. image:: /auto_examples/01_paper/images/sphx_glr_plot_paper_decoding_discrete_001.png
    :align: center


.. rst-class:: sphx-glr-script-out

 Out::

    Weight
    Term                   
    mentalizing    2.414315
    emotional      2.327844
    social         1.451740
    mind           1.203048
    intentions     1.161271
    mental_states  1.028513
    intention      1.017633
    attribution    0.952983
    number         0.950172
    emotion        0.943711


Temporoparietal, medial parietal, and dorsomedial prefrontal seeds
------------------------------------------------------------------



.. code-block:: python

    coords = [[-56, -52, 18],
              [0, -58, 38],
              [4, 54, 26]]
    radii = [6] * len(coords)

    roi_img = create_sphere(coords, radius=radii, mask=mask)
    fig = plotting.plot_roi(roi_img, display_mode='ortho',
                            cut_coords=[-52, -56, 18],
                            draw_cross=False)

    df, _ = decode_roi(model, roi_img)
    df = df.sort_values(by='Weight', ascending=False)
    print(df.head(10))




.. image:: /auto_examples/01_paper/images/sphx_glr_plot_paper_decoding_discrete_002.png
    :align: center


.. rst-class:: sphx-glr-script-out

 Out::

    Weight
    Term                   
    person         3.533031
    mentalizing    3.525109
    social         3.014061
    mental_states  2.622679
    emotional      2.361156
    situation      2.315302
    mind           2.246191
    default        1.981821
    mental         1.920779
    selectively    1.881442


Temporoparietal, left superior temporal sulcus, and left inferior frontal gyrus seeds
-------------------------------------------------------------------------------------



.. code-block:: python

    coords = [[-56, -52, 18],
              [-54, -40, 0],
              [-50, 26, 6]]
    radii = [6] * len(coords)

    roi_img = create_sphere(coords, radius=radii, mask=mask)
    fig = plotting.plot_roi(roi_img, display_mode='ortho',
                            cut_coords=[-52, -56, 18],
                            draw_cross=False)

    df, _ = decode_roi(model, roi_img)
    df = df.sort_values(by='Weight', ascending=False)
    print(df.head(10))



.. image:: /auto_examples/01_paper/images/sphx_glr_plot_paper_decoding_discrete_003.png
    :align: center


.. rst-class:: sphx-glr-script-out

 Out::

    Weight
    Term                 
    words        6.686298
    word         4.812359
    emotional    4.664474
    language     3.887118
    semantic     3.673409
    ambiguous    2.989912
    perception   2.462263
    mentalizing  2.278094
    meaning      2.237231
    knowledge    2.099289


**Total running time of the script:** ( 1 minutes  5.327 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_paper_decoding_discrete.py <plot_paper_decoding_discrete.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_paper_decoding_discrete.ipynb <plot_paper_decoding_discrete.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
