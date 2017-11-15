

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
    mentalizing    0.019629
    emotional      0.018926
    social         0.011803
    mind           0.009781
    intentions     0.009441
    mental_states  0.008362
    intention      0.008273
    attribution    0.007748
    number         0.007725
    emotion        0.007672


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
    person         0.009575
    mentalizing    0.009553
    social         0.008168
    mental_states  0.007108
    emotional      0.006399
    situation      0.006275
    mind           0.006087
    default        0.005371
    mental         0.005205
    selectively    0.005099


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
    words        0.018120
    word         0.013042
    emotional    0.012641
    language     0.010534
    semantic     0.009955
    ambiguous    0.008103
    perception   0.006673
    mentalizing  0.006174
    meaning      0.006063
    knowledge    0.005689


**Total running time of the script:** ( 1 minutes  5.683 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_paper_decoding_discrete.py <plot_paper_decoding_discrete.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_paper_decoding_discrete.ipynb <plot_paper_decoding_discrete.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
