# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _pap3:

=========================================
 Encoding
=========================================

Encode text from Rubin et al. (2017) into images.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from os.path import join

from nilearn import plotting

from gclda.model import Model
from gclda.decode import encode
from gclda.utils import get_resource_path

###############################################################################
# Load model
# ----------------------------------
model_file = join(get_resource_path(), 'models/model_Neurosynth2015Filtered2_temp.pklz')
model = Model.load(model_file)
model.display_model_summary()

###############################################################################
# First text
# ----------------------
text = 'motor'
text_img, _ = encode(model, text)
fig = plotting.plot_stat_map(text_img, display_mode='z',
                             threshold=0.00001,
                             cut_coords=[-18, 4, 32, 60])

###############################################################################
# Second text
# ---------------------
text = 'effort difficult demands'
text_img, _ = encode(model, text)
fig = plotting.plot_stat_map(text_img, display_mode='z',
                             threshold=0.00001,
                             cut_coords=[-30, -4, 26, 50])

###############################################################################
# Third text
# ------------------
text = 'painful stimulation during a language task'
text_img, _ = encode(model, text)
fig = plotting.plot_stat_map(text_img, display_mode='z',
                             threshold=0.00001,
                             cut_coords=[-2, 22, 44, 66])
