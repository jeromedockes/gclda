# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _enc1:

=========================================
 Encode text into image
=========================================

An example of decode.Decoder.encode.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from os.path import join
import matplotlib.pyplot as plt

from nilearn import plotting

from gclda.model import Model
from gclda.decode import Decoder
from gclda.utils import get_resource_path

###############################################################################
# Load model and initialize decoder
# ----------------------------------
model_file = join(get_resource_path(), 'models/model_Neurosynth2015Filtered2_temp.pklz')
model = Model.load(model_file)
decoder = Decoder(model)

###############################################################################
# Encode text into image
# ----------------------
text = 'painful stimulation during a language task'
text_img, topic_weights = decoder.encode(text)

###############################################################################
# Show encoded image
# ---------------------
fig = plotting.plot_stat_map(text_img, display_mode='z',
                             threshold=0.00001,
                             cut_coords=[-2, 22, 44, 66])

###############################################################################
# Plot topic weights
# ------------------
fig2, ax2 = plt.subplots()
ax2.plot(topic_weights)
ax2.set_xlabel('Topic #')
ax2.set_ylabel('Weight')
fig2.show()
