# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _dec2:

========================================
 Decode unthresholded map
========================================

An example of decode.decode_continuous.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from os.path import join
import matplotlib.pyplot as plt

import nibabel as nib
from nilearn import plotting

from gclda.model import Model
from gclda.decode import decode_continuous
from gclda.utils import get_resource_path

###############################################################################
# Load model and initialize decoder
# ----------------------------------
model_file = join(get_resource_path(), 'models/Neurosynth2015Filtered2',
                  'model_200topics_2015Filtered2_10000iters.pklz')
model = Model.load(model_file)

###############################################################################
# Read in image to decode
# --------------------------------------
file_to_decode = '../data/faces_specificity_z.nii.gz'
img_to_decode = nib.load(file_to_decode)
fig = plotting.plot_stat_map(img_to_decode, display_mode='z',
                             threshold=3.290527,
                             cut_coords=[-28, -4, 20, 50])

###############################################################################
# Decode image
# -------------
df, topic_weights = decode_continuous(model, img_to_decode)

###############################################################################
# Get associated terms
# ---------------------
df = df.sort_values(by='Weight', ascending=False)
print(df.head(10))

###############################################################################
# Plot topic weights
# ------------------
fig2, ax2 = plt.subplots()
ax2.plot(topic_weights)
ax2.set_xlabel('Topic #')
ax2.set_ylabel('Weight')
fig2.show()
