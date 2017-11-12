# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _dec2:

=========================================
 Decode binary region of interest
=========================================

An example of decode.Decoder.decode_roi.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from os.path import join
import matplotlib.pyplot as plt

from nilearn import plotting
from nltools.mask import create_sphere

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
# Create region of interest (ROI) image
# --------------------------------------
coords = [[-52, -56, 18]]
radii = [6] * len(coords)

roi_img = create_sphere(coords, radius=radii, mask=model.dataset.mask_img)
fig = plotting.plot_roi(roi_img, display_mode='ortho',
                        cut_coords=[-52, -56, 18],
                        draw_cross=False)

###############################################################################
# Decode ROI
# -----------
df, topic_weights = decoder.decode_roi(roi_img)

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
