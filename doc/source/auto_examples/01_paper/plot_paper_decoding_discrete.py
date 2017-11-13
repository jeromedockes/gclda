# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _pap2:

=========================================
 Discrete decoding
=========================================

Decode ROIs from Rubin et al. (2017).

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from os.path import join

import nibabel as nib
from nilearn import plotting
from nltools.mask import create_sphere

from gclda.model import Model
from gclda.decode import decode_roi
from gclda.utils import get_resource_path

###############################################################################
# Load model and initialize decoder
# ----------------------------------
model_file = join(get_resource_path(), 'models/model_Neurosynth2015Filtered2_temp.pklz')
model = Model.load(model_file)
model.display_model_summary()

# Create mask image
mask_data = (model.dataset.mask_img.get_data()!=0).astype(int)
affine = model.dataset.mask_img.affine
mask = nib.Nifti1Image(mask_data, affine)

###############################################################################
# Temporoparietal seed
# --------------------------------------
coords = [[-52, -56, 18]]
radii = [6] * len(coords)

roi_img = create_sphere(coords, radius=radii, mask=mask)
fig = plotting.plot_roi(roi_img, display_mode='ortho',
                        cut_coords=[-52, -56, 18],
                        draw_cross=False)

df, _ = decode_roi(model, roi_img)
df = df.sort_values(by='Weight', ascending=False)
print(df.head(10))

###############################################################################
# Temporoparietal, medial parietal, and dorsomedial prefrontal seeds
# ------------------------------------------------------------------
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

#######################################################################################
# Temporoparietal, left superior temporal sulcus, and left inferior frontal gyrus seeds
# -------------------------------------------------------------------------------------
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
