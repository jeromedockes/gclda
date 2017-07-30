# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Utility functions.
"""
from __future__ import print_function, division

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def plot_brain(data, model):
    """
    Create a figure of a brain.
    """
    # Encoding isn't working yet.
    template_image = model.dataset.masker.volume.get_data().astype(float)

    ijk_extrema = np.vstack((np.array([0, 0, 0]), np.array(template_image.shape)))
    xyz_extrema = nib.affines.apply_affine(model.dataset.masker.volume.affine,
                                           ijk_extrema).astype(int)

    template_image[template_image == 0] = np.nan
    data[data == 0] = np.nan

    mid_i, mid_j, mid_k = (np.array(data.shape) / 2).astype(int)
    mid_x, mid_y, mid_z = nib.affines.apply_affine(model.dataset.masker.volume.affine,
                                                   np.array([mid_i, mid_j, mid_k])).astype(int)

    fig = plt.figure(figsize=(10, 10), dpi=200)

    ax1 = fig.add_subplot(221)
    ax1.imshow(np.rot90(template_image[:, mid_j, :], k=1), 'gray',
               extent=[xyz_extrema[0, 0], xyz_extrema[1, 0],
                       xyz_extrema[0, 2], xyz_extrema[1, 2]])
    ax1.imshow(np.rot90(data[:, mid_j, :], k=1), alpha=0.7,
               extent=[xyz_extrema[0, 0], xyz_extrema[1, 0],
                       xyz_extrema[0, 2], xyz_extrema[1, 2]])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_xlim((-80, 80))
    ax1.set_xticks(range(-80, 81, 20))
    ax1.set_ylim((-60, 90))
    ax1.set_yticks(range(-60, 91, 15))
    ax1.set_title('Coronal View: Y = {0}'.format(mid_y))

    ax2 = fig.add_subplot(222)
    ax2.imshow(np.rot90(template_image[mid_i, :, :], k=1), 'gray',
               extent=[xyz_extrema[0, 1], xyz_extrema[1, 1],
                       xyz_extrema[0, 2], xyz_extrema[1, 2]])
    ax2.imshow(np.rot90(data[mid_i, :, :], k=1), alpha=0.7,
               extent=[xyz_extrema[0, 1], xyz_extrema[1, 1],
                       xyz_extrema[0, 2], xyz_extrema[1, 2]])
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.set_xlim((-110, 80))
    ax2.set_xticks(range(-110, 81, 20))
    ax2.set_ylim((-75, 90))
    ax2.set_yticks(range(-75, 91, 15))
    ax2.set_title('Sagittal View: X = {0}'.format(mid_x))

    ax3 = fig.add_subplot(223)
    ax3.imshow(np.rot90(template_image[:, :, mid_k], k=1), 'gray',
               extent=[xyz_extrema[0, 0], xyz_extrema[1, 0],
                       xyz_extrema[0, 1], xyz_extrema[1, 1]])
    ax3.imshow(np.rot90(data[:, :, mid_k], k=1), alpha=0.7,
               extent=[xyz_extrema[0, 0], xyz_extrema[1, 0],
                       xyz_extrema[0, 1], xyz_extrema[1, 1]])
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_xlim((-80, 80))
    ax3.set_yticks(range(-80, 81, 20))
    ax3.set_ylim((-110, 80))
    ax3.set_yticks(range(-110, 81, 20))
    ax3.set_title('Axial View: Z = {0}'.format(mid_z))
    return fig
