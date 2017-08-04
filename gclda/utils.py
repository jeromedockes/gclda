# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Utility functions.
"""
from __future__ import print_function, division

from builtins import range
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import matplotlib.pyplot as plt


def plot_brain(data, underlay, x=0, y=0, z=0):
    """
    Create a figure of a brain.

    Generates a 2x2 figure of brain slices in the axial, coronal, and sagittal
    directions (bottom right subplot is left empty). A functional image is
    overlaid onto the anatomical template from a nibabel image.

    Parameters
    ----------
    data : :obj:`numpy.ndarray` of :obj:`numpy.float32`
        3D array of voxel values to plot over structural underlay.

    underlay : :obj:`nibabel.Nifti1Image`
        Structural scan to use as underlay in figure.

    x : int, optional
        X-coordinate in stereotactic space to use for sagittal view in figure.
        Default is 0.

    y : int, optional
        Y-coordinate in stereotactic space to use for coronal view in figure.
        Default is 0.

    z : int, optional
        Z-coordinate in stereotactic space to use for axial view in figure.
        Default is 0.

    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        A 2x2 figure with three views of the brain (axial, coronal, and
        sagittal).

    """
    template_data = underlay.get_data().astype(float)
    if data.shape != template_data.shape:
        raise Exception('Input dimensions {0} do not match template '
                        'dimensions {1}.'.format(data.shape, template_data.shape))

    ijk_extrema = np.vstack((np.array([0, 0, 0]), np.array(template_data.shape)))
    xyz_extrema = nib.affines.apply_affine(underlay.affine, ijk_extrema).astype(int)

    template_data[template_data == 0] = np.nan
    data[data == 0] = np.nan

    show_i, show_j, show_k = nib.affines.apply_affine(npl.inv(underlay.affine),
                                                      np.array([x, y, z])).astype(int)

    fig = plt.figure(figsize=(10, 10), dpi=200)

    ax1 = fig.add_subplot(221)
    ax1.imshow(np.rot90(template_data[:, show_j, :], k=1), 'gray',
               extent=[xyz_extrema[0, 0], xyz_extrema[1, 0],
                       xyz_extrema[0, 2], xyz_extrema[1, 2]])
    ax1.imshow(np.rot90(data[:, show_j, :], k=1), alpha=0.7,
               extent=[xyz_extrema[0, 0], xyz_extrema[1, 0],
                       xyz_extrema[0, 2], xyz_extrema[1, 2]])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_xlim((-80, 80))
    ax1.set_xticks(list(range(-80, 81, 20)))
    ax1.set_ylim((-60, 90))
    ax1.set_yticks(list(range(-60, 91, 15)))
    ax1.set_title('Coronal View: Y = {0}'.format(y))

    ax2 = fig.add_subplot(222)
    ax2.imshow(np.rot90(template_data[show_i, :, :], k=1), 'gray',
               extent=[xyz_extrema[0, 1], xyz_extrema[1, 1],
                       xyz_extrema[0, 2], xyz_extrema[1, 2]])
    ax2.imshow(np.rot90(data[show_i, :, :], k=1), alpha=0.7,
               extent=[xyz_extrema[0, 1], xyz_extrema[1, 1],
                       xyz_extrema[0, 2], xyz_extrema[1, 2]])
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.set_xlim((-110, 80))
    ax2.set_xticks(list(range(-110, 81, 20)))
    ax2.set_ylim((-75, 90))
    ax2.set_yticks(list(range(-75, 91, 15)))
    ax2.set_title('Sagittal View: X = {0}'.format(x))

    ax3 = fig.add_subplot(223)
    ax3.imshow(np.rot90(template_data[:, :, show_k], k=1), 'gray',
               extent=[xyz_extrema[0, 0], xyz_extrema[1, 0],
                       xyz_extrema[0, 1], xyz_extrema[1, 1]])
    ax3.imshow(np.rot90(data[:, :, show_k], k=1), alpha=0.7,
               extent=[xyz_extrema[0, 0], xyz_extrema[1, 0],
                       xyz_extrema[0, 1], xyz_extrema[1, 1]])
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_xlim((-80, 80))
    ax3.set_yticks(list(range(-80, 81, 20)))
    ax3.set_ylim((-110, 80))
    ax3.set_yticks(list(range(-110, 81, 20)))
    ax3.set_title('Axial View: Z = {0}'.format(z))
    return fig
