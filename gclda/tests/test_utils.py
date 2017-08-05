# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Tests for GC-LDA model module.
"""
from os.path import join

import matplotlib
import nibabel as nib
from neurosynth.tests.utils import get_test_dataset
from gclda.tests.utils import get_test_data_path


def test_plot_brain():
    """Ensure that utils.plot_brain returns a matplotlib figure.
    """
    from gclda.utils import plot_brain
    ns_dset = get_test_dataset()
    underlay = ns_dset.masker.volume

    test_file = join(get_test_data_path(), 'continuous.nii.gz')
    test_data = nib.load(test_file).get_data()

    fig = plot_brain(test_data, underlay)
    assert isinstance(fig, matplotlib.figure.Figure)
