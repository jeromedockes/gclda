# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Tests for GC-LDA model module.
"""
from os.path import join, isdir

import matplotlib
import numpy as np
import nibabel as nib
from neurosynth.tests.utils import get_test_dataset
from gclda.utils import weight_priors
from gclda.tests.utils import get_test_data_path


def test_get_resource_path():
    """Test gclda.utils.get_resource_path.
    """
    from gclda.utils import get_resource_path
    assert isdir(get_resource_path())


def test_weight_priors():
    """Test gclda.utils.weight_priors
    For some reason `array_equal` fails, but `allclose` passes.
    """
    test = np.array([0.235, 0.245, 0.255, 0.265])
    t_p = np.array([1, 2, 3, 4])
    p_w = .1
    weighted_priors = weight_priors(t_p, p_w)
    assert np.allclose(weighted_priors, test)


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
