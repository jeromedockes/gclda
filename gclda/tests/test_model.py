# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Tests for GC-LDA model module.
"""
from os import remove
from os.path import join, isfile

from gclda.model import Model
from gclda.dataset import Dataset
from gclda.tests.utils import get_test_data_path


def test_init():
    """Smoke test for Model class."""
    dataset_file = join(get_test_data_path(), 'gclda_dataset.pkl')
    dset = Dataset.load(dataset_file)
    model = Model(dset, n_topics=50, n_regions=1, symmetric=False,
                  alpha=.1, beta=.01, gamma=.01, delta=1.,
                  dobs=25, roi_size=10., seed_init=1)
    assert isinstance(model, Model)


def test_load_model():
    """Ensure model can be loaded from pickled file.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    model = Model.load(model_file)
    assert isinstance(model, Model)


def test_save_model():
    """Ensure dataset can be saved to pickled file.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    temp_file = join(get_test_data_path(), 'temp.pkl')
    model = Model.load(model_file)
    model.save(temp_file)
    file_found = isfile(temp_file)
    assert file_found

    # Perform cleanup
    remove(temp_file)
