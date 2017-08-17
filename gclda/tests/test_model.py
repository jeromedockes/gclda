# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Tests for GC-LDA model module.
"""
import sys
from glob import glob
from shutil import rmtree
from os import remove
from os.path import join, isfile
try:
    # 2.7
    from StringIO import StringIO
except ImportError:
    # 3+
    from io import StringIO

from gclda.model import Model
from gclda.dataset import Dataset
from gclda.tests.utils import get_test_data_path


def test_init():
    """Smoke test for Model class.
    """
    dataset_file = join(get_test_data_path(), 'gclda_dataset.pkl')
    dset = Dataset.load(dataset_file)
    model = Model(dset, n_topics=50, n_regions=1, symmetric=False,
                  alpha=.1, beta=.01, gamma=.01, delta=1.,
                  dobs=25, roi_size=10., seed_init=1)
    assert isinstance(model, Model)


def test_asymmetric():
    """Test running a model with symmetric ROIs.
    """
    dataset_file = join(get_test_data_path(), 'gclda_dataset.pkl')
    dset = Dataset.load(dataset_file)
    model = Model(dset, n_topics=50, n_regions=2, symmetric=False,
                  alpha=.1, beta=.01, gamma=.01, delta=1.,
                  dobs=25, roi_size=10., seed_init=1)
    initial_iter = model.iter
    model.run_complete_iteration()
    assert model.iter == initial_iter + 1


def test_symmetric():
    """Test running a model with symmetric ROIs.
    """
    dataset_file = join(get_test_data_path(), 'gclda_dataset.pkl')
    dset = Dataset.load(dataset_file)
    model = Model(dset, n_topics=50, n_regions=2, symmetric=True,
                  alpha=.1, beta=.01, gamma=.01, delta=1.,
                  dobs=25, roi_size=10., seed_init=1)
    initial_iter = model.iter
    model.run_complete_iteration()
    assert model.iter == initial_iter + 1


def test_run_iteration():
    """Test functions needed to run each iteration.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    model = Model.load(model_file)
    initial_iter = model.iter
    model.run_complete_iteration()
    assert model.iter == initial_iter + 1


def test_load_model():
    """Test gclda.model.Model.load.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    model = Model.load(model_file)
    assert isinstance(model, Model)


def test_load_model2():
    """Test gclda.model.Model.load with gzipped file.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pklz')
    model = Model.load(model_file)
    assert isinstance(model, Model)


def test_save_model():
    """Test gclda.model.Model.save.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    temp_file = join(get_test_data_path(), 'temp.pkl')
    model = Model.load(model_file)
    model.save(temp_file)
    file_found = isfile(temp_file)
    assert file_found

    # Perform cleanup
    remove(temp_file)


def test_save_model2():
    """Test gclda.model.Model.save with gzipped file.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pklz')
    temp_file = join(get_test_data_path(), 'temp.pklz')
    model = Model.load(model_file)
    model.save(temp_file)
    file_found = isfile(temp_file)
    assert file_found

    # Perform cleanup
    remove(temp_file)


def test_save_model_params():
    """Ensure appropriate files are created.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    temp_dir = join(get_test_data_path(), 'temp')

    model = Model.load(model_file)
    model.save_model_params(temp_dir, n_top_words=2)
    files_found = [isfile(join(temp_dir, 'Topic_X_Word_Probs.csv')),
                   isfile(join(temp_dir, 'Topic_X_Word_CountMatrix.csv')),
                   isfile(join(temp_dir, 'ActivationAssignments.csv'))]
    assert all(files_found)

    # Perform cleanup
    rmtree(temp_dir)

def test_save_topic_figures():
    """Writes out images for topics.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    temp_dir = join(get_test_data_path(), 'temp')

    model = Model.load(model_file)
    model.save_topic_figures(temp_dir, n_top_words=5)
    figures = glob(join(temp_dir, '*.png'))
    assert len(figures) == model.n_topics

    # Perform cleanup
    rmtree(temp_dir)


def test_display_model_summary():
    """Prints model information to the console.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    model = Model.load(model_file)

    captured_output = StringIO()  # Create StringIO object
    sys.stdout = captured_output  #  and redirect stdout.
    model.display_model_summary()  # Call unchanged function.
    sys.stdout = sys.__stdout__  # Reset redirect.

    assert len(captured_output.getvalue()) > 0
