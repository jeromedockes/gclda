# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Tests for GC-LDA dataset module.
"""
from shutil import rmtree
from os import remove
from os.path import join, isfile

import neurosynth
from gclda.dataset import Dataset
from gclda.tests.utils import get_test_data_path


def test_import_from_counts():
    """Ensure that Dataset files can be generated using counts file.
    """
    from gclda.dataset import import_neurosynth
    counts_file = join(get_test_data_path(), 'feature_counts.txt')
    ns_dset_file = join(get_test_data_path(), 'neurosynth_dataset.pkl')
    temp_dir = join(get_test_data_path(), 'temp')

    ns_dset = neurosynth.Dataset.load(ns_dset_file)
    import_neurosynth(ns_dset, 'temp', out_dir=get_test_data_path(),
                      counts_file=counts_file)
    files_found = [isfile(join(temp_dir, 'pmids.txt')),
                   isfile(join(temp_dir, 'peak_indices.txt')),
                   isfile(join(temp_dir, 'word_labels.txt')),
                   isfile(join(temp_dir, 'word_indices.txt'))]
    assert all(files_found)

    # Perform cleanup
    rmtree(temp_dir)


def test_import_from_abstracts():
    """Ensure that Dataset files can be generated using abstracts file.
    """
    from gclda.dataset import import_neurosynth
    abstracts_file = join(get_test_data_path(), 'abstracts.csv')
    ns_dset_file = join(get_test_data_path(), 'neurosynth_dataset.pkl')
    temp_dir = join(get_test_data_path(), 'temp')

    ns_dset = neurosynth.Dataset.load(ns_dset_file)
    import_neurosynth(ns_dset, temp_dir, out_dir=get_test_data_path(),
                      abstracts_file=abstracts_file)
    files_found = [isfile(join(temp_dir, 'pmids.txt')),
                   isfile(join(temp_dir, 'peak_indices.txt')),
                   isfile(join(temp_dir, 'word_labels.txt')),
                   isfile(join(temp_dir, 'word_indices.txt'))]
    assert all(files_found)

    # Perform cleanup
    rmtree(temp_dir)


def test_import_from_email():
    """Ensure that Dataset files can be generated using email.
    """
    from gclda.dataset import import_neurosynth
    email = 'tsalo006@fiu.edu'
    ns_dset_file = join(get_test_data_path(), 'neurosynth_dataset.pkl')
    temp_dir = join(get_test_data_path(), 'temp')

    ns_dset = neurosynth.Dataset.load(ns_dset_file)
    import_neurosynth(ns_dset, 'temp', out_dir=get_test_data_path(),
                      email=email)
    files_found = [isfile(join(temp_dir, 'pmids.txt')),
                   isfile(join(temp_dir, 'peak_indices.txt')),
                   isfile(join(temp_dir, 'word_labels.txt')),
                   isfile(join(temp_dir, 'word_indices.txt'))]
    assert all(files_found)

    # Perform cleanup
    rmtree(temp_dir)


def test_init():
    """Smoke test for Dataset class.
    """
    dataset_dir = get_test_data_path()
    dset = Dataset('dataset_files', dataset_dir)
    assert isinstance(dset, Dataset)


def test_load_dataset():
    """Ensure dataset can be loaded from pickled file.
    """
    dataset_file = join(get_test_data_path(), 'gclda_dataset.pkl')
    dset = Dataset.load(dataset_file)
    assert isinstance(dset, Dataset)


def test_save_dataset():
    """Ensure dataset can be saved to pickled file.
    """
    dataset_file = join(get_test_data_path(), 'gclda_dataset.pkl')
    temp_file = join(get_test_data_path(), 'temp.pkl')
    dset = Dataset.load(dataset_file)
    dset.save(temp_file)
    file_found = isfile(temp_file)
    assert file_found

    # Perform cleanup
    remove(temp_file)
