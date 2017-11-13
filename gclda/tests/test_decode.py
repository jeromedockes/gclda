# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Tests for GC-LDA decode module.
"""
from os.path import join

import nibabel as nib

from gclda.model import Model
from gclda.decode import decode_roi, decode_continuous, encode
from gclda.tests.utils import get_test_data_path


def test_decode_roi_from_file():
    """Acceptance test of ROI-based decoding with str input.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    roi_file = join(get_test_data_path(), 'roi.nii.gz')
    model = Model.load(model_file)
    decoded_df, _ = decode_roi(model, roi_file)
    assert decoded_df.shape[0] == model.n_word_labels


def test_decode_roi_from_img():
    """Acceptance test of ROI-based decoding with nibabel.Nifti1Image input.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    roi_file = join(get_test_data_path(), 'roi.nii.gz')
    roi_img = nib.load(roi_file)
    model = Model.load(model_file)
    decoded_df, _ = decode_roi(model, roi_img)
    assert decoded_df.shape[0] == model.n_word_labels


def test_decode_roi_with_priors():
    """Acceptance test of ROI-based decoding with topic priors.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    roi_file = join(get_test_data_path(), 'roi.nii.gz')
    model = Model.load(model_file)
    _, priors = decode_roi(model, roi_file)
    decoded_df, _ = decode_roi(model, roi_file, topic_priors=priors)
    assert decoded_df.shape[0] == model.n_word_labels


def test_decode_continuous_from_file():
    """Acceptance test of continuous image-based decoding with str input.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    continuous_file = join(get_test_data_path(), 'continuous.nii.gz')
    model = Model.load(model_file)
    decoded_df, _ = decode_continuous(model, continuous_file)
    assert decoded_df.shape[0] == model.n_word_labels


def test_decode_continuous_from_img():
    """Acceptance test of continuous image-based decoding with
    nibabel.Nifti1Image input.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    continuous_file = join(get_test_data_path(), 'continuous.nii.gz')
    continuous_img = nib.load(continuous_file)
    model = Model.load(model_file)
    decoded_df, _ = decode_continuous(model, continuous_img)
    assert decoded_df.shape[0] == model.n_word_labels


def test_decode_continuous_with_priors():
    """Acceptance test of continuous image-based decoding with topic priors.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    continuous_file = join(get_test_data_path(), 'continuous.nii.gz')
    model = Model.load(model_file)
    _, priors = decode_continuous(model, continuous_file)
    decoded_df, _ = decode_continuous(model, continuous_file, topic_priors=priors)
    assert decoded_df.shape[0] == model.n_word_labels


def test_encode_from_str():
    """Acceptance test of test-to-image encoding with str input.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    text = 'anterior insula was analyzed'
    model = Model.load(model_file)
    encoded_img, _ = encode(model, text)
    assert encoded_img.shape == model.dataset.mask_img.shape


def test_encode_from_list():
    """Acceptance test of test-to-image encoding with list input.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    text = ['anterior', 'insula', 'was', 'analyzed']
    model = Model.load(model_file)
    encoded_img, _ = encode(model, text)
    assert encoded_img.shape == model.dataset.mask_img.shape


def test_encode_with_priors():
    """Acceptance test of test-to-image encoding.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    text = 'anterior insula was analyzed'
    model = Model.load(model_file)
    _, priors = encode(model, text)
    encoded_img, _ = encode(model, text, topic_priors=priors)
    assert encoded_img.shape == model.dataset.mask_img.shape
