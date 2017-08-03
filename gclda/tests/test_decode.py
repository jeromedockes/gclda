# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Tests for GC-LDA decode module.
"""
from os.path import join

from gclda.model import Model
from gclda.decode import Decoder
from gclda.tests.utils import get_test_data_path


def test_init():
    """Smoke test for Decoder class.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    model = Model.load(model_file)
    decoder = Decoder(model)
    assert isinstance(decoder, Decoder)


def test_decode_roi():
    """Acceptance test of ROI-based decoding.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    roi_file = join(get_test_data_path(), 'roi.nii.gz')
    model = Model.load(model_file)
    decoder = Decoder(model)
    decoded_df = decoder.decode_roi(roi_file)
    assert decoded_df.shape[0] == model.n_word_labels


def test_decode_continuous():
    """Acceptance test of continuous image-based decoding.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    continuous_file = join(get_test_data_path(), 'continuous.nii.gz')
    model = Model.load(model_file)
    decoder = Decoder(model)
    decoded_df = decoder.decode_continuous(continuous_file)
    assert decoded_df.shape[0] == model.n_word_labels


def test_encode():
    """Acceptance test of test-to-image encoding.
    """
    model_file = join(get_test_data_path(), 'gclda_model.pkl')
    text = 'painful stimulation during a language task'
    model = Model.load(model_file)
    decoder = Decoder(model)
    encoded_matrix = decoder.encode(text)
    assert encoded_matrix.shape == model.dataset.masker.volume.shape
