# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Functions for functional decoding/reverse inference using a GCLDA model.
"""
from __future__ import print_function, division

from builtins import object
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.masking import apply_mask, unmask
from sklearn.feature_extraction.text import CountVectorizer

from .utils import weight_priors
from .due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Describes decoding methods using GC-LDA.')
def decode_roi(model, roi, topic_priors=None, prior_weight=1.):
    """
    Perform image-to-text decoding for discrete image inputs (e.g., regions
    of interest, significant clusters).

    Parameters
    ----------
    model : :obj:`gclda.model.Model`
        Model object needed for decoding.
    roi : :obj:`nibabel.Nifti1Image` or :obj:`str`
        Binary image to decode into text. If string, path to a file with
        the binary image.
    topic_priors : :obj:`numpy.ndarray` of :obj:`float`, optional
        A 1d array of size (n_topics) with values for topic weighting.
        If None, no weighting is done. Default is None.
    prior_weight : :obj:`float`, optional
        The weight by which the prior will affect the decoding.
        Default is 1.

    Returns
    -------
    decoded_df : :obj:`pandas.DataFrame`
        A DataFrame with the word-tokens and their associated weights.
    topic_weights : :obj:`numpy.ndarray` of :obj:`float`
        The weights of the topics used in decoding.

    Notes
    -----
    ======================    ==============================================================
    Notation                  Meaning
    ======================    ==============================================================
    :math:`v`                 Voxel
    :math:`t`                 Topic
    :math:`w`                 Word type
    :math:`r`                 Region of interest (ROI)
    :math:`p(v|t)`            Probability of topic given voxel (``p_topic_g_voxel``)
    :math:`p(t|r)`            Probability of topic given ROI (``topic_weights``)
    :math:`p(w|t)`            Probability of word type given topic (``p_word_g_topic``)
    ======================    ==============================================================

    1.  Compute
        :math:`p(v|t)`.
            - From :obj:`gclda.model.Model.get_spatial_probs()`
    2.  Compute topic weight vector (:math:`\\tau_{t}`) by adding across voxels
        within ROI.
            - :math:`\\tau_{t} = \sum_{i} {p(t|v_{i})}`
    3.  Multiply :math:`\\tau_{t}` by
        :math:`p(w|t)`.
            - :math:`p(w|r) \propto \\tau_{t} \cdot p(w|t)`
    4.  The resulting vector (``word_weights``) reflects arbitrarily scaled
        term weights for the ROI.
    """
    if isinstance(roi, str):
        roi = nib.load(roi)
    elif not isinstance(roi, nib.Nifti1Image):
        raise IOError('Input roi must be either a nifti image '
                      '(nibabel.Nifti1Image) or a path to one.')

    dset_aff = model.dataset.mask_img.affine
    if not np.array_equal(roi.affine, dset_aff):
        raise ValueError('Input roi must have same affine as mask img:'
                         '\n{0}\n{1}'.format(np.array2string(roi.affine),
                                             np.array2string(dset_aff)))

    # Load ROI file and get ROI voxels overlapping with brain mask
    mask_vec = model.dataset.mask_img.get_data().ravel().astype(bool)
    roi_vec = roi.get_data().astype(bool).ravel()
    roi_vec = roi_vec[mask_vec]
    roi_idx = np.where(roi_vec)[0]
    p_topic_g_voxel, _ = model.get_spatial_probs()
    p_topic_g_roi = p_topic_g_voxel[roi_idx, :]  # p(T|V) for voxels in ROI only
    topic_weights = np.sum(p_topic_g_roi, axis=0)  # Sum across words
    if topic_priors is not None:
        weighted_priors = weight_priors(topic_priors, prior_weight)
        topic_weights *= weighted_priors

    # Multiply topic_weights by topic-by-word matrix (p_word_g_topic).
    n_word_tokens_per_topic = np.sum(model.n_word_tokens_word_by_topic, axis=0)
    p_word_g_topic = model.n_word_tokens_word_by_topic / n_word_tokens_per_topic[None, :]
    p_word_g_topic = np.nan_to_num(p_word_g_topic, 0)
    word_weights = np.dot(p_word_g_topic, topic_weights)

    decoded_df = pd.DataFrame(index=model.dataset.word_labels,
                              columns=['Weight'], data=word_weights)
    decoded_df.index.name = 'Term'
    return decoded_df, topic_weights


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Describes decoding methods using GC-LDA.')
def decode_continuous(model, image, topic_priors=None, prior_weight=1.):
    """
    Perform image-to-text decoding for continuous inputs (e.g.,
    unthresholded statistical maps).

    Parameters
    ----------
    model : :obj:`gclda.model.Model`
        Model object needed for decoding.
    image : :obj:`nibabel.Nifti1Image` or :obj:`str`
        Whole-brain image to decode into text. Must be in same space as
        model and dataset. Model's template available in
        `model.dataset.mask_img`.
    topic_priors : :obj:`numpy.ndarray` of :obj:`float`, optional
        A 1d array of size (n_topics) with values for topic weighting.
        If None, no weighting is done. Default is None.
    prior_weight : :obj:`float`, optional
        The weight by which the prior will affect the decoding.
        Default is 1.

    Returns
    -------
    decoded_df : :obj:`pandas.DataFrame`
        A DataFrame with the word-tokens and their associated weights.
    topic_weights : :obj:`numpy.ndarray` of :obj:`float`
        The weights of the topics used in decoding.

    Notes
    -----
    ======================    ==============================================================
    Notation                  Meaning
    ======================    ==============================================================
    :math:`v`                 Voxel
    :math:`t`                 Topic
    :math:`w`                 Word type
    :math:`i`                 Input image
    :math:`p(v|t)`            Probability of topic given voxel (``p_topic_g_voxel``)
    :math:`p(t|i)`            Topic weight vector (``topic_weights``)
    :math:`p(w|t)`            Probability of word type given topic (``p_word_g_topic``)
    :math:`\omega`            1d array from input image (``input_values``)
    ======================    ==============================================================

    1.  Compute :math:`p(t|v)`
        (``p_topic_g_voxel``).
            - From :obj:`gclda.model.Model.get_spatial_probs()`
    2.  Squeeze input image to 1d array :math:`\omega` (``input_values``).
    3.  Compute topic weight vector (:math:`\\tau_{t}`) by multiplying
        :math:`p(t|v)` by input image.
            - :math:`\\tau_{t} = p(t|v) \cdot \omega`
    4.  Multiply :math:`\\tau_{t}` by
        :math:`p(w|t)`.
            - :math:`p(w|i) \propto \\tau_{t} \cdot p(w|t)`
    5.  The resulting vector (``word_weights``) reflects arbitrarily scaled
        term weights for the input image.

    """
    if isinstance(image, str):
        image = nib.load(image)
    elif not isinstance(image, nib.Nifti1Image):
        raise IOError('Input image must be either a nifti image '
                      '(nibabel.Nifti1Image) or a path to one.')

    # Load image file and get voxel values
    input_values = apply_mask(image, model.dataset.mask_img)
    p_topic_g_voxel, _ = model.get_spatial_probs()
    topic_weights = np.squeeze(np.dot(p_topic_g_voxel.T, input_values[:, None]))
    if topic_priors is not None:
        weighted_priors = weight_priors(topic_priors, prior_weight)
        topic_weights *= weighted_priors

    # Multiply topic_weights by topic-by-word matrix (p_word_g_topic).
    n_word_tokens_per_topic = np.sum(model.n_word_tokens_word_by_topic, axis=0)
    p_word_g_topic = model.n_word_tokens_word_by_topic / n_word_tokens_per_topic[None, :]
    p_word_g_topic = np.nan_to_num(p_word_g_topic, 0)
    word_weights = np.dot(p_word_g_topic, topic_weights)

    decoded_df = pd.DataFrame(index=model.dataset.word_labels, columns=['Weight'],
                              data=word_weights)
    decoded_df.index.name = 'Term'
    return decoded_df, topic_weights


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Describes decoding methods using GC-LDA.')
def encode(model, text, out_file=None, topic_priors=None, prior_weight=1.):
    """
    Perform text-to-image encoding.

    Parameters
    ----------
    model : :obj:`gclda.model.Model`
        Model object needed for decoding.
    text : :obj:`str` or :obj:`list`
        Text to encode into an image.
    out_file : :obj:`str`, optional
        If not None, writes the encoded image to a file.
    topic_priors : :obj:`numpy.ndarray` of :obj:`float`, optional
        A 1d array of size (n_topics) with values for topic weighting.
        If None, no weighting is done. Default is None.
    prior_weight : :obj:`float`, optional
        The weight by which the prior will affect the encoding.
        Default is 1.

    Returns
    -------
    img : :obj:`nibabel.Nifti1Image`
        The encoded image.
    topic_weights : :obj:`numpy.ndarray` of :obj:`float`
        The weights of the topics used in encoding.

    Notes
    -----
    ======================    ==============================================================
    Notation                  Meaning
    ======================    ==============================================================
    :math:`v`                 Voxel
    :math:`t`                 Topic
    :math:`w`                 Word type
    :math:`h`                 Input text
    :math:`p(v|t)`            Probability of topic given voxel (``p_topic_g_voxel``)
    :math:`\\tau_{t}`          Topic weight vector (``topic_weights``)
    :math:`p(w|t)`            Probability of word type given topic (``p_word_g_topic``)
    :math:`\omega`            1d array from input image (``input_values``)
    ======================    ==============================================================

    1.  Compute :math:`p(v|t)`
        (``p_voxel_g_topic``).
            - From :obj:`gclda.model.Model.get_spatial_probs()`
    2.  Compute :math:`p(t|w)`
        (``p_topic_g_word``).
    3.  Vectorize input text according to model vocabulary.
    4.  Reduce :math:`p(t|w)` to only include word types in input text.
    5.  Compute :math:`p(t|h)` (``p_topic_g_text``) by multiplying :math:`p(t|w)`
        by word counts for input text.
    6.  Sum topic weights (:math:`\\tau_{t}`) across
        words.
            - :math:`\\tau_{t} = \sum_{i}{p(t|h_{i})}`
    7.  Compute voxel
        weights.
            - :math:`p(v|h) \propto p(v|t) \cdot \\tau_{t}`
    8.  The resulting array (``voxel_weights``) reflects arbitrarily scaled
        voxel weights for the input text.
    9.  Unmask and reshape ``voxel_weights`` into brain image.
    """
    if isinstance(text, list):
        text = ' '.join(text)

    # Assume that words in word_labels are underscore-separated.
    # Convert to space-separation for vectorization of input string.
    vocabulary = [term.replace('_', ' ') for term in model.dataset.word_labels]
    max_len = max([len(term.split(' ')) for term in vocabulary])
    vectorizer = CountVectorizer(vocabulary=model.dataset.word_labels,
                                 ngram_range=(1, max_len))
    word_counts = np.squeeze(vectorizer.fit_transform([text]).toarray())
    keep_idx = np.where(word_counts > 0)[0]
    text_counts = word_counts[keep_idx]

    n_topics_per_word_token = np.sum(model.n_word_tokens_word_by_topic, axis=1)
    p_topic_g_word = model.n_word_tokens_word_by_topic / n_topics_per_word_token[:, None]
    p_topic_g_word = np.nan_to_num(p_topic_g_word, 0)
    p_topic_g_text = p_topic_g_word[keep_idx]  # p(T|W) for words in text only
    prod = p_topic_g_text * text_counts[:, None]  # Multiply p(T|W) by words in text
    topic_weights = np.sum(prod, axis=0)  # Sum across words
    if topic_priors is not None:
        weighted_priors = weight_priors(topic_priors, prior_weight)
        topic_weights *= weighted_priors

    _, p_voxel_g_topic = model.get_spatial_probs()
    voxel_weights = np.dot(p_voxel_g_topic, topic_weights)
    img = unmask(voxel_weights, model.dataset.mask_img)

    if out_file is not None:
        img.to_filename(out_file)
    return img, topic_weights
