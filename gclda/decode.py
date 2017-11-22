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
    1.  Compute p_topic_g_voxel.
            - From Model
    2.  Compute topic weight vector (tau_t).
            - topic_weights = np.sum(p_topic_g_voxel, axis=1) (across voxels)
    3.  Multiply tau_t by topic-by-word matrix (p_word_g_topic).
    4.  The resulting vector (tau_t*p_word_g_topic) should be word weights
        for your selected studies.
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
    1.  Compute p_topic_g_voxel.
    2.  Compute topic weight vector (tau_t) by multiplying p_topic_g_voxel
        by input image.
    3.  Multiply tau_t by topic-by-word matrix (p_word_g_topic).
    4.  The resulting vector (tau_t*p_word_g_topic) should be word weights
        for your map, but the values are scaled based on the input image, so
        they won't necessarily mean much.

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
    r"""
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
    The matrix p_topic_g_word is computed from the input text

    Terminology:

    .. math ::
        P_{tw}: Matrix of stuff

    Derive topic weights:

    .. math ::
        P_{tw} &= p(t = i|w = j)

               &= \frac{p(w = j|t = i) * p(t = i)}{p(w)}

               &= \frac{\Phi_{ji}}{\sum_{i=1}^{T} \Phi_{ji}}

        \tau_{t} = \sum_{w} P_{tw}

    Derive voxel weights:

    .. math ::
        values = \tau_{t} \cdot A

    Summary:

    1.  Compute p_topic_g_word.
            - p_topic_g_word = p_word_g_topic * p_topic / p_word
            - p_topic is uniform (1/n topics)
    2.  Compute topic weight vector (tau_t).
            - tau_t = np.sum(p_topic_g_word, axis=1) (across words)
    3.  Multiply tau_t by topic-by-voxel matrix of smoothed p_voxel_g_topic
        (A; not sure where it is, but I don't think it's the same as A in
        model.py).
    4.  The resulting map (tau_t*A) is the encoded image. Values are *not*
        probabilities.
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
