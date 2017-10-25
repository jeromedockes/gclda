# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Class and functions for functional decoding.
"""
from __future__ import print_function, division

from builtins import object
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.masking import apply_mask, unmask
from sklearn.feature_extraction.text import CountVectorizer

from .due import due, Doi


@due.dcite(Doi('10.1371/journal.pcbi.1005649'),
           description='Describes decoding methods using GC-LDA.')
class Decoder(object):
    """
    Class object for a gcLDA decoder
    """
    def __init__(self, model):
        """
        Class object for a gcLDA decoder
        """
        self.model = model
        self.dataset = model.dataset

    def decode_roi(self, roi, topic_priors=None):
        """
        Perform image-to-text decoding for discrete image inputs (e.g., regions
        of interest, significant clusters).

        1.  Compute p_topic_g_voxel.
                - I think you need p_voxel_g_topic for this, then you do:
                - p_topic_g_voxel = p_voxel_g_topic * p_topic / p_voxel
                - What is p_voxel here?
        2.  Compute topic weight vector (tau_t).
                - topic_weights = np.sum(p_topic_g_voxel, axis=1) (across voxels)
        3.  Multiply tau_t by topic-by-word matrix (p_word_g_topic).
        4.  The resulting vector (tau_t*p_word_g_topic) should be word weights
            for your selected studies.
        """
        if type(roi) == str:
            roi = nib.load(roi)

        if not np.array_equal(roi.affine, self.model.dataset.mask_img.affine):
            str1 = np.array2string(roi.affine)
            str2 = np.array2string(self.model.dataset.mask_img.affine)
            raise ValueError('Input roi must have same affine as mask img:'
                             '\n{0}\n{1}'.format(str1, str2))

        # Load ROI file and get ROI voxels overlapping with brain mask
        roi_arr = roi.get_data() & self.model.dataset.mask_img.get_data()
        roi_voxels = np.where(roi_arr > 0)[0]

        p_topic_g_voxel, _ = self.model.get_spatial_probs()
        p_topic_g_roi = p_topic_g_voxel[roi_voxels, :]  # p(T|V) for voxels in ROI only
        topic_weights = np.sum(p_topic_g_roi, axis=0)  # Sum across words
        if topic_priors is not None:
            topic_weights *= topic_priors
        topic_weights /= np.sum(topic_weights)  # tau_t

        # Multiply topic_weights by topic-by-word matrix (p_word_g_topic).
        n_word_tokens_per_topic = np.sum(self.model.n_word_tokens_word_by_topic, axis=0)
        p_word_g_topic = self.model.n_word_tokens_word_by_topic / n_word_tokens_per_topic[None, :]
        p_word_g_topic = np.nan_to_num(p_word_g_topic, 0)
        word_weights = np.dot(p_word_g_topic, topic_weights)

        decoded_df = pd.DataFrame(index=self.model.dataset.word_labels, columns=['Weight'],
                                  data=word_weights)
        decoded_df.index.name = 'Term'
        return decoded_df, topic_weights

    def decode_continuous(self, image, topic_priors=None):
        """
        Perform image-to-text decoding for continuous inputs (e.g.,
        unthresholded statistical maps).

        1.  Compute p_topic_g_voxel.
        2.  Compute topic weight vector (tau_t) by multiplying p_topic_g_voxel
            by input image.
        3.  Multiply tau_t by topic-by-word matrix (p_word_g_topic).
        4.  The resulting vector (tau_t*p_word_g_topic) should be word weights
            for your map, but the values are scaled based on the input image, so
            they won't necessarily mean much.
        """
        # Load image file and get voxel values
        input_values = apply_mask(image, self.model.dataset.mask_img)
        p_topic_g_voxel, _ = self.model.get_spatial_probs()
        topic_weights = np.abs(np.squeeze(np.dot(p_topic_g_voxel.T, input_values[:, None])))
        if topic_priors is not None:
            topic_weights *= topic_priors
        topic_weights /= np.sum(topic_weights)  # tau_t

        # Multiply topic_weights by topic-by-word matrix (p_word_g_topic).
        n_word_tokens_per_topic = np.sum(self.model.n_word_tokens_word_by_topic, axis=0)
        p_word_g_topic = self.model.n_word_tokens_word_by_topic / n_word_tokens_per_topic[None, :]
        p_word_g_topic = np.nan_to_num(p_word_g_topic, 0)
        word_weights = np.dot(p_word_g_topic, topic_weights)

        decoded_df = pd.DataFrame(index=self.model.dataset.word_labels, columns=['Weight'],
                                  data=word_weights)
        decoded_df.index.name = 'Term'
        return decoded_df, topic_weights

    def encode(self, text, out_file=None, topic_priors=None):
        """
        Perform text-to-image encoding.

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
        vocabulary = [term.replace('_', ' ') for term in self.model.dataset.word_labels]
        max_len = max([len(term.split(' ')) for term in vocabulary])
        vectorizer = CountVectorizer(vocabulary=self.model.dataset.word_labels,
                                     ngram_range=(1, max_len))
        word_counts = np.squeeze(vectorizer.fit_transform([text]).toarray())
        keep_idx = np.where(word_counts > 0)[0]
        text_counts = word_counts[keep_idx]

        n_topics_per_word_token = np.sum(self.model.n_word_tokens_word_by_topic, axis=1)
        p_topic_g_word = self.model.n_word_tokens_word_by_topic / n_topics_per_word_token[:, None]
        p_topic_g_word = np.nan_to_num(p_topic_g_word, 0)
        p_topic_g_text = p_topic_g_word[keep_idx]  # p(T|W) for words in text only
        prod = p_topic_g_text * text_counts[:, None]  # Multiply p(T|W) by words in text
        topic_weights = np.sum(prod, axis=0)  # Sum across words
        if topic_priors is not None:
            topic_weights *= topic_priors
        topic_weights /= np.sum(topic_weights)  # tau_t

        _, p_voxel_g_topic = self.model.get_spatial_probs()
        voxel_weights = np.dot(p_voxel_g_topic, topic_weights)
        voxel_weights_matrix = unmask(voxel_weights, self.model.dataset.mask_img)

        img = nib.Nifti1Image(voxel_weights_matrix, self.model.dataset.mask_img.affine)
        if out_file is not None:
            img.to_filename(out_file)
        return img, topic_weights
