# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Class and functions for functional decoding.
"""


class Decoder(object):
    """
    Class object for a gcLDA decoder
    """
    def __init__(self, model):
        """
        Class object for a gcLDA decoder
        """
        self.model = model

    def decode_discrete(self, selected_ids, topic_priors=None):
        """
        Perform image-to-text decoding for discrete inputs (e.g., regions of
        interest, significant clusters, or sets of studies within the dataset
        selected via some other method).

        1.  Compute p_topic_g_voxel.
                - I think you need p_voxel_g_topic for this, then you do:
                - p_topic_g_voxel = p_voxel_g_topic * p_topic / p_voxel
                - What is p_voxel here?
        2.  Compute topic weight vector (tau_t).
                - tau_t = np.sum(p_topic_g_voxel, axis=1) (across voxels)
        3.  Multiply tau_t by topic-by-word matrix (p_word_g_topic).
        4.  The resulting vector (tau_t*p_word_g_topic) should be word weights
            for your selected studies.
        """
        pass

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
        pass

    def encode(self, text, topic_priors=None):
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
        pass
