# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Class and functions for model-related stuff.
"""
from __future__ import print_function, division
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
from os import mkdir
from os.path import join, isdir
import pickle

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class Model(object):
    """
    Class object for a gcLDA model.

    Creates a gcLDA model using a dataset object and hyperparameter arguments.

    Parameters
    ----------
    dataset : :obj:`gclda.dataset.Dataset`
        Dataset object containing data needed for model.

    n_topics : int, optional
        Number of topics to generate in model. The default is 100.

    n_regions : int, optional
        Number of subregions per topic (>=1). The default is 2.

    alpha : float, optional
        Prior count on topics for each document. The default is 0.1.

    beta : float, optional
        Prior count on word-types for each topic. The default is 0.01.

    gamma : float, optional
        Prior count added to y-counts when sampling z assignments. The
        default is 0.01.

    delta : float, optional
        Prior count on subregions for each topic. The default is 1.0.

    dobs : int, optional
        Spatial region 'default observations' (# observations weighting
        Sigma estimates in direction of default 'roi_size' value). The
        default is 25.

    roi_size : float, optional
        Default spatial 'region of interest' size (default value of
        diagonals in covariance matrix for spatial distribution, which the
        distributions are biased towards). The default is 50.0.

    symmetric : bool, optional
        Whether or not to use symmetry constraint on subregions. Symmetry
        requires n_regions = 2. The default is False.

    seed_init : int, optional
        Initial value of random seed. The default is 1.

    Attributes
    ----------
    model_name : str
        Identifier (based on parameter values) for the model.

    wtoken_topic_idx : :obj:`numpy.ndarray` of :obj:`numpy.int64`
        A number-of-words-by-1 vector of word->topic assignments.

    peak_topic_idx : :obj:`numpy.ndarray` of :obj:`numpy.int64`
        A number-of-peaks-by-1 vector of peak->topic assignments.

    peak_region_idx : :obj:`numpy.ndarray` of :obj:`numpy.int64`
        A number-of-peaks-by-1 vector of peak->region assignments.

    n_peak_tokens_doc_by_topic : :obj:`numpy.ndarray` of :obj:`numpy.int64`
        An n-documents-by-n-topics array. Each cell is the number of
        peak-tokens for a given document assigned to a given topic.

    n_peak_tokens_region_by_topic : :obj:`numpy.ndarray` of :obj:`numpy.int64`
        An n-regions-by-n-topics array. Each cell is the number of
        peak-tokens for a given region assigned to a given topic.

    n_word_tokens_word_by_topic : :obj:`numpy.ndarray` of :obj:`numpy.int64`
        An n-words-by-n-topics array. Each cell is the number of
        word-tokens for a given word assigned to a given topic.

    n_word_tokens_doc_by_topic : :obj:`numpy.ndarray` of :obj:`numpy.int64`
        An n-documents-by-n-topics array. Each cell is the number of
        word-tokens for a given document assigned to a given topic.

    total_n_word_tokens_by_topic : :obj:`numpy.ndarray` of :obj:`numpy.int64`
        A 1-by-number-of-words vector. Total number of word-tokens assigned
        to each topic (across all documents).

    """
    def __init__(self, dataset, n_topics=100, n_regions=2, symmetric=False,
                 alpha=.1, beta=.01, gamma=.01, delta=1.0,
                 dobs=25, roi_size=50.0, seed_init=1):

        print('Constructing GC-LDA Model')

        # --- Checking to make sure parameters are valid
        if (symmetric is True) and (n_regions != 2):
            # symmetric model only valid if R = 2
            raise ValueError('Cannot run a symmetric model unless #Subregions (n_regions) == 2 !')

        # --- Assign dataset object to model
        self.dataset = dataset

        # --- Initialize sampling parameters
        self.iter = 0  # Tracks the global sampling iteration of the model
        self.seed_init = seed_init  # Random seed for initializing model
        self.seed = 0  # Tracks current random seed to use (gets incremented
                       # after initialization and each sampling update)

        # --- Set up gcLDA model hyper-parameters from input
        # Pseudo-count hyperparams need to be floats so that when sampling
        # distributions are computed the count matrices/vectors are converted
        # to floats
        self.n_topics = n_topics  # Number of topics (T)
        self.n_regions = n_regions  # Number of subregions (R)
        self.alpha = float(alpha)  # Prior count on topics for each doc (\alpha)
        self.beta = float(beta)  # Prior count on word-types for each topic (\beta)
        self.gamma = float(gamma)  # Prior count added to y-counts when sampling
                                   # z assignments (\gamma)
        self.delta = float(delta)  # Prior count on subregions for each topic (\delta)
        self.roi_size = float(roi_size)  # Default ROI (default covariance spatial region
                                         # we regularize towards) (not in paper)
        self.dobs = int(dobs)  # Sample constant (# observations weighting
                               # sigma in direction of default covariance)
                               # (not in paper)
        self.symmetric = symmetric  # Use constrained symmetry on subregions?
                                    # (only for n_regions = 2)

        # Get model name
        self.model_name = self._get_model_name()

        # --- Get dimensionalities of vectors/matrices from dataset object
        self.n_peak_tokens = len(self.dataset.ptoken_doc_idx)  # Number of peak-tokens
        self.n_word_labels = len(self.dataset.word_labels)  # Number of word-types
        self.n_docs = len(self.dataset.pmids)  # Number of documents
        self.n_peak_dims = self.dataset.peak_vals.shape[1]  # Dimensionality of peak_locs data

        #  --- Preallocate vectors of assignment indices
        self.wtoken_topic_idx = np.zeros(len(self.dataset.wtoken_word_idx),
                                         dtype=int)  # word->topic assignments
        self.peak_topic_idx = np.zeros(self.n_peak_tokens, dtype=int)  # peak->topic assignments
        self.peak_region_idx = np.zeros(self.n_peak_tokens, dtype=int)  # peak->region assignments

        #  --- Preallocate count matrices
        # Peaks: D x T: Number of peak-tokens assigned to each topic per document
        self.n_peak_tokens_doc_by_topic = np.zeros(shape=(self.n_docs,
                                                          self.n_topics), dtype=int)

        # Peaks: R x T: Number of peak-tokens assigned to each subregion per topic
        self.n_peak_tokens_region_by_topic = np.zeros(shape=(self.n_regions,
                                                             self.n_topics), dtype=int)

        # Words: W x T: Number of word-tokens assigned to each topic per word-type
        self.n_word_tokens_word_by_topic = np.zeros(shape=(self.n_word_labels,
                                                           self.n_topics), dtype=int)

        # Words: D x T: Number of word-tokens assigned to each topic per document
        self.n_word_tokens_doc_by_topic = np.zeros(shape=(self.n_docs,
                                                          self.n_topics), dtype=int)

        # Words: 1 x T: Total number of word-tokens assigned to each topic (across all docs)
        self.total_n_word_tokens_by_topic = np.zeros(shape=(1, self.n_topics), dtype=int)

        # --- Preallocate Gaussians for all subregions
        # Regions_Mu & Regions_Sigma: Gaussian mean and covariance for all
        # subregions of all topics
        # Formed using lists (over topics) of lists (over subregions) of numpy
        # arrays
        #   regions_mu = (n_topics, n_regions, 1, n_peak_dims)
        #   regions_sigma = (n_topics, n_regions, n_peak_dims, n_peak_dims)
        self.regions_mu = []
        self.regions_sigma = []
        for i_topic in range(self.n_topics):
            topic_mu = []
            topic_sigma = []
            for j_region in range(self.n_regions):
                topic_mu.append(np.zeros(shape=(1, self.n_peak_dims)))
                topic_sigma.append(np.zeros(shape=(self.n_peak_dims, self.n_peak_dims)))
            self.regions_mu.append(topic_mu)  # (\mu^{(t)}_r)
            self.regions_sigma.append(topic_sigma)  # (\sigma^{(t)}_r)

        # Initialize lists for tracking log-likelihood of data over sampling iterations
        self.loglikely_iter = []  # Tracks iteration we compute each
                                  # loglikelihood at
        self.loglikely_x = []  # Tracks log-likelihood of peak tokens
        self.loglikely_w = []  # Tracks log-likelihood of word tokens
        self.loglikely_tot = []  # Tracks log-likelihood of peak + word tokens

    def initialize(self):
        """
        Random Initialization: Initial z, y, r assignments.
        Get Initial Spatial Estimates
        """
        print('Initializing GC-LDA Model')

        # --- Seed random number generator
        np.random.seed(self.seed_init)  # pylint: disable=no-member

        # --- Randomly initialize peak->topic assignments (y) ~ unif(1...n_topics)
        self.peak_topic_idx[:] = np.random.randint(self.n_topics,  # pylint: disable=no-member
                                                   size=(self.n_peak_tokens))

        # --- Initialize peak->subregion assignments (r)
        #   if asymmetric model, randomly sample r ~ unif(1...n_regions)
        #   if symmetric model use deterministic assignment :
        #       if peak_val[0] > 0, r = 1, else r = 0
        if not self.symmetric:
            self.peak_region_idx[:] = np.random.randint(self.n_regions,  # pylint: disable=no-member
                                                        size=(self.n_peak_tokens))
        else:
            self.peak_region_idx[:] = (self.dataset.peak_vals[:, 0] > 0).astype(int)

        # Update model vectors and count matrices to reflect y and r assignments
        for i_peak_token in range(self.n_peak_tokens):
            # document -idx (d)
            doc = self.dataset.ptoken_doc_idx[i_peak_token]
            topic = self.peak_topic_idx[i_peak_token]  # peak-token -> topic assignment (y_i)
            region = self.peak_region_idx[i_peak_token]  # peak-token -> subregion assignment (c_i)
            self.n_peak_tokens_doc_by_topic[doc, topic] += 1  # Increment document-by-topic
                                                              # counts
            self.n_peak_tokens_region_by_topic[region, topic] += 1  # Increment region-by-topic

        # --- Randomly Initialize Word->Topic Assignments (z) for each word
        # token w_i: sample z_i proportional to p(topic|doc_i)
        for i_word_token in range(len(self.dataset.wtoken_word_idx)):
            # w_i word-type
            word = self.dataset.wtoken_word_idx[i_word_token]

            # w_i doc-index
            doc = self.dataset.wtoken_doc_idx[i_word_token]

            # Estimate p(t|d) for current doc
            p_topic_g_doc = self.n_peak_tokens_doc_by_topic[doc] + self.gamma

            # Sample a topic from p(t|d) for the z-assignment
            probs = np.cumsum(p_topic_g_doc)  # Compute a cdf of the sampling
                                              # distribution for z
            # Which elements of cdf are less than random sample?
            sample_locs = probs < np.random.rand() * probs[-1]  # pylint: disable=no-member
            sample_locs = np.where(sample_locs)  # How many elements of cdf are
                                                 # less than sample
            topic = len(sample_locs[0])  # z = # elements of cdf less than
                                         # rand-sample

            # Update model assignment vectors and count-matrices to reflect z
            self.wtoken_topic_idx[i_word_token] = topic  # Word-token -> topic assignment (z_i)
            self.n_word_tokens_word_by_topic[word, topic] += 1
            self.total_n_word_tokens_by_topic[0, topic] += 1
            self.n_word_tokens_doc_by_topic[doc, topic] += 1

        # --- Get Initial Spatial Parameter Estimates
        self._update_regions()

        # --- Get Log-Likelihood of data for Initialized model and save to
        # variables tracking loglikely
        self._compute_log_likelihood()

    # -------------------------------------------------------------------------------
    # <<<<< Model Parameter Update Methods >>>> Update z, Update y/r, Update regions  |
    # -------------------------------------------------------------------------------

    def run_complete_iteration(self, loglikely_freq=1, verbose=2):
        """
        Run a complete update cycle (sample z, sample y&r, update regions).


        Parameters
        ----------
        loglikely_freq : int, optional
            The frequency with which log-likelihood is updated. Default value
            is 1 (log-likelihood is updated every iteration).

        verbose : {0, 1, 2}, optional
            Determines how much info is printed to console. 0 = none,
            1 = a little, 2 = a lot. Default value is 2.
        """
        self.iter += 1  # Update total iteration count

        if verbose == 2:
            print('Iter {0:04d}: Sampling z'.format(self.iter))
        self.seed += 1
        self._update_word_topic_assignments(self.seed)  # Update z-assignments

        if verbose == 2:
            print('Iter {0:04d}: Sampling y|r'.format(self.iter))
        self.seed += 1
        self._update_peak_assignments(self.seed)  # Update y-assignments

        if verbose == 2:
            print('Iter {0:04d}: Updating spatial params'.format(self.iter))
        self._update_regions()  # Update gaussian estimates for all subregions

        # Only update loglikelihood every 'loglikely_freq' iterations
        # (Computing log-likelihood isn't necessary and slows things down a bit)
        if self.iter % loglikely_freq == 0:
            if verbose == 2:
                print('Iter {0:04d}: Computing log-likelihood'.format(self.iter))
            self._compute_log_likelihood()  # Compute log-likelihood of
                                                        # model in current state
            if verbose > 0:
                print('Iter {0:04d} Log-likely: x = {1:10.1f}, w = {2:10.1f}, '
                      'tot = {3:10.1f}'.format(self.iter, self.loglikely_x[-1],
                                               self.loglikely_w[-1],
                                               self.loglikely_tot[-1]))

    def _update_word_topic_assignments(self, randseed):
        """
        Update wtoken_topic_idx (z) indicator variables assigning words->topics.

        Parameters
        ----------
        randseed : int
            Random seed for this iteration.
        """
        # --- Seed random number generator
        np.random.seed(randseed)  # pylint: disable=no-member

        # Loop over all word tokens
        for i_word_token in range(len(self.dataset.wtoken_word_idx)):
            # Get indices for current token
            word = self.dataset.wtoken_word_idx[i_word_token]  # w_i word-type
            doc = self.dataset.wtoken_doc_idx[i_word_token]  # w_i doc-index
            topic = self.wtoken_topic_idx[i_word_token]  # current topic assignment for
                                                         # word token w_i

            # Decrement count-matrices to remove current wtoken_topic_idx
            self.n_word_tokens_word_by_topic[word, topic] -= 1
            self.total_n_word_tokens_by_topic[0, topic] -= 1
            self.n_word_tokens_doc_by_topic[doc, topic] -= 1

            # Get Sampling distribution:
            #    p(z_i|z,d,w) ~ p(w|t) * p(t|d)
            #                 ~ p_w_t * p_topic_g_doc
            p_word_g_topic = (self.n_word_tokens_word_by_topic[word, :] + self.beta) /\
                             (self.total_n_word_tokens_by_topic + \
                              self.beta * self.n_word_labels)
            p_topic_g_doc = self.n_peak_tokens_doc_by_topic[doc, :] + self.gamma
            probs = p_word_g_topic * p_topic_g_doc  # The unnormalized sampling distribution

            # Sample a z_i assignment for the current word-token from the sampling distribution
            probs = np.squeeze(probs) / np.sum(probs)  # Normalize the sampling
                                                       # distribution
            # Numpy returns a [1 x T] vector with a '1' in the index of sampled topic
            vec = np.random.multinomial(1, probs)  # pylint: disable=no-member
            vec_loc = np.where(vec)  # Transform the indicator vector into a
                                     # single z-index (stored in tuple)
            topic = vec_loc[0][0]  # Extract the sampled z value from the tuple

            # Update the indices and the count matrices using the sampled z assignment
            self.wtoken_topic_idx[i_word_token] = topic  # Update w_i topic-assignment
            self.n_word_tokens_word_by_topic[word, topic] += 1
            self.total_n_word_tokens_by_topic[0, topic] += 1
            self.n_word_tokens_doc_by_topic[doc, topic] += 1

    def _update_peak_assignments(self, randseed):
        """
        Update y / r indicator variables assigning peaks->topics/subregions.

        Parameters
        ----------
        randseed : int
            Random seed for this iteration.
        """
        # --- Seed random number generator
        np.random.seed(randseed)  # pylint: disable=no-member

        # Retrieve p(x|r,y) for all subregions
        peak_probs = self._get_peak_probs()

        # Iterate over all peaks x, and sample a new y and r assignment for each
        for i_peak_token in range(self.n_peak_tokens):
            doc = self.dataset.ptoken_doc_idx[i_peak_token]
            topic = self.peak_topic_idx[i_peak_token]
            region = self.peak_region_idx[i_peak_token]

            # Decrement count in Subregion x Topic count matrix
            self.n_peak_tokens_region_by_topic[region, topic] -= 1

            # Decrement count in Document x Topic count matrix
            self.n_peak_tokens_doc_by_topic[doc, topic] -= 1

            # Retrieve the probability of generating current x from all
            # subregions: [R x T] array of probs
            p_x_subregions = (peak_probs[i_peak_token, :, :]).transpose()

            # --- Compute the probabilities of all subregions given doc:
            # p(r|d) ~ p(r|t) * p(t|d) ---
            # Counts of subregions per topic + prior: p(r|t)
            p_region_g_topic = self.n_peak_tokens_region_by_topic + self.delta

            # Normalize the columns such that each topic's distribution over
            # subregions sums to 1
            p_region_g_topic = p_region_g_topic / np.sum(p_region_g_topic, axis=0)

            # Counts of topics per document + prior: p(t|d)
            p_topic_g_doc = self.n_peak_tokens_doc_by_topic[doc, :] + self.alpha

            # Compute p(subregion | document): p(r|d) ~ p(r|t) * p(t|d)
            # [R x T] array of probs
            p_region_g_doc = np.ones([self.n_regions, 1]) * p_topic_g_doc * p_region_g_topic

            # --- Compute the multinomial probability: p(z|y) ---
            # Need the current vector of all z and y assignments for current doc
            # The multinomial from which z is sampled is proportional to number
            # of y assigned to each topic, plus constant \gamma
            doc_y_counts = self.n_peak_tokens_doc_by_topic[doc, :] + self.gamma
            doc_z_counts = self.n_word_tokens_doc_by_topic[doc, :]
            p_z_y = np.zeros([1, self.n_topics])
            p_z_y[:] = self._compute_prop_multinomial_from_zy_vectors(doc_z_counts, doc_y_counts)

            ## Get the full sampling distribution:
            # [R x T] array containing the proportional probability of all y/r combinations
            probs_pdf = p_x_subregions * p_region_g_doc * np.dot(np.ones([self.n_regions, 1]),
                                                                 p_z_y)

            # Convert from a [R x T] matrix into a [R*T x 1] array we can sample from
            probs_pdf = probs_pdf.transpose().ravel()

            # Normalize the sampling distribution
            probs_pdf = probs_pdf / np.sum(probs_pdf)

            # Sample a single element (corresponding to a y_i and c_i assignment
            # for the peak token) from the sampling distribution
            # Returns a [1 x R*T] vector with a '1' in location that was sampled
            vec = np.random.multinomial(1, probs_pdf)  # pylint: disable=no-member
            vec_loc = np.where(vec)  # Converts the indicator vector into a linear
                                     # index value (stored in a tuple)
            sample_idx = vec_loc[0][0]  # Extract the linear index value from the tuple

            # Transform the linear index of the sampled element into the
            # subregion/topic (r/y) assignment indices
            # Subregion sampled (r)
            region = np.remainder(sample_idx, self.n_regions)  # pylint: disable=no-member
            topic = int(np.floor(sample_idx / self.n_regions))  # Topic sampled (y)

            # Update the indices and the count matrices using the sampled y/r assignments
            self.n_peak_tokens_region_by_topic[region, topic] += 1  # Increment count in
                                                                    # Subregion x Topic count
                                                                    # matrix
            self.n_peak_tokens_doc_by_topic[doc, topic] += 1  # Increment count in
                                                              # Document x Topic count matrix
            self.peak_topic_idx[i_peak_token] = topic  # Update y->topic assignment
            self.peak_region_idx[i_peak_token] = region  # Update y->subregion assignment

    def _update_regions(self):
        """
        Update spatial distribution parameters (Gaussians params for all
        subregions).

        Updates regions_mu and regions_sigma, indicating location and
        distribution of each subregion.
        """
        # Generate default ROI based on default_width
        default_roi = self.roi_size * np.eye(self.n_peak_dims)

        if not self.symmetric:
            # --- If model subregions not symmetric ---
            # For each region, compute a mean and a regularized covariance matrix
            for i_topic in range(self.n_topics):
                for j_region in range(self.n_regions):
                    # -- Get all peaks assigned to current topic & subregion --
                    idx = (self.peak_topic_idx == i_topic) & (self.peak_region_idx == j_region)
                    vals = self.dataset.peak_vals[idx]
                    n_obs = self.n_peak_tokens_region_by_topic[j_region, i_topic]

                    # -- Estimate Mean --
                    # If there are no observations, we set mean equal to zeros,
                    # otherwise take MLE
                    if n_obs == 0:
                        mu = np.zeros([self.n_peak_dims])
                    else:
                        mu = np.mean(vals, axis=0)

                    # -- Estimate Covariance --
                    # if there are 1 or fewer observations, we set sigma_hat
                    # equal to default ROI, otherwise take MLE
                    if n_obs <= 1:
                        c_hat = default_roi
                    else:
                        c_hat = np.cov(np.transpose(vals))

                    # Regularize the covariance, using the ratio of observations
                    # to dobs (default constant # observations)
                    d_c = n_obs / (n_obs + self.dobs)
                    sigma = d_c * c_hat + (1-d_c) * default_roi

                    # --  Store estimates in model object --
                    self.regions_mu[i_topic][j_region][:] = mu
                    self.regions_sigma[i_topic][j_region][:] = sigma
        else:
            # --- If model subregions are symmetric ---
            # With symmetric subregions, we jointly compute all estimates for
            # subregions 1 & 2, constraining the means to be symmetric w.r.t.
            # the origin along x-dimension
            for i_topic in range(self.n_topics):

                # -- Get all peaks assigned to current topic & subregion 1 --
                idx1 = (self.peak_topic_idx == i_topic) & (self.peak_region_idx == 0)
                vals1 = self.dataset.peak_vals[idx1]
                n_obs1 = self.n_peak_tokens_region_by_topic[0, i_topic]

                # -- Get all peaks assigned to current topic & subregion 2 --
                idx2 = (self.peak_topic_idx == i_topic) & (self.peak_region_idx == 1)
                vals2 = self.dataset.peak_vals[idx2]
                n_obs2 = self.n_peak_tokens_region_by_topic[1, i_topic]

                # -- Get all peaks assigned to current topic & either subregion --
                allvals = self.dataset.peak_vals[idx1 | idx2]

                # --------------------
                # -- Estimate Means --
                # --------------------

                # -- Estimate Independent Mean For Subregion 1 --
                # If there are no observations, we set mean equal to zeros, otherwise take MLE
                if n_obs1 == 0:
                    m = np.zeros([self.n_peak_dims])
                else:
                    m = np.mean(vals1, axis=0)

                # -- Estimate Independent Mean For Subregion 2 --
                # If there are no observations, we set mean equal to zeros, otherwise take MLE
                if n_obs2 == 0:
                    n = np.zeros([self.n_peak_dims])
                else:
                    n = np.mean(vals2, axis=0)

                # -- Estimate the weighted means of all dims, where for dim1 we
                # compute the mean w.r.t. absolute distance from the origin
                weighted_mean_dim1 = (-m[0]*n_obs1 + n[0]*n_obs2) / (n_obs1 + n_obs2)
                weighted_mean_otherdims = np.mean(allvals[:, 1:], axis=0)

                # Store weighted mean estimates
                mu1 = np.zeros([1, self.n_peak_dims])
                mu2 = np.zeros([1, self.n_peak_dims])
                mu1[0, 0] = -weighted_mean_dim1
                mu1[0, 1:] = weighted_mean_otherdims
                mu2[0, 0] = weighted_mean_dim1
                mu2[0, 1:] = weighted_mean_otherdims

                # --  Store estimates in model object --
                self.regions_mu[i_topic][0][:] = mu1
                self.regions_mu[i_topic][1][:] = mu2

                # --------------------------
                # -- Estimate Covariances --
                # --------------------------

                # Covariances are estimated independently
                # Cov for subregion 1
                if n_obs1 <= 1:
                    c_hat1 = default_roi
                else:
                    c_hat1 = np.cov(np.transpose(vals1))
                # Cov for subregion 2
                if n_obs2 <= 1:
                    c_hat2 = default_roi
                else:
                    c_hat2 = np.cov(np.transpose(vals2))

                # Regularize the covariances, using the ratio of observations to sample_constant
                d_c_1 = (n_obs1) / (n_obs1 + self.dobs)
                d_c_2 = (n_obs2) / (n_obs2 + self.dobs)
                sigma1 = d_c_1 * c_hat1 + (1-d_c_1) * default_roi
                sigma2 = d_c_2 * c_hat2 + (1-d_c_2) * default_roi

                # --  Store estimates in model object --
                self.regions_sigma[i_topic][0][:] = sigma1
                self.regions_sigma[i_topic][1][:] = sigma2

    # --------------------------------------------------------------------------------
    # <<<<< Utility Methods for GC-LDA >>>>> Log-Likelihood, Get Peak-Probs , mnpdf  |
    # --------------------------------------------------------------------------------

    def _compute_log_likelihood(self, update_vectors=True):
        """
        Compute Log-likelihood of a dataset object given current model.

        Computes the log-likelihood of data in any dataset object (either train
        or test) given the posterior predictive distributions over peaks and
        word-types for the model. Note that this is not computing the joint
        log-likelihood of model parameters and data.

        Parameters
        ----------
        update_vectors : bool, optional
            Whether to update model's log-likelihood vectors or not.

        Returns
        -------
        x_loglikely : float
            Total log-likelihood of all peak tokens.

        w_loglikely : float
            Total log-likelihood of all word tokens.

        tot_loglikely : float
            Total log-likelihood of peak + word tokens.

        References
        ----------
        [1] Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009).
        Distributed algorithms for topic models. Journal of Machine Learning
        Research, 10(Aug), 1801-1828.
        """
        # --- Pre-compute all probabilities from count matrices that are needed
        # for loglikelihood computations

        # Compute docprobs for y = ND x NT: p( y_i=t | d )
        doccounts = self.n_peak_tokens_doc_by_topic + self.alpha
        doccounts_sum = np.sum(doccounts, axis=1)
        docprobs_y = np.transpose(np.transpose(doccounts) / doccounts_sum)

        # Compute docprobs for z = ND x NT: p( z_i=t | y^(d) )
        doccounts = self.n_peak_tokens_doc_by_topic + self.gamma
        doccounts_sum = np.sum(doccounts, axis=1)
        docprobs_z = np.transpose(np.transpose(doccounts) / doccounts_sum)

        # Compute regionprobs = NR x NT: p( r | t )
        regioncounts = (self.n_peak_tokens_region_by_topic) + self.delta
        regioncounts_sum = np.sum(regioncounts, axis=0)
        regionprobs = regioncounts / regioncounts_sum

        # Compute wordprobs = NW x NT: p( w | t )
        wordcounts = self.n_word_tokens_word_by_topic + self.beta
        wordcounts_sum = np.sum(wordcounts, axis=0)
        wordprobs = wordcounts / wordcounts_sum

        # --- Get the matrix giving p(x_i|r,t) for all x:
        #    NY x NT x NR matrix of probabilities of all peaks given all
        #    topic/subregion spatial distributions
        peak_probs = self._get_peak_probs()

        # -----------------------------------------------------------------------------
        # --- Compute observed peaks (x) Loglikelihood:
        # p(x|model, doc) = p(topic|doc) * p(subregion|topic) * p(x|subregion)
        #                    = p_topic_g_doc * p_region_g_topic * p_x_r
        x_loglikely = 0  # Initialize variable tracking total loglikelihood of all x tokens

        # Go over all observed peaks and add p(x|model) to running total
        for i_peak_token in range(self.n_peak_tokens):
            doc = self.dataset.ptoken_doc_idx[i_peak_token] - 1  # convert didx from 1-idx to 0-idx
            p_x = 0  # Running total for p(x|d) across subregions:
                     # Compute p(x_i|d) for each subregion separately and then
                     # sum across the subregions
            for j_region in range(self.n_regions):
                # p(t|d) - p(topic|doc)
                p_topic_g_doc = docprobs_y[doc]

                # p(r|t) - p(subregion|topic)
                p_region_g_topic = regionprobs[j_region]

                # p(r|d) - p(subregion|document) = p(topic|doc)*p(subregion|topic)
                p_region_g_doc = p_topic_g_doc * p_region_g_topic

                # p(x|r) - p(x|subregion)
                p_x_r = peak_probs[i_peak_token, :, j_region]

                # p(x|subregion,doc) = sum_topics ( p(subregion|doc) * p(x|subregion) )
                p_x_rd = np.dot(p_region_g_doc, p_x_r)
                p_x += p_x_rd  # Add probability for current subregion to total
                               # probability for token across subregions
            # Add probability for current token to running total for all x tokens
            x_loglikely += np.log(p_x)  # pylint: disable=no-member

        # -----------------------------------------------------------------------------
        # --- Compute observed words (w) Loglikelihoods:
        # p(w|model, doc) = p(topic|doc) * p(word|topic)
        #                    = p_topic_g_doc * p_w_t
        w_loglikely = 0  # Initialize variable tracking total loglikelihood of all w tokens

        # Compute a matrix of posterior predictives over words:
        # = ND x NW p(w|d) = sum_t ( p(t|d) * p(w|t) )
        p_wtoken_g_doc = np.dot(docprobs_z, np.transpose(wordprobs))

        # Go over all observed word tokens and add p(w|model) to running total
        for i_word_token in range(len(self.dataset.wtoken_word_idx)):
            word_token = self.dataset.wtoken_word_idx[i_word_token] - 1  # convert wtoken_word_idx
                                                                    # from 1-idx to 0-idx
            doc = self.dataset.wtoken_doc_idx[i_word_token] - 1  # convert wtoken_doc_idx from
                                                            # 1-idx to 0-idx
            p_wtoken = p_wtoken_g_doc[doc, word_token]  # Probability of sampling current
                                                        # w token from d
            # Add log-probability of current token to running total for all w tokens
            w_loglikely += np.log(p_wtoken)  # pylint: disable=no-member
        tot_loglikely = x_loglikely + w_loglikely

        # -----------------------------------------------------------------------------
        # --- Update model log-likelihood history vector (if update_vectors == True)
        if update_vectors:
            self.loglikely_iter.append(self.iter)
            self.loglikely_x.append(x_loglikely)
            self.loglikely_w.append(w_loglikely)
            self.loglikely_tot.append(tot_loglikely)

        # --- Return loglikely values (used when computing log-likelihood for a
        # dataset-object containing hold-out data)
        return (x_loglikely, w_loglikely, tot_loglikely)

    def _get_peak_probs(self):
        """
        Compute a matrix giving p(x|r,t), using all x values in a dataset
        object, and each topic's spatial parameters.

        Returns
        -------
        peak_probs : :obj:`numpy.ndarray` of :obj:`numpy.64`
            nPeaks x nTopics x nRegions matrix of probabilities, giving
            probability of sampling each peak (x) from all subregions.
        """
        peak_probs = np.zeros(shape=(self.n_peak_tokens, self.n_topics,
                                     self.n_regions), dtype=float)
        for i_topic in range(self.n_topics):
            for j_region in range(self.n_regions):
                pdf = multivariate_normal.pdf(self.dataset.peak_vals,
                                              mean=self.regions_mu[i_topic][j_region][0],
                                              cov=self.regions_sigma[i_topic][j_region])
                peak_probs[:, i_topic, j_region] = pdf
        return peak_probs

    def _compute_prop_multinomial_from_zy_vectors(self, z, y):
        """
        Compute proportional multinomial probabilities of current x vector given
        current y vector, for all proposed y_i values.
        Note that this only returns values proportional to the relative
        probabilities of all proposals for y_i.

        Parameters
        ----------
        z : :obj:`numpy.ndarray` of :obj:`numpy.int64`
            A 1-by-T vector of current z counts for document d.

        y : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A 1-by-T vector of current y counts (plus gamma) for document d.

        Returns
        -------
        p : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A 1-by-T vector giving the proportional probability of z, given
            that topic t was incremented.
        """
        # Compute the proportional probabilities in log-space
        logp = z * np.log((y+1) / y)  # pylint: disable=no-member
        p = np.exp(logp - np.max(logp))  # Add a constant before exponentiating
                                         # to avoid any underflow issues
        return p

    # --------------------------------------------------------------------------------
    # <<<<< Export Methods >>>>> Print Topics, Model parameters, and Figures to file |
    # --------------------------------------------------------------------------------

    def get_spatial_probs(self):
        """
        Get conditional probability of selecting each voxel in the brain mask
        given each topic.

        Returns
        -------
        p_voxel_g_topic : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A voxel-by-topic array of conditional probabilities: p(voxel|topic).
            For cell ij, the value is the probability of voxel i being selected
            given topic j has already been selected.

        p_topic_g_voxel : :obj:`numpy.ndarray` of :obj:`numpy.float64`
            A voxel-by-topic array of conditional probabilities: p(topic|voxel).
            For cell ij, the value is the probability of topic j being selected
            given voxel i is active.
        """
        masker = self.dataset.masker
        affine = masker.volume.affine
        mask_ijk = np.vstack(np.where(masker.volume.get_data() > 0)).T
        mask_xyz = nib.affines.apply_affine(affine, mask_ijk)

        spatial_dists = np.zeros((mask_xyz.shape[0], self.n_topics), float)
        for i_topic in range(self.n_topics):
            for j_region in range(self.n_regions):
                pdf = multivariate_normal.pdf(mask_xyz,
                                              mean=self.regions_mu[i_topic][j_region][0],
                                              cov=self.regions_sigma[i_topic][j_region])
                spatial_dists[:, i_topic] += pdf
        p_topic_g_voxel = spatial_dists / np.sum(spatial_dists, axis=1)[:, None]
        p_topic_g_voxel = np.nan_to_num(p_topic_g_voxel, 0)  # might be unnecessary

        p_voxel_g_topic = spatial_dists / np.sum(spatial_dists, axis=0)[None, :]
        p_voxel_g_topic = np.nan_to_num(p_voxel_g_topic, 0)  # might be unnecessary

        return p_topic_g_voxel, p_voxel_g_topic

    def save(self, filename):
        """
        Pickle the Model instance to the provided file.

        Parameters
        ----------
        filename : str
            Pickle file to write Model instance to.
        """
        with open(filename, 'w') as fo:
            pickle.dump(self, fo)

    @classmethod
    def load(cls, filename):
        """
        Load a pickled Model instance from file.

        Parameters
        ----------
        filename : str
            Pickle file containing a saved Model instance.
        """
        try:
            with open(filename, 'r') as fi:
                dataset = pickle.load(fi)
        except UnicodeDecodeError:
            # Need to try this for python3
            with open(filename, 'r') as fi:
                dataset = pickle.load(fi, encoding='latin')

        return dataset

    def print_all_model_params(self, outputdir):
        """
        Run all export-methods: calls all print-methods to export parameters to
        files.

        Parameters
        ----------
        outputdir : str
            The name of the output directory.
        """
        # If output directory doesn't exist, make it
        if not isdir(outputdir):
            mkdir(outputdir)

        # print topic-word distributions for top-K words in easy-to-read format
        outfilestr = join(outputdir, 'Topic_X_Word_Probs.csv')
        self._print_topic_word_probs(outfilestr, 20)

        # print topic x word count matrix: m.n_word_tokens_word_by_topic
        outfilestr = join(outputdir, 'Topic_X_Word_CountMatrix.csv')
        self._print_topic_word_counts(outfilestr)

        # print activation-assignments to topics and subregions:
        # Peak_x, Peak_y, Peak_z, peak_topic_idx, peak_region_idx
        outfilestr = join(outputdir, 'ActivationAssignments.csv')
        self._print_activation_assignments(outfilestr)

    def _print_activation_assignments(self, outfilestr):
        """
        Print Peak->Topic and Peak->Subregion assignments for all x-tokens in
        dataset.

        Parameters
        ----------
        outfilestr : str
            The name of the output file.
        """
        with open(outfilestr, 'w+') as fid:
            # Print the column-headers
            fid.write('Peak_X,Peak_Y,Peak_Z,Topic_Assignment,Subregion_Assignment\n')

            # For each peak-token, print(out its coordinates and current topic/subregion assignment
            for i_peak_token in range(self.n_peak_tokens):
                # Note that we convert topic/subregion indices to 1-base idx
                outstr = '{0},{1},{2},{3},{4}\n'.format(self.dataset.peak_vals[i_peak_token, 0],
                                                        self.dataset.peak_vals[i_peak_token, 1],
                                                        self.dataset.peak_vals[i_peak_token, 2],
                                                        self.peak_topic_idx[i_peak_token]+1,
                                                        self.peak_region_idx[i_peak_token]+1)
                fid.write(outstr)

    def _print_topic_word_counts(self, outfilestr):
        """
        Print Topic->Word counts for all topics and words.

        Parameters
        ----------
        outfilestr : str
            The name of the output file.
        """
        with open(outfilestr, 'w+') as fid:
            # Print the topic-headers
            fid.write('WordLabel,')
            for i_topic in range(self.n_topics):
                fid.write('Topic_{0:02d},'.format(i_topic+1))
            fid.write('\n')

            # For each row / wlabel: wlabel-string and its count under each
            # topic (the \phi matrix before adding \beta and normalizing)
            for i_word in range(self.n_word_labels):
                fid.write('{0},'.format(self.dataset.word_labels[i_word]))

                # Print counts under all topics
                for j_topic in range(self.n_topics):
                    fid.write('{0},'.format(self.n_word_tokens_word_by_topic[i_word,
                                                                             j_topic]))
                # Newline for next wlabel row
                fid.write('\n')

    def _print_topic_word_probs(self, outfilestr, n_top_words=15):
        """
        Print Topic->Word probability distributions for top K words to File.

        Parameters
        ----------
        outfilestr : str
            The name of the output file.

        n_top_words : int, optional
            The number of top words to be written out for each topic.
        """
        with open(outfilestr, 'w+') as fid:
            # Compute topic->word probs and marginal topic-probs
            wprobs = self.n_word_tokens_word_by_topic + self.beta

            # Marginal topicprobs
            topic_probs = np.sum(wprobs, axis=0) / np.sum(wprobs)
            wprobs = wprobs / np.sum(wprobs, axis=0)  # Normalized word-probs

            # Get the sorted probabilities and indices of words under each topic
            rnk_vals = np.sort(wprobs, axis=0)
            rnk_vals = rnk_vals[::-1]
            rnk_idx = np.argsort(wprobs, axis=0)
            rnk_idx = rnk_idx[::-1]

            # Print the topic-headers
            for i_topic in range(self.n_topics):
                # Print each topic and it's marginal probability to columns
                fid.write('Topic_{0:02d},{1:.4f},'.format(i_topic+1,
                                                          topic_probs[i_topic]))
            fid.write('\n')

            # Print the top K word-strings and word-probs for each topic
            for i in range(n_top_words):
                for j_topic in range(self.n_topics):
                    # Print the kth word in topic t and it's probability
                    fid.write('{0},{1:.4f},'.format(self.dataset.word_labels[rnk_idx[i, j_topic]],
                                                    rnk_vals[i, j_topic]))
                fid.write('\n')

    def print_topic_figures(self, outputdir, backgroundpeakfreq=10):
        """
        Print Topic Figures: Spatial distributions and Linguistic distributions
        for top K words.

        Parameters
        ----------
        outputdir : str
            Output directory for topic figures.

        backgroundpeakfreq : int, optional
            Determines what proportion of peaks we show in the background of
            each figure. Default = 10.
        """
        # If output directory doesn't exist, make it
        if not isdir(outputdir):
            mkdir(outputdir)

        # Display parameters
        # ^^ This would need to be changed for handling different data-types
        opts_axlims = [[-75, 75], [-110, 90], [-60, 80]]
        regioncolors = ['r', 'b', 'm', 'g', 'c', 'b']

        # Get a subset of values to use as background (to illustrate extent of all peaks)
        backgroundvals = self.dataset.peak_vals[list(range(1, len(self.dataset.peak_vals)-1,
                                                           backgroundpeakfreq)), :]
        backgroundvals = np.transpose(backgroundvals)

        # Loop over all topics and make a figure for each
        for i_topic in range(self.n_topics):
            # Set up save file name (convert to base-1 indexing)
            outfilestr = '{0}/Topic_{1:02d}.png'.format(outputdir, i_topic+1)

            # Create figure
            fig = plt.figure(figsize=(10, 10), dpi=80)

            # <<<< Plot all points for topic from 3 different angles >>>>

            # --- REAR-VIEW: X-BY-Z
            ax1 = fig.add_subplot(221)
            ax1.axis('equal')
            # Plot background points in gray
            ax1.scatter(backgroundvals[0], backgroundvals[2], color='0.6',
                        s=12, marker='o', alpha=.15)
            # Plot all subregion points in the subregion colors
            for j_region in range(self.n_regions):
                idx = (self.peak_topic_idx == i_topic) & (self.peak_region_idx == j_region)
                vals = self.dataset.peak_vals[idx]
                valsplot = np.transpose(vals)
                ax1.scatter(valsplot[0], valsplot[2], c=regioncolors[j_region],
                            s=12, lw=0, marker='^', alpha=.5)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Z')
            ax1.set_xlim(opts_axlims[0])
            ax1.set_ylim(opts_axlims[2])

            # --- SIDE-VIEW: Y-BY-Z
            ax2 = fig.add_subplot(222)
            ax2.axis('equal')
            # Plot background points in gray
            ax2.scatter(backgroundvals[1], backgroundvals[2], color='0.6',
                        s=12, marker='o', alpha=.15)
            # Plot all subregion points in the subregion colors
            for j_region in range(self.n_regions):
                idx = (self.peak_topic_idx == i_topic) & (self.peak_region_idx == j_region)
                vals = self.dataset.peak_vals[idx]
                valsplot = np.transpose(vals)
                ax2.scatter(valsplot[1], valsplot[2], c=regioncolors[j_region],
                            s=12, lw=0, marker='^', alpha=.5)
            ax2.set_xlabel('Y')
            ax2.set_ylabel('Z')
            ax2.set_xlim(opts_axlims[1])
            ax2.set_ylim(opts_axlims[2])

            # --- TOP-VIEW: X-BY-Y
            ax3 = fig.add_subplot(223)
            ax3.axis('equal')

            # --- Plot background points in gray
            ax3.scatter(backgroundvals[0], backgroundvals[1], color='0.6',
                        s=12, marker='o', alpha=.15)

            # --- Plot all subregion points in the subregion colors
            for j_region in range(self.n_regions):
                idx = (self.peak_topic_idx == i_topic) & (self.peak_region_idx == j_region)
                vals = self.dataset.peak_vals[idx]
                valsplot = np.transpose(vals)
                ax3.scatter(valsplot[0], valsplot[1], c=regioncolors[j_region],
                            s=12, lw=0, marker='^', alpha=.5)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_xlim(opts_axlims[0])
            ax3.set_ylim(opts_axlims[1])

            # <<<< Print words & Region-probs >>>>

            # Get strings giving top K words and probs for the current topic
            n_top_words = 12
            wprobs = self.n_word_tokens_word_by_topic[:, i_topic] + self.beta
            wprobs = wprobs / np.sum(wprobs)
            # Get rankings of words
            wrnk = np.argsort(wprobs)
            wrnk = wrnk[::-1]

            # Create strings showing (1) top-K words (2) top-k probs for current topic
            outstr_labels = ''
            outstr_vals = ''
            for j_top_word in range(n_top_words):
                outstr_labels += '{0}\n'.format(self.dataset.word_labels[wrnk[j_top_word]])
                outstr_vals += '{0:5.3f}\n'.format(wprobs[wrnk[j_top_word]])

            # Fourth axis: Show top-k words and word-probs, then show region-probs
            ax4 = fig.add_subplot(224)
            ax4.set_xticklabels([])
            ax4.set_yticklabels([])
            ax4.set_yticks([])
            ax4.set_xticks([])
            ax4.set_title('Top k Words')
            plt.text(0.15, 0.98, outstr_labels, horizontalalignment='left',
                     verticalalignment='top')
            plt.text(0.65, 0.98, outstr_vals, horizontalalignment='left',
                     verticalalignment='top')

            # Now get subregion-probs for current topic
            rprobs = self.n_peak_tokens_region_by_topic[:, i_topic] + float(self.delta)
            rprobs = rprobs / sum(rprobs)

            # Print the region probs and means to axis
            outstr_region = 'Region-ID    p(r|t)     mu_1  mu_2  mu_3'
            plt.text(.03, .30, outstr_region, color='k',
                     horizontalalignment='left', verticalalignment='top')
            for j_region in range(self.n_regions):
                outstr_region = 'Region {0:02d}: {1:6.2f}  |  {2:6.1f}  {3:6.1f}  {4:6.1f}'
                outstr_region = outstr_region.format(j_region+1, rprobs[j_region],
                                                     self.regions_mu[i_topic][j_region][0][0],
                                                     self.regions_mu[i_topic][j_region][0][1],
                                                     self.regions_mu[i_topic][j_region][0][2])
                plt.text(.03, .22 - (.06*j_region), outstr_region,
                         color=regioncolors[j_region],
                         horizontalalignment='left', verticalalignment='top')

            # Save figure to file and close it
            fig.savefig(outfilestr, dpi=fig.dpi)
            plt.close(fig)

    # -----------------------------------------------------------------------------------------
    # <<<<< Utility Methods for Displaying Model >>>> Display Model summary, Get Model-String |
    # -----------------------------------------------------------------------------------------

    def display_model_summary(self, debug=False):
        """
        Print model summary to console.

        Parameters
        ----------
        debug : bool, optional
            Setting debug to True will print out additional information useful
            for debugging the model. Default = False.
        """
        print('--- Model Summary ---')
        print(' Current State:')
        print('\t Current Iteration   = {0}'.format(self.iter))
        print('\t Initialization Seed = {0}'.format(self.seed_init))
        if self.loglikely_tot:
            print('\t Current Log-Likely  = {0}'.format(self.loglikely_tot[-1]))
        else:
            print('\t Current Log-Likely  = ** Not Available: '
                  'Model not yet initialized **')
        print(' Model Hyper-Parameters:')
        print('\t Symmetric = {0}'.format(self.symmetric))
        print('\t n_topics  = {0}'.format(self.n_topics))
        print('\t n_regions = {0}'.format(self.n_regions))
        print('\t alpha     = {0:.3f}'.format(self.alpha))
        print('\t beta      = {0:.3f}'.format(self.beta))
        print('\t gamma     = {0:.3f}'.format(self.gamma))
        print('\t delta     = {0:.3f}'.format(self.delta))
        print('\t roi_size  = {0:.3f}'.format(self.roi_size))
        print('\t dobs      = {0}'.format(self.dobs))
        print(' Model Training-Data Information:')
        print('\t Dataset Label                 = {0}'.format(self.dataset.dataset_label))
        print('\t Word-Tokens (n_word_tokens)   = {0}'.format(len(self.dataset.wtoken_word_idx)))
        print('\t Peak-Tokens (n_peak_tokens)   = {0}'.format(self.n_peak_tokens))
        print('\t Word-Types (n_word_labels)    = {0}'.format(self.n_word_labels))
        print('\t Documents (n_docs)            = {0}'.format(self.n_docs))
        print('\t Peak-Dimensions (n_peak_dims) = {0}'.format(self.n_peak_dims))

        if debug:
            print(' DEBUG: Count matrices dimensionality:')
            print('\t n_peak_tokens_doc_by_topic = '
                  '{0!r}'.format(self.n_peak_tokens_doc_by_topic.shape))
            print('\t n_peak_tokens_region_by_topic = '
                  '{0!r}'.format(self.n_peak_tokens_region_by_topic.shape))
            print('\t n_word_tokens_word_by_topic = '
                  '{0!r}'.format(self.n_word_tokens_word_by_topic.shape))
            print('\t total_n_word_tokens_by_topic = '
                  '{0!r}'.format(self.total_n_word_tokens_by_topic.shape))
            print('\t regions_mu = {0!r}'.format(np.shape(self.regions_mu)))
            print('\t regions_sigma = {0!r}'.format(np.shape(self.regions_sigma)))
            print(' DEBUG: Indicator vectors:')
            print('\t wtoken_topic_idx = {0!r}'.format(self.wtoken_topic_idx.shape))
            print('\t peak_topic_idx = {0!r}'.format(self.peak_topic_idx.shape))
            print('\t peak_region_idx = {0!r}'.format(self.peak_region_idx.shape))
            print(' DEBUG: Sums (1):')
            print('\t sum(n_peak_tokens_doc_by_topic) = '
                  '{0!r}'.format(np.sum(self.n_peak_tokens_doc_by_topic)))
            print('\t sum(n_peak_tokens_region_by_topic) = '
                  '{0!r}'.format(np.sum(self.n_peak_tokens_region_by_topic)))
            print('\t sum(n_word_tokens_word_by_topic) = '
                  '{0!r}'.format(np.sum(self.n_word_tokens_word_by_topic)))
            print('\t sum(n_word_tokens_doc_by_topic) = '
                  '{0!r}'.format(np.sum(self.n_word_tokens_doc_by_topic)))
            print('\t sum(total_n_word_tokens_by_topic) = '
                  '{0!r}'.format(np.sum(self.total_n_word_tokens_by_topic)))
            print(' DEBUG: Sums (2):')
            print('\t sum(n_peak_tokens_doc_by_topic, axis=0) = '
                  '{0!r}'.format(np.sum(self.n_peak_tokens_doc_by_topic, axis=0)))
            print('\t sum(n_peak_tokens_region_by_topic, axis=0) = '
                  '{0!r}'.format(np.sum(self.n_peak_tokens_region_by_topic, axis=0)))

    def _get_model_name(self):
        """
        Get a model-string, unique to current dataset label + parameter
        settings.

        Returns
        -------
        outstr : str
            The name of the model.
        """
        outstr = ('{0}_{1}T_{2}R_alpha{3:.3f}_beta{4:.3f}_'
                  'gamma{5:.3f}_delta{6:.3f}_{7}dobs_{8:.1f}roi_{9}symmetric_'
                  '{10}').format(self.dataset.dataset_label, self.n_topics,
                                 self.n_regions, self.alpha, self.beta,
                                 self.gamma, self.delta, self.dobs,
                                 self.roi_size, self.symmetric, self.seed_init)
        return outstr


if __name__ == '__main__':
    print('Calling model.py as a script')
