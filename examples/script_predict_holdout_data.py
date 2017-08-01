# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
This script illustrates how to evaluate the predicted loglikelihood of a
holdout dataset using a trained gclda model
"""
from __future__ import print_function

from gclda.dataset import Dataset
from gclda.model import Model

# ------------------------------------------------------
# --- Configure gcLDA model and dataset Parameters   ---
# ------------------------------------------------------
dataset_label = '2015Filtered2_TrnTst1P1'

current_iter = 1000 # Saved model iteration

n_topics = 100  # Number of topics
n_regions = 2  # Number of subregions (any positive integer, but must equal 2 if
			   # symmetric subregion model)
alpha = .1  # Prior count on topics for each doc
beta = .01  # Prior count on word-types for each topic
gamma = .01  # Prior count added to y-counts when sampling z assignments
delta = 1.0  # Prior count on subregions for each topic
roi_size = 50  # Default spatial 'Region of interest' size (default value of
			   # diagonals in covariance matrix for spatial distribution, which
			   # the distributions are biased towards)
dobs = 25  # Region 'default observations' (# pseudo-observations biasing Sigma
		   # estimates in direction of default 'roi' value)
symmetric = True  # Use symmetry constraint on subregions? (symmetry requires n_regions = 2)
seed_init = 1  # Initial value of random seed

# --- Set up model_str identifier for saving/loading results based on model params ---
model_str = ('{0}_{1}T_{2}R_alpha{3:.3f}_beta{4:.3f}_gamma{5:.3f}_delta{6:.3f}_'
			 '{7}dobs_{8:.1f}roi_{9}symmetric_{10}').format(dataset_label, n_topics,
														    n_regions, alpha, beta,
														    gamma, delta, dobs,
														    roi_size, symmetric, seed_init)

# ----------------------------------------------
# --- Create dataset object for holdout-data ---
# ----------------------------------------------
# Assumes that the document-indices are aligned between train/test datasets

# Create a dataset object for test-data
test_name = '2015Filtered2_TrnTst1P2'
data_dir = '../datasets/neurosynth/'

# Create dataset object & Import data
test_dset = Dataset(test_name, data_dir)

# --------------------------------
# --- Load trained gcLDA model ---
# --------------------------------
# Set up model filename to load
results_rootdir = 'gclda_results'
results_outputdir = '{0}/{1}'.format(results_rootdir, model_str)
results_modelfile = '{0}/results_iter{1:02d}.p'.format(results_outputdir,
													   current_iter)

# Load compressed model object
print('loading model')
model = Model.load(results_modelfile)

# -----------------------------------------------------------------
# --- Run Log-likely computation on both training and test-data ---
# -----------------------------------------------------------------
# Recompute gcLDA log-likelihood on training data
loglikelyout_train = model._compute_log_likelihood(model.dataset, False)

# Compute trained gcLDA log-likelihood on holdout data
loglikelyout_test = model._compute_log_likelihood(test_dset, False)

# Print log-likelihoods for train and test:
print('log-likely train:')
print('loglikely_x | loglikely_w | loglikely_tot')
print('{0:5.1f}  {1:5.1f}  {2:5.1f}'.format(loglikelyout_train[0],
											loglikelyout_train[1],
											loglikelyout_train[2]))

print('log-likely test:')
print('loglikely_x | loglikely_w | loglikely_tot')
print('{0:5.1f}  {1:5.1f}  {2:5.1f}'.format(loglikelyout_test[0],
										    loglikelyout_test[1],
											loglikelyout_test[2]))
