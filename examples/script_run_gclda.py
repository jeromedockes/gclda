"""
This script is useful for training a full model.
It can either create a new model or load an existing model (depending on the
value of 'current_iter'), and then perform training updates on the model.
It will save a compressed model to disk every so often, in case program needs
to be halted and later resumed.
"""
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
import os
from os.path import join
import gzip
import pickle as pickle

from gclda.dataset import Dataset
from gclda.model import Model

## Set up dataset label & load/save directories
# Set up dataset-label and dataset-directory we use for building dataset object
dataset_label = '2015Filtered2_TrnTst1p1'
data_directory = '../datasets/neurosynth/'

# Root-directory where model results are saved
results_rootdir = 'gclda_results'

# Sampling parameters
current_iter = 0  # Current model iteration: if 0, start new model, otherwise
                  # load & resume sampling existing model
total_iterations = 1000  # Total iterations to run up to
save_freq = 25 	# How often we save a model object and topic-distributions to file
loglikely_freq = 5 	# How often we compute log-likelihood (which slows training
                    # down a bit, but is useful for tracking model progress)
sampler_verbosity = 2  # How much information about sampler progress gets
                       # printed to console (2 is max, 0 is min)

# gcLDA model parameters
n_topics = 100	# Number of topics
n_regions = 2 	# Number of subregions (any positive integer, but must equal 2
                # if symmetric subregion model)
alpha = .1 	# Prior count on topics for each doc
beta = .01 	# Prior count on word-types for each topic
gamma = .01  # Prior count added to y-counts when sampling z assignments
delta = 1.0  # Prior count on subregions for each topic
roi_size = 50  # Default spatial 'region of interest' size (default value of
               # diagonals in covariance matrix for spatial distribution, which the
               # distributions are biased towards)
dobs = 25  # Region 'default observations' (# pseudo-observations biasing Sigma
           # estimates in direction of default 'roi_size' value)
symmetric = True  # Use symmetry constraint on subregions? (symmetry requires n_regions = 2)
seed_init = 1  # Initial value of random seed

# --- Set up model_str identifier for saving/loading results based on model params ---
model_str = ('{0}_{1}T_{2}R_alpha{3:.3f}_beta{4:.3f}_'
             'gamma{5:.3f}_delta{6:.3f}_{7}dobs_{8:.1f}roi_{9}symmetric_'
             '{10}').format(dataset_label, n_topics, n_regions, alpha, beta,
                            gamma, delta, int(dobs), roi_size, symmetric, seed_init)

# -----------------------------------------------------------------------------
# --- Set up directories and either Initialize model or load existing model ---
# -----------------------------------------------------------------------------

# --- Set up directories for saving/loading results ---
if not os.path.isdir(results_rootdir):
    os.mkdir(results_rootdir)
results_outputdir = join(results_rootdir, model_str)
if not os.path.isdir(results_outputdir):
    os.mkdir(results_outputdir)

# --- Initialize / Load model object (depending on current_iter) ---
if current_iter == 0:
    # --- If starting a new model ---
    # Create dataset object & Import data
    dataset = Dataset(dataset_label, data_directory)  # Create dataset object
    dataset.import_all_data()  # Import all data from txt files into object
    dataset.display_dataset_summary()  # Print dataset summary to console

    # Create a model object (using the dataset object and all parameter settings) and initialize
    model = Model(dataset, n_topics, n_regions, alpha, beta, gamma, delta, dobs,
                  roi_size, symmetric, seed_init)
    model.initialize() # Randomly initialize all z, y and r assignments.
                       # Get initial spatial estimates
else:
    # --- If resuming existing model ---
    print('Resuming model at iteration {0:02d}'.format(current_iter))

    # Set up loadfilename
    results_loadfile = '{0}/results_iter{1:02d}.p'.format(results_outputdir, current_iter)

    # Load compressed model object
    with gzip.open(results_loadfile, 'rb') as f:
        model = pickle.load(f)

# Display initialized or loaded model summary
model.display_model_summary()

#  Run the gclda model until model.iter = total_iterations
for i in range(model.iter, total_iterations):
    # Cycle through an iteration of all parameter updates (update_z, update_y, update_regions)
    model.run_complete_iteration(loglikely_freq, sampler_verbosity)

    # Save model and topic-word distributions every 'savefreq' iterations
    if model.iter % save_freq == 0:
        # Set up savefilenames
        savefilenm_pickle = '{0}/results_iter{1:02d}.p'.format(results_outputdir, model.iter)
        savefilenm_csv = '{0}/results_iter{1:02d}.csv'.format(results_outputdir, model.iter)

        # Save a gzip compressed model object to file
        with gzip.open(savefilenm_pickle, 'wb') as f:
            pickle.dump(model, f)

        # Print topic-word distributions
        model.print_topic_word_probs(savefilenm_csv)
