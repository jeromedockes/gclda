"""
This script is useful for training a full model.
It can either create a new model or load an existing model (depending on the
value of 'model.iter'), and then perform training updates on the model.
It will save a compressed model to disk every so often, in case program needs
to be halted and later resumed.
"""
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
import os
from os.path import join

from gclda.dataset import Dataset
from gclda.model import Model
from gclda.tests.utils import get_resource_path

# Set up dataset label & load/save directories
DATASET_LABEL = '20170728'
DATA_DIR = join(get_resource_path(), 'neurosynth/')

# Root-directory where model results are saved
OUT_DIR = join(get_resource_path(), 'models/')

# Sampling parameters
N_ITERS = 50  # Total iterations to run up to
SAVE_FREQ = 1000  # How often we save a model object and topic-distributions to file
LOGLIKELY_FREQ = 100  # How often we compute log-likelihood (which slows training
                      # down a bit, but is useful for tracking model progress)
VERBOSITY = 2  # How much information about sampler progress gets
               # printed to console (2 is max, 0 is min)

# gcLDA model parameters
N_TOPICS = 400	# Number of topics
N_REGIONS = 2  # Number of subregions (any positive integer, but must equal 2
               # if SYMMETRIC subregion model)
ALPHA = .1 	# Prior count on topics for each doc
BETA = .01 	# Prior count on word-types for each topic
GAMMA = .01  # Prior count added to y-counts when sampling z assignments
DELTA = 1.0  # Prior count on subregions for each topic
ROI_SIZE = 50  # Default spatial 'region of interest' size (default value of
               # diagonals in covariance matrix for spatial distribution, which the
               # distributions are biased towards)
DOBS = 25  # Region 'default observations' (# pseudo-observations biasing Sigma
           # estimates in direction of default 'ROI_SIZE' value)
SYMMETRIC = True  # Use symmetry constraint on subregions? (symmetry requires N_REGIONS = 2)
SEED_INIT = 1  # Initial value of random seed

# -----------------------------------------------------------------------------
# --- Set up directories and either Initialize model or load existing model ---
# -----------------------------------------------------------------------------

# --- Set up directories for saving/loading results ---
if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

# --- If starting a new model ---
# Create dataset object & Import data
dataset = Dataset(dataset_label=DATASET_LABEL, data_directory=DATA_DIR)  # Create dataset object
dataset.display_dataset_summary()  # Print dataset summary to console

# Create a model object (using the dataset object and all parameter settings) and initialize
model = Model(dataset, n_topics=N_TOPICS, n_regions=N_REGIONS,
              alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
              dobs=DOBS, roi_size=ROI_SIZE, symmetric=SYMMETRIC,
              seed_init=SEED_INIT)

model.initialize() # Randomly initialize all z, y and r assignments.
                   # Get initial spatial estimates

# Display initialized or loaded model summary
model.display_model_summary()

#  Run the gclda model until model.iter = N_ITERS
for i in range(model.iter, N_ITERS):
    # Cycle through an iteration of all parameter updates (update_z, update_y, update_regions)
    model.run_complete_iteration(LOGLIKELY_FREQ, VERBOSITY)

    # Save model and topic-word distributions every 'savefreq' iterations
    if model.iter % SAVE_FREQ == 0:
        OUT_FILE = join(OUT_DIR, 'model_{0}_temp.pkl'.format(DATASET_LABEL))
        model.save(OUT_FILE)

OUT_FILE = join(OUT_DIR, 'model_{0}.pkl'.format(DATASET_LABEL))
model.save(OUT_FILE)
