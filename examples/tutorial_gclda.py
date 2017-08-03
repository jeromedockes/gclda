# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Tutorial Overview
This file illustrates basic usage of the python_gclda toolbox. Specifically,
we perform the following steps:
-   Import the python modules
-   Build a gclda dataset object using a small subset of the Neurosynth corpus,
    and import the data into this object from raw txt files
-   Build a gclda model instance
-   Train the gclda model instance (using fewer iterations than should be used
    for actual modeling)
-   Export figures that illustrate all topics in the trained model
-   View a few of the topics

Note: this tutorial will assume that your working directory is the 'examples/'
subdirectory within the gclda package. If it is not, the relative paths to the
datasets need to be modified when creating the variable 'dataset_label' below.
"""
from builtins import range
from os import mkdir
from os.path import join, isdir

from gclda.dataset import Dataset
from gclda.model import Model

# -----------------------------------------------
# --- Create a dataset object and import data ---
# -----------------------------------------------
# For this tutorial, we use a subset of 1000 documents from the Neurosynth dataset.
# This will run faster, but produce sparser and noisier topics than the full dataset.

# Create dataset object instance: 'dat'
dataset_label = '2015Filtered2_1000docs'  # The directory name containing
                                          # the dataset .txt files, which
                                          # will be used as a 'dataset label'
dataset_dir = '../datasets/neurosynth/'  # The relative path from the working
                                         # directory to the root-directory
                                         # containing the dataset folder
dat = Dataset(dataset_label, dataset_dir)  # Create a gclda dataset object 'dat'

# View dataset object after importing data:
dat.display_dataset_summary()

# -----------------------------------
# --- Create a gclda model object ---
# -----------------------------------
# Create gclda model, using T=100 topics, R=2 subregions per topic, and default
# values for all other hyper-parameters
# (See other sample scripts for details about all hyperparameters)

n_topics = 100
n_regions = 2  # Number of subregions in the gaussian mixture model used for
               # each topic's spatial distribution

# Create the model instance
model = Model(dat, n_topics, n_regions)

# Randomly initialize the model
model.initialize()

# View the model after initialization:
model.display_model_summary()

# ---------------------------------------------------
# --- Run the gcLDA training for a few iterations ---
# ---------------------------------------------------
# Note that we run for just a few iterations for the sake of time here.
# When training a model, we recommend running at least 1000 total iterations.

iterations = 25
# During training, the model will print details about the model log-likelihood, etc., to the
# console.
# The first parameter controls how often we compute the log-likelihood (which slows
# inference down slightly).
# The second parameter controls how much information gets printed to the console
# (0 = minimal, 2 = maximal).
for i in range(iterations):
    model.run_complete_iteration(1, 1)

# ---------------------------------------------
# --- Export figures illustrating the model ---
# ---------------------------------------------

# Here we export figures to file illustrating all topics in our trained model.
# Results will be placed into the subdirectories in the folder:
# 'examples/gclda_tutorial_results/' based on model parameter settings and how
# many iterations have run.
# Note that these topics will be much noiser than for a properly trained model
# (although some of the topics in these example results will capture known
# functional regions; e.g., topics 14/15). For results similar to those
# presented in our papers, you should:
# - Use the complete dataset (and appropriate hyper-parameter settings)
# - Train until log-likelihood converges

# Set up a rootdirectory to serve as a directory to store all tutorial results
results_rootdir = 'gclda_tutorial_results'  # We note that these results come from the
                                            # tutorial, as opposed to the scripts for
                                            # running full models
if not isdir(results_rootdir):
    mkdir(results_rootdir)

# We first use the method 'get_model_display_string' to get a string identifier that is
# unique to the combination of:
#  - dataset_label
#  - All model hyperparameters
# This is useful for saving model output

# Get model string identifier to use as a results directory
model_string = model.model_name

# Append the current model iteration to this directory name
output_data_dir = join(results_rootdir, '{0}_Iteration_{1}'.format(model_string,
                                                                   model.iter))

# We create a subdirectory of the outputdirectory to store figures
output_figures_dir = join(output_data_dir, 'Figures')

# Export some files giving topic-word distributions, as well as detailed accounts of
# all token->assignments
model.print_all_model_params(output_data_dir)

# Export Figures illustrating the spatial-extent and high probability word-types for
# the each topic in the model
model.print_topic_figures(output_figures_dir, 1)
