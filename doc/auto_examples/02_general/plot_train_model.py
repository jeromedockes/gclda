# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _mod1:

=========================================
 Load dataset and train model
=========================================

An example of dataset.Dataset and model.Model.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from os.path import join

from gclda.dataset import Dataset
from gclda.model import Model
from gclda.utils import get_resource_path

###############################################################################
# Initialize dataset
# ----------------------------------
dataset_label = 'Neurosynth2015Filtered2_1000docs'
dataset_dir = join(get_resource_path(), 'datasets')
dataset = Dataset(dataset_label, dataset_dir)
dataset.display_dataset_summary()

###############################################################################
# Initialize model
# ----------------------
model = Model(dataset, n_topics=200, n_regions=2,
              alpha=.1, beta=.01, gamma=.01, delta=1.0,
              dobs=25, roi_size=50, symmetric=True,
              seed_init=1)
model.display_model_summary()

###############################################################################
# Run model (10 iterations)
# -------------------------
n_iterations = 10
for i in range(model.iter, n_iterations):
    model.run_complete_iteration(loglikely_freq=10, verbose=1)
model.display_model_summary()
