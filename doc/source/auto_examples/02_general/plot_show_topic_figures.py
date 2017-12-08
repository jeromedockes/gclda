# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _plot1:

=========================================
 Show topic distributions
=========================================

Plot topic figures and show them.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from os.path import join

import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.masking import unmask

from gclda.model import Model
from gclda.utils import get_resource_path

###############################################################################
# Load model
# ----------------------------------
model_file = join(get_resource_path(), 'models/Neurosynth2015Filtered2',
                  'model_200topics_2015Filtered2_10000iters.pklz')
model = Model.load(model_file)

###############################################################################
# Get spatial probability
# -----------------------
p_voxel_g_topic, _ = model.get_spatial_probs()

###############################################################################
# Show topic 10
# -----------------------
topic_no = 10
img = unmask(p_voxel_g_topic[:, topic_no], model.dataset.mask_img)

# Get strings giving top K words and probs for the current topic
wprobs = model.n_word_tokens_word_by_topic[:, topic_no] + model.beta
wprobs = wprobs / np.sum(wprobs)
word_probs = zip(*[model.dataset.word_labels, wprobs])
df = pd.DataFrame(columns=['term', 'probability'], data=word_probs)
df = df.sort_values(by='probability', ascending=False).reset_index(drop=True)
print df.head(12)

fig = plotting.plot_stat_map(img, display_mode='z')

###############################################################################
# Show topic 59
# ------------------------
topic_no = 58
img = unmask(p_voxel_g_topic[:, topic_no], model.dataset.mask_img)

# Get strings giving top K words and probs for the current topic
wprobs = model.n_word_tokens_word_by_topic[:, topic_no] + model.beta
wprobs = wprobs / np.sum(wprobs)
word_probs = zip(*[model.dataset.word_labels, wprobs])
df = pd.DataFrame(columns=['term', 'probability'], data=word_probs)
df = df.sort_values(by='probability', ascending=False).reset_index(drop=True)
print df.head(12)

fig = plotting.plot_stat_map(img, display_mode='z')

###############################################################################
# Show topic 150
# -----------------------
topic_no = 149
img = unmask(p_voxel_g_topic[:, topic_no], model.dataset.mask_img)

# Get strings giving top K words and probs for the current topic
wprobs = model.n_word_tokens_word_by_topic[:, topic_no] + model.beta
wprobs = wprobs / np.sum(wprobs)
word_probs = zip(*[model.dataset.word_labels, wprobs])
df = pd.DataFrame(columns=['term', 'probability'], data=word_probs)
df = df.sort_values(by='probability', ascending=False).reset_index(drop=True)
print df.head(12)

fig = plotting.plot_stat_map(img, display_mode='z')
