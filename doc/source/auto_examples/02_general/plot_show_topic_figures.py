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
from shutil import rmtree
import matplotlib.pyplot as plt

from IPython.display import Image, display

from gclda.model import Model
from gclda.utils import get_resource_path

###############################################################################
# Load model
# ----------------------------------
model_file = join(get_resource_path(), 'models/model_Neurosynth2015Filtered2_temp.pklz')
model = Model.load(model_file)

###############################################################################
# Save topic figures
# -----------------------
out_dir = 'temp/'
model.save_topic_figures(outputdir=out_dir)

###############################################################################
# Show first topic figure
# -----------------------
display(Image(filename=join(out_dir, 'Topic_11.png')))

###############################################################################
# Show second topic figure
# ------------------------
display(Image(filename=join(out_dir, 'Topic_59.png')))

###############################################################################
# Show third topic figure
# -----------------------
display(Image(filename=join(out_dir, 'Topic_150.png')))

###############################################################################
# Clean up generated files
# ------------------------
rmtree(out_dir)
