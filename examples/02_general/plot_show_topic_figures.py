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

from IPython.display import Image, SVG

from gclda.model import Model
from gclda.utils import get_resource_path

###############################################################################
# Load model
# ----------------------------------
model_file = join(get_resource_path(), 'models/Neurosynth2015Filtered2',
                  'model_200topics_2015Filtered2_10000iters.pklz')
model = Model.load(model_file)

###############################################################################
# Save topic figures
# -----------------------
out_dir = 'temp/'
model.save_topic_figures(outputdir=out_dir)

###############################################################################
# Show first topic figure
# -----------------------
SVG(filename=join(out_dir, 'Topic_11.png'))
#Image(filename=join(out_dir, 'Topic_11.png'), embed=True)

###############################################################################
# Show second topic figure
# ------------------------
Image(filename=join(out_dir, 'Topic_59.png'), embed=True)

###############################################################################
# Show third topic figure
# -----------------------
Image(filename=join(out_dir, 'Topic_150.png'), embed=True)

###############################################################################
# Clean up generated files
# ------------------------
rmtree(out_dir)
