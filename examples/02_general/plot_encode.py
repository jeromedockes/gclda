# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _enc1:

=========================================
 Encode text into image
=========================================

An example of decode.encode.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from os.path import join
import matplotlib.pyplot as plt

from nilearn import plotting

from gclda.model import Model
from gclda.decode import encode
from gclda.utils import get_resource_path

###############################################################################
# Load model
# ----------------------------------
model_file = join(get_resource_path(), 'models/Neurosynth2015Filtered2',
                  'model_200topics_2015Filtered2_10000iters.pklz')
model = Model.load(model_file)

###############################################################################
# Encode text into image
# ----------------------
# text = 'painful stimulation during a language task'
text_file = '/tmp/gclda_query_list.txt'
with open(text_file) as f:
    queries = [l.strip() for l in f]

for text in queries:
    print(text)
    text_img, topic_weights = encode(model, text)

    ###############################################################################
    # Show encoded image
    # ---------------------
    fig = plotting.plot_stat_map(
        text_img, display_mode='z',
        threshold=0.00001,
        cut_coords=12,
        output_file='/tmp/gclda_figures/{}.png'.format(text.replace(' ', '_')))
                                # cut_coords=[-2, 22, 44, 66])

    ###############################################################################
# Plot topic weights
# ------------------
# fig2, ax2 = plt.subplots()
# ax2.plot(topic_weights)
# ax2.set_xlabel('Topic #')
# ax2.set_ylabel('Weight')
# fig2.show()
# plt.show()
