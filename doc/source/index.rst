.. gclda documentation master file, created by
   sphinx-quickstart on Fri Aug 11 10:19:08 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GC-LDA: Generalized Correspondence Latent Dirichlet Allocation
==============================================================

The `gclda package`_ can be used to perform functional decoding and encoding of
neuroimaging results.

.. _gclda package: https://github.com/tsalo/gclda

.. image:: https://travis-ci.org/tsalo/gclda.svg?branch=master
   :target: https://travis-ci.org/tsalo/gclda
.. image:: https://coveralls.io/repos/github/tsalo/gclda/badge.svg?branch=master
   :target: https://coveralls.io/github/tsalo/gclda?branch=master
.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0

Citations
==================
If you use GC-LDA, please cite:

  Rubin, T., Koyejo, O. O., Jones, M. N., & Yarkoni, T. (2016).
  `Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain.`_
  In Advances in Neural Information Processing Systems
  (pp. 1118-1126).

Additionally, if you use GC-LDA's decoding/encoding methods, please cite:

  Rubin, T. N., Koyejo, O., Gorgolewski, K. J., Jones, M. N., Poldrack, R. A.,
  & Yarkoni, T. (2017).
  `Decoding brain activity using a large-scale probabilistic functional-anatomical atlas of human cognition.`_
  PLOS Computational Biology, 13(10), e1005649.

The GC-LDA datasets and models available with the package are derived from
Neurosynth, so if you use those data, please cite:

  Yarkoni, T., Poldrack, R. A., Nichols, T. E., Van Essen, D. C., & Wager, T. D.
  (2011).
  `Large-scale automated synthesis of human functional neuroimaging data.`_
  Nature methods, 8(8), 665-670.

Finally, the `gclda` code may use a number of tools not listed above, so it is
recommended to use `duecredit`_ to output references associated with any code
you may run.

.. _Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain.: http://pilab.psy.utexas.edu/publications/Rubin_NIPS_2016.pdf
.. _Decoding brain activity using a large-scale probabilistic functional-anatomical atlas of human cognition.: https://doi.org/10.1371/journal.pcbi.1005649
.. _Large-scale automated synthesis of human functional neuroimaging data.: https://doi.org/10.1038/nmeth.1635

.. _duecredit: https://github.com/duecredit/duecredit

Contents
==================
.. toctree::
   :maxdepth: 2

   introduction
   api
   auto_examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
