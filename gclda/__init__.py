# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:

from .version import __version__
from .due import due, Doi, BibTeX

due.cite(BibTeX("""@incollection{NIPS2016_6274,
                title = {Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain},
                author = {Rubin, Timothy and Koyejo, Oluwasanmi O and Jones, Michael N and Yarkoni, Tal},
                booktitle = {Advances in Neural Information Processing Systems 29},
                editor = {D. D. Lee and M. Sugiyama and U. V. Luxburg and I. Guyon and R. Garnett},
                pages = {1118--1126},
                year = {2016},
                publisher = {Curran Associates, Inc.},
                url = {http://papers.nips.cc/paper/6274-generalized-correspondence-lda-models-gc-lda-for-identifying-functional-regions-in-the-brain.pdf}
                }
                """),
         description='Introduces GC-LDA.',
         version=__version__, path='gclda', cite_module=True)
due.cite(BibTeX("""@article {Rubin059618,
            	author = {Rubin, Timothy N and Koyejo, Oluwasanmi and Gorgolewski, Krzysztof J and Jones, Michael N and Poldrack, Russell A and Yarkoni, Tal},
            	title = {Decoding brain activity using a large-scale probabilistic functional-anatomical atlas of human cognition},
            	year = {2016},
            	doi = {10.1101/059618},
            	publisher = {Cold Spring Harbor Labs Journals},
            	URL = {http://www.biorxiv.org/content/early/2016/06/18/059618},
            	eprint = {http://www.biorxiv.org/content/early/2016/06/18/059618.full.pdf},
            	journal = {bioRxiv}
                }
                """),
         description='Describes uses of GC-LDA.',
         version=__version__, path='gclda', cite_module=True)
