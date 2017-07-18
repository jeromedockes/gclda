# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:

from .due import due, Doi, BibTex

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
         version=__version__, path='python_gclda_package', cite_module=True)
due.cite(Doi('10.1101/059618'),
         description='Describes uses of GC-LDA.',
         version=__version__, path='python_gclda_package', cite_module=True)
