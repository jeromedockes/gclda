# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
GCLDA- A Python package for performing Generalized Correspondence Latent
Dirichlet Allocation.
"""
from .dataset import Dataset
from .model import Model
from .decode import Decoder
from .version import __version__
from .due import due, Doi, BibTeX

__all__ = ['dataset', 'model', 'decode', 'utils']

# Citation for the algorithm.
due.cite(BibTeX('@incollection{NIPS2016_6274,'\
                'title = {Generalized Correspondence-LDA Models (GC-LDA) for '\
                'Identifying Functional Regions in the Brain},'\
                'author = {Rubin, Timothy and Koyejo, Oluwasanmi O and '\
                'Jones, Michael N and Yarkoni, Tal},'\
                'booktitle = {Advances in Neural Information Processing Systems 29},'\
                'editor = {D. D. Lee and M. Sugiyama and U. V. Luxburg and '\
                'I. Guyon and R. Garnett},'\
                'pages = {1118--1126},'\
                'year = {2016},'\
                'publisher = {Curran Associates, Inc.},'\
                'url = {http://papers.nips.cc/paper/6274-generalized-correspondence-'\
                'lda-models-gc-lda-for-identifying-functional-regions-in-the-brain.pdf}'\
                '}'),
         description='Introduces GC-LDA.',
         version=__version__, path='gclda', cite_module=True)

# Cleanup
del due, Doi, BibTeX
