# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Class and functions for dataset-related stuff.
"""
from __future__ import print_function
from os.path import join

import numpy as np


class Dataset(object):
    """
    Class object for a gcLDA dataset
    """

    def __init__(self, dataset_label, data_directory):
        """
        Class object for a gcLDA dataset
        """
        # Dataset Info
        self.dataset_label = dataset_label
        self.data_directory = data_directory

        # List of Word-labels
        self.word_labels = []  # List of word-strings (wtoken_word_idx values are an
                               # indices into this list)

        # Word-indices
        self.wtoken_word_idx = []  # list of word-indices for word-tokens
        self.wtoken_doc_idx = []  # List of document-indices for word-tokens
        self.n_word_tokens = 0

        # Peak-indices
        self.ptoken_doc_idx = []  # List of document-indices for peak-tokens x

        # Matrix with values for each peak token x
        self.peak_vals = np.ndarray(shape=(0, 0), dtype=int)
        self.n_peak_tokens = 0  # Number of peak tokens (x)
        self.n_peak_dims = 0  # Dimensionality of x data

        # Document info (pmid)
        self.pmids = []

    # -------------------------------------------------------------------
    #  Functions for importing raw data from files into dataset object
    # -------------------------------------------------------------------

    def import_all_data(self):
        """
        Import all data into the dataset object
        """
        self.import_word_labels()
        self.import_doc_labels()
        self.import_word_indices()
        self.import_peak_indices()

    def import_word_labels(self):
        """
        Import all word-labels into a list
        """
        # Initialize word_labels variable
        self.word_labels = []

        # Initialize filestring to read from
        filestr = join(self.data_directory, self.dataset_label, 'wordlabels.txt')
        with open(filestr, 'r') as fid:
            # Read all word_labels from file into self.word_labels
            for line in fid:
                self.word_labels.append(line.strip())

    def import_doc_labels(self):
        """
        Import all document-pmids into a list
        """
        # Initialize word_labels variable
        self.pmids = []

        # Initialize filestring to read from
        filestr = join(self.data_directory, self.dataset_label, 'pmids.txt')
        with open(filestr, 'r') as fid:
            # Read all word_labels from file into self.word_labels
            for line in fid:
                self.pmids.append(int(line.strip()))

    def import_word_indices(self):
        """
        Import all word-indices into a wtoken_word_idx and wtoken_doc_idx vector
        """
        # Initialize word-index variables
        self.wtoken_doc_idx = []
        self.wtoken_word_idx = []

        # Initialize filestring to read from
        filestr = join(self.data_directory, self.dataset_label, 'wordindices.txt')
        with open(filestr, 'r') as fid:
            # Skip header-line
            line = fid.readline()

            # Read all word-indices from file into ints and append
            # self.wtoken_doc_idx and self.wtoken_word_idx
            for line in fid:
                linedat = line.strip().split(',')
                self.wtoken_doc_idx.append(int(linedat[0]))
                self.wtoken_word_idx.append(int(linedat[1]))
            self.n_word_tokens = len(self.wtoken_word_idx)

    def import_peak_indices(self):
        """
        Import all peak-indices into lists
        """
        # Initialize peak-index variables
        self.ptoken_doc_idx = []
        tmp_peak_vals = []

        # Initialize filestring to read from
        filestr = join(self.data_directory, self.dataset_label, 'peakindices.txt')
        with open(filestr, 'r') as fid:
            # Skip header-line
            line = fid.readline()

            # Read all docindices and x/y/z coordinates into lists
            for line in fid:
                linedat = line.strip().split(',')
                self.ptoken_doc_idx.append(int(linedat[0]))

                # Append the ndims remaining vals to a N x Ndims array
                # ^^^If using different data (non-integer valued) 'int' needs
                # to be changed to 'float'
                tmp_peak_vals.append(map(float, linedat[1:]))

            # Directly convert the N x Ndims array to np array
            self.peak_vals = np.array(tmp_peak_vals)

            # Get the n_peaks and dimensionality of peak data from the shape of peak_vals
            tmp = self.peak_vals.shape
            self.n_peak_tokens, self.n_peak_dims = tmp

    # -------------------------------------------------------------------
    #  Additional utility functions
    # -------------------------------------------------------------------

    def apply_stop_list(self, stoplistlabel):
        """
        Apply a stop list
        """
        print('Not yet implemented')

    # -------------------------------------------------------------------
    #  Functions for viewing dataset object
    # -------------------------------------------------------------------

    def display_dataset_summary(self):
        """
        View dataset summary
        """
        print('--- Dataset Summary ---')
        print('\t self.dataset_label  = {0!r}'.format(self.dataset_label))
        print('\t self.data_directory = {0!r}'.format(self.data_directory))
        print('\t # word-types:   {0}'.format(len(self.word_labels)))
        print('\t # word-indices: {0}'.format(self.n_word_tokens))
        print('\t # peak-indices: {0}'.format(self.n_peak_tokens))
        print('\t # documents:    {0}'.format(len(self.pmids)))  # ^^^ Update
        print('\t # peak-dims:    {0}'.format(self.n_peak_dims))

    def view_word_labels(self, n_word_labels=1000):
        """
        View word labels
        """
        print('First {0} word_labels:'.format(n_word_labels))
        for i in range(min(n_word_labels, len(self.word_labels))):
            print(self.word_labels[i])
        print('...')

    def view_doc_labels(self, n_pmids=1000):
        """
        View doclabels
        """
        print('First {0} pmids:'.format(n_pmids))
        for i in range(min(n_pmids, len(self.pmids))):
            print(self.pmids[i])
        print('...')

    def view_word_indices(self, n_word_indices=100):
        """
        View N wordindices
        """
        print('First {0} wtoken_doc_idx, wtoken_word_idx:'.format(n_word_indices))
        for i in range(min(n_word_indices, len(self.wtoken_word_idx))):
            print(self.wtoken_doc_idx[i], self.wtoken_word_idx[i])
        print('...')

    def view_peak_indices(self, n_peak_indices=100):
        """
        View N peak
        """
        print('Peak Locs Dimensions: {0}'.format(self.peak_vals.shape))
        print('First {0} ptoken_doc_idx, peak_x, peak_y, peak_z:'.format(n_peak_indices))
        for i in range(min(n_peak_indices, len(self.wtoken_word_idx))):
            print(self.ptoken_doc_idx[i], self.peak_vals[i])

        print('...')

if __name__ == '__main__':
    print('Calling dataset.py as a script')

    GC_DATA = Dataset('2015Filtered2_1000docs', '../datasets/neurosynth/')
    GC_DATA.import_all_data()
    GC_DATA.display_dataset_summary()
