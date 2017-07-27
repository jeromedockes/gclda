# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Class and functions for dataset-related stuff.
"""
from __future__ import print_function
from os import mkdir
from os.path import join, isfile, isdir

import numpy as np
import pandas as pd


def import_neurosynth(neurosynth_dataset, dataset_label, out_dir='.',
                      counts_file=None, abstracts_file=None, email=None):
    """
    Transform Neurosynth's data into gcLDA-compatible files.
    """
    dataset_dir = join(out_dir, dataset_label)
    if not isdir(dataset_dir):
        mkdir(dataset_dir)
    
    # Word indices file
    orig_vocab = neurosynth_dataset.get_feature_names()
    if counts_file is None or not isfile(counts_file):
        from sklearn.feature_extraction.text import CountVectorizer
        
        if abstracts_file is None or not isfile(abstracts_file):
            from neurosynth.base.dataset import download_abstracts
            abstracts_df = download_abstracts(neurosynth_dataset, email=email)
            abstracts_df.to_csv(join(dataset_dir, 'dataset_abstracts.csv'),
                                index=False)
        else:
            abstracts_df = pd.read_csv(abstracts_file)
        vectorizer = CountVectorizer(vocabulary=orig_vocab)
        weights = vectorizer.fit_transform(abstracts_df['abstract'].tolist()).toarray()
        counts_df = pd.DataFrame(index=abstracts_df['pmid'], columns=orig_vocab,
                                data=weights)
        counts_df.to_csv(join(dataset_dir, 'feature_counts.txt'), sep='\t',
                         index_label='pmid')
    else:
        counts_df = pd.read_csv(counts_file, index_col='pmid', sep='\t')

    # Use subset of studies with abstracts for everything else
    counts_df.index = counts_df.index.astype(str)
    pmids = counts_df.index.tolist()[:1000]
    docidx_mapper = {pmid: i for (i, pmid) in enumerate(pmids)}
    
    # Create docidx column
    counts_df['id'] = counts_df.index
    counts_df['docidx'] = counts_df['id'].map(docidx_mapper)
    counts_df = counts_df.dropna(subset=['docidx'])
    counts_df = counts_df.drop('id', 1)
    
    # Remove words not found anywhere in the corpus
    counts_df = counts_df.loc[:, (counts_df != 0).any(axis=0)]
    
    # Get updated vocabulary
    word_labels = counts_df.columns.tolist()
    word_labels.remove('docidx')
    widx_mapper = {word: i for (i, word) in enumerate(word_labels)}
    
    # Melt dataframe and create widx column
    widx_df = pd.melt(counts_df, id_vars=['docidx'], var_name='word',
                      value_name='count')
    widx_df['widx'] = widx_df['word'].map(widx_mapper)
    
    # Replicate rows based on count
    widx_df = widx_df.loc[np.repeat(widx_df.index.values, widx_df['count'])]
    widx_df = widx_df[['docidx', 'widx']].astype(int)
    widx_df.sort_values(by=['docidx', 'widx'], inplace=True)
    widx_df.to_csv(join(dataset_dir, 'word_indices.txt'), sep='\t', index=False)
    
    # Word labels file
    word_labels_df = pd.DataFrame(columns=['word_label'], data=word_labels)
    word_labels_df.to_csv(join(dataset_dir, 'word_labels.txt'), sep='\t',
                          index=False)
    
    # Peak indices file
    peak_df = neurosynth_dataset.activations    
    peak_df['docidx'] = peak_df['id'].astype(str).map(docidx_mapper)
    peak_df = peak_df.dropna(subset=['docidx'])
    peak_indices = peak_df[['docidx', 'x', 'y', 'z']].values
    peak_indices_df = pd.DataFrame(columns=['docidx', 'x', 'y', 'z'],
                                   data=peak_indices)
    peak_indices_df['docidx'] = peak_indices_df['docidx'].astype(int)
    peak_indices_df.to_csv(join(dataset_dir, 'peak_indices.txt'), sep='\t',
                           index=False)
    
    # PMIDs file
    pmids_df = pd.DataFrame(columns=['pmid'], data=pmids)
    pmids_df.to_csv(join(dataset_dir, 'pmids.txt'), sep='\t', index=False)


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

    def import_data(self):
        """
        Import all data into the dataset object
        """
        # Import all word-labels into a list
        wlabels_file = join(self.data_directory, self.dataset_label, 'word_labels.txt')
        wlabels_df = pd.read_csv(wlabels_file, sep='\t')
        self.word_labels = wlabels_df['word_label'].tolist()

        # Import all document-pmids into a list
        pmids_file = join(self.data_directory, self.dataset_label, 'pmids.txt')
        pmids_df = pd.read_csv(pmids_file, sep='\t')
        self.pmids = pmids_df['pmid'].tolist()
        
        # Import all word-indices into a wtoken_word_idx and wtoken_doc_idx vector
        widx_file = join(self.data_directory, self.dataset_label, 'word_indices.txt')
        widx_df = pd.read_csv(widx_file, sep='\t')
        self.wtoken_doc_idx = widx_df['docidx'].tolist()
        self.wtoken_word_idx = widx_df['widx'].tolist()
        self.n_word_tokens = len(self.wtoken_word_idx)
        
        # Import all peak-indices into lists
        pidx_file = join(self.data_directory, self.dataset_label, 'peak_indices.txt')
        pidx_df = pd.read_csv(pidx_file, sep='\t')
        self.ptoken_doc_idx = pidx_df['docidx'].tolist()
        self.peak_vals = pidx_df[['x', 'y', 'z']].values
        self.n_peak_tokens, self.n_peak_dims = self.peak_vals.shape

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
        print('\t     word-types:   {0}'.format(len(self.word_labels)))
        print('\t     word-indices: {0}'.format(self.n_word_tokens))
        print('\t     peak-indices: {0}'.format(self.n_peak_tokens))
        print('\t     documents:    {0}'.format(len(self.pmids)))
        print('\t     peak-dims:    {0}'.format(self.n_peak_dims))

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
    GC_DATA.import_data()
    GC_DATA.display_dataset_summary()
