# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
Class and functions for dataset-related stuff.
"""
from __future__ import print_function
from os import mkdir
from os.path import join, isfile, isdir
import cPickle as pickle

import numpy as np
import pandas as pd
from neurosynth.tests.utils import get_resource_path
from neurosynth.base.mask import Masker


def import_neurosynth(neurosynth_dataset, dataset_label, out_dir='.',
                      counts_file=None, abstracts_file=None, email=None):
    """Transform Neurosynth's data into gcLDA-compatible files.

    This function produces four files (word_indices.txt, word_labels.txt,
    peak_indices.txt, and pmids.txt) in a specified folder
    ({out_dir}/{dataset_label}), using data from a Neurosynth dataset and
    associated abstracts. These four files are necessary for creating a Dataset
    instance and running gcLDA.

    Parameters
    ----------
    neurosynth_dataset :
        A Neurosynth Dataset object containing data needed by gcLDA.

    dataset_label : str
        The name of the gcLDA dataset to be created. A folder will be created in
        `out_dir` named after the dataset and output files will be saved there.

    out_dir : str, optional
        Output directory. Parent folder of a new folder named after
        `dataset_label` where output files will be saved. By default, it uses
        the current directory.

    counts_file : str, optional
        A tab-delimited text file containing feature counts for the dataset.
        The first column is 'pmid', used for identifying articles. Other columns
        are features (e.g., unigrams and bigrams from Neurosynth), where each
        value is the number of times the feature is found in a given article.
        This file is different from the features.txt file provided by
        Neurosynth, as it should contain counts instead of tf-idf frequencies,
        but it should have the same format. Only one of `counts_file`,
        `abstracts_file`, and `email` needs to be specified.

    abstracts_file : str, optional
        A csv file containing abstracts of articles in the database. The first
        column is 'pmid', used for identifying articles. The second column is
        'abstract' and contains the article's abstract. The `abstracts_file` can
        be created using `download_abstracts` in the Neurosynth Python package.
        Only one of `counts_file`, `abstracts_file`, and `email` needs to be
        specified.

    email : str, optional
        A valid email address. If neither `counts_file` or `abstracts_file` is
        provided, then `import_neurosynth` will attempt to download article
        abstracts using Neurosynth's `download_abstracts` function. This calls
        PubMed to get PMIDs and abstracts, which requires an email address.
        Only one of `counts_file`, `abstracts_file`, and `email` needs to be
        specified.

    """
    dataset_dir = join(out_dir, dataset_label)
    if not isdir(dataset_dir):
        mkdir(dataset_dir)

    # Word indices file
    orig_vocab = neurosynth_dataset.get_feature_names()
    if counts_file is None or not isfile(counts_file):
        from sklearn.feature_extraction.text import CountVectorizer

        if abstracts_file is None or not isfile(abstracts_file):
            if email is not None:
                from neurosynth.base.dataset import download_abstracts
                abstracts_df = download_abstracts(neurosynth_dataset, email=email)
                abstracts_df.to_csv(join(dataset_dir, 'dataset_abstracts.csv'),
                                    index=False)
            else:
                raise Exception('A valid email address must be provided if '
                                'counts_file or abstracts_file are not.')
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
    pmids = counts_df.index.tolist()
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
    Class object for a gcLDA dataset.

    A Dataset contains data needed to run gcLDA models. It can also be used to
    view dataset information and can be saved to a pickled file.

    Parameters
    ----------
    dataset_label : str
        The name of the gcLDA dataset. Also the name of a subfolder in
        `data_directory` containing four files (word_indices.txt,
        word_labels.txt, peak_indices.txt, and pmids.txt) with the data needed
        to create the dataset.

    data_directory : str
        The path to the folder containing the data. Should contain a
        subdirectory named after `dataset_label` with files needed to generate
        a Dataset.

    mask_file : str, optional
        A brain mask file used to define the voxels included in the dataset. If
        not provided, the mask file used by Neurosynth will be used by default.

    Attributes
    ----------
    dataset_label : str
        The name of the dataset.

    masker : :obj:`neurosynth.base.mask.Masker`
        A Neurosynth Masker object used for masking and unmasking voxels
        according to a binary brain mask in stereotactic space.

    word_labels : :obj:`list` of :obj:`str`
        List of word-strings (wtoken_word_idx values are indices into this
        list).

    pmids : :obj:`list` of :obj:`numpy.int64`
        List of PubMed IDs (i.e., studies) in dataset.

    wtoken_doc_idx : :obj:`list` of :obj:`numpy.int64`
        List of document-indices for word-tokens.

    wtoken_word_idx : :obj:`list` of :obj:`numpy.int64`
        List of word-indices for word-tokens.

    ptoken_doc_idx : :obj:`list` of :obj:`numpy.int64`
        List of document-indices for peak-tokens x.

    peak_vals : :obj:`numpy.ndarray` of :obj:`numpy.int64`
        A focus-x-3 array of X, Y, and Z coordinates of foci in dataset in
        stereotactic (generally MNI152) space.


    """

    def __init__(self, dataset_label, data_directory, mask_file=None):
        # Dataset Info
        self.dataset_label = dataset_label

        if mask_file is None:
            resource_dir = get_resource_path()
            mask_file = join(resource_dir, 'MNI152_T1_2mm_brain.nii.gz')
        self.masker = Masker(mask_file)

        # Import all word-labels into a list
        wlabels_file = join(data_directory, self.dataset_label, 'word_labels.txt')
        wlabels_df = pd.read_csv(wlabels_file, sep='\t')
        self.word_labels = wlabels_df['word_label'].tolist()  # List of word-strings
                                                              # (wtoken_word_idx values are
                                                              # indices into this list)

        # Import all document-pmids into a list
        pmids_file = join(data_directory, self.dataset_label, 'pmids.txt')
        pmids_df = pd.read_csv(pmids_file, sep='\t')
        self.pmids = pmids_df['pmid'].tolist()

        # Import all word-indices into a wtoken_word_idx and wtoken_doc_idx vector
        widx_file = join(data_directory, self.dataset_label, 'word_indices.txt')
        widx_df = pd.read_csv(widx_file, sep='\t')
        self.wtoken_doc_idx = widx_df['docidx'].tolist()  # List of document-indices for word-tokens
        self.wtoken_word_idx = widx_df['widx'].tolist()  # List of word-indices for word-tokens

        # Import all peak-indices into lists
        pidx_file = join(data_directory, self.dataset_label, 'peak_indices.txt')
        pidx_df = pd.read_csv(pidx_file, sep='\t')
        self.ptoken_doc_idx = pidx_df['docidx'].tolist()  # List of document-indices for
                                                          # peak-tokens x
        self.peak_vals = pidx_df[['x', 'y', 'z']].values

    def save(self, filename):
        """
        Pickle the Dataset instance to the provided file.
        """
        with open(filename, 'w') as fo:
            pickle.dump(self, fo)

    @classmethod
    def load(cls, filename):
        """
        Load a pickled Dataset instance from file.
        """
        try:
            with open(filename, 'r') as fi:
                dataset = pickle.load(fi)
        except UnicodeDecodeError:
            # Need to try this for python3
            with open(filename, 'r') as fi:
                dataset = pickle.load(fi, encoding='latin')

        return dataset

    # -------------------------------------------------------------------
    #  Additional utility functions
    # -------------------------------------------------------------------

    def apply_stop_list(self, stoplistlabel):
        """
        Apply a stop list

        TODO: Implement, but put in import_neurosynth
        """
        print('Not yet implemented')

    # -------------------------------------------------------------------
    #  Functions for viewing dataset object
    # -------------------------------------------------------------------

    def display_dataset_summary(self):
        """
        View dataset summary.
        """
        print('--- Dataset Summary ---')
        print('self.dataset_label  = {0!r}'.format(self.dataset_label))
        print('\tword-types:   {0}'.format(len(self.word_labels)))
        print('\tword-indices: {0}'.format(len(self.wtoken_word_idx)))
        print('\tpeak-indices: {0}'.format(self.peak_vals.shape[0]))
        print('\tdocuments:    {0}'.format(len(self.pmids)))
        print('\tpeak-dims:    {0}'.format(self.peak_vals.shape[1]))

    def view_word_labels(self, n_word_labels=1000):
        """
        View first `n_word_labels` words in dataset.
        """
        print('First {0} word_labels:'.format(n_word_labels))
        for i in range(min(n_word_labels, len(self.word_labels))):
            print(self.word_labels[i])

        print('...')

    def view_doc_labels(self, n_pmids=1000):
        """
        View first `n_pmids` PMIDs in dataset.
        """
        print('First {0} pmids:'.format(n_pmids))
        for i in range(min(n_pmids, len(self.pmids))):
            print(self.pmids[i])

        print('...')

    def view_word_indices(self, n_word_indices=100):
        """
        View first `n_word_indices` word indices.
        """
        print('First {0} wtoken_doc_idx, wtoken_word_idx:'.format(n_word_indices))
        for i in range(min(n_word_indices, len(self.wtoken_word_idx))):
            print(self.wtoken_doc_idx[i], self.wtoken_word_idx[i])

        print('...')

    def view_peak_indices(self, n_peak_indices=100):
        """
        View first `n_peak_indices` peak indices.
        """
        print('Peak Locs Dimensions: {0}'.format(self.peak_vals.shape))
        print('First {0} ptoken_doc_idx, peak_x, peak_y, peak_z:'.format(n_peak_indices))
        for i in range(min(n_peak_indices, len(self.wtoken_word_idx))):
            print(self.ptoken_doc_idx[i], self.peak_vals[i])

        print('...')

if __name__ == '__main__':
    print('Calling dataset.py as a script')

    GC_DATA = Dataset('2015Filtered2_1000docs', '../datasets/neurosynth/')
    GC_DATA.display_dataset_summary()
