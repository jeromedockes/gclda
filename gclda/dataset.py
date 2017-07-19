# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
import numpy as np


# Class object for a gcLDA dataset
class Dataset:

    def __init__(self, dataset_label, data_directory):
        # Dataset Info
        self.dataset_label = dataset_label
        self.data_directory = data_directory

        # List of Word-labels
        self.wlabels = []  # List of word-strings (widx values are an indices into this list)

        # Word-indices
        self.widx = []  # list of word-indices for word-tokens
        self.wdidx = []  # List of document-indices for word-tokens
        self.nwords = 0

        # Peak-indices
        self.peak_didx = []  # List of document-indices for peak-tokens x
        self.peak_vals = np.ndarray(shape=(0, 0), dtype=int)  # Matrix with values for each peak token x
        self.npeaks = 0  # Number of peak tokens (x)
        self.peak_ndims = 0  # Dimensionality of x data

        # Document info (pmid)
        self.pmids = []

    # -------------------------------------------------------------------
    #  Functions for importing raw data from files into dataset object
    # -------------------------------------------------------------------

    # --- Import all data into the dataset object
    def import_all_data(self):
        self.import_word_labels()
        self.import_doc_labels()
        self.import_word_indices()
        self.import_peak_indices()

    # --- Import all word-labels into a list
    def import_word_labels(self):
        # Initialize wlabels variable
        self.wlabels = []
        # Initialize filestring to read from
        filestr = self.data_directory + self.dataset_label + '/wordlabels.txt'
        with open(filestr, 'r') as fid:
            # Read all wlabels from file into self.wlabels
            for line in fid:
                self.wlabels.append(line.strip())

    # --- Import all document-pmids into a list
    def import_doc_labels(self):
        # Initialize wlabels variable
        self.pmids = []
        # Initialize filestring to read from
        filestr = self.data_directory + self.dataset_label + '/pmids.txt'
        with open(filestr, 'r') as fid:
            # Read all wlabels from file into self.wlabels
            for line in fid:
                self.pmids.append(int(line.strip()))

    # --- Import all word-indices into a widx and wdidx vector
    def import_word_indices(self):
        # Initialize word-index variables
        self.wdidx = []
        self.widx = []
        # Initialize filestring to read from
        filestr = self.data_directory + self.dataset_label + '/wordindices.txt'
        with open(filestr, 'r') as fid:
            # Skip header-line
            line = fid.readline()
            # Read all word-indices from file into ints and append self.wdidx and self.widx
            inc = 0
            for line in fid:
                linedat = line.strip().split(',')
                self.wdidx.append(int(linedat[0]))
                self.widx.append(int(linedat[1]))
            self.nwords = len(self.widx)

    # --- Import all peak-indices into lists
    def import_peak_indices(self):
        # Initialize peak-index variables
        self.peak_didx = []
        tmp_peak_vals = []
        # Initialize filestring to read from
        filestr = self.data_directory + self.dataset_label + '/peakindices.txt'
        with open(filestr, 'r') as fid:
            # Skip header-line
            line = fid.readline()
            # Read all docindices and x/y/z coordinates into lists
            inc = 0
            for line in fid:
                linedat = line.strip().split(',')
                self.peak_didx.append(int(linedat[0]))
                # Append the ndims remaining vals to a N x Ndims array
                # ^^^If using different data (non-integer valued) 'int' needs to be changed to 'float'
                tmp_peak_vals.append(map(float, linedat[1:]))
            self.peak_vals = np.array(tmp_peak_vals)  # Directly convert the N x Ndims array to np array
            # Get the npeaks and dimensionality of peak data from the shape of peak_vals
            tmp = self.peak_vals.shape
            self.npeaks, self.peak_ndims = tmp

    # -------------------------------------------------------------------
    #  Additional utility functions
    # -------------------------------------------------------------------

    def apply_stop_list(self, stoplistlabel):
        print 'Not yet implemented'

    # -------------------------------------------------------------------
    #  Functions for viewing dataset object
    # -------------------------------------------------------------------

    # View dataset summary
    def display_dataset_summary(self):
        print "--- Dataset Summary ---"
        print "\t self.dataset_label  = %r" % self.dataset_label
        print "\t self.data_directory = %r" % self.data_directory
        print "\t # word-types:   %d" % len(self.wlabels)
        print "\t # word-indices: %d" % self.nwords
        print "\t # peak-indices: %d" % self.npeaks
        print "\t # documents:    %d" % len(self.pmids)  # ^^^ Update
        print "\t # peak-dims:    %d" % self.peak_ndims

    # View wordlabels
    def view_word_labels(self, N=1000):
        print 'First %d wlabels:' % N
        for i in range(min(N, len(self.wlabels))):
            print self.wlabels[i]
        print '...'

    # View doclabels
    def view_doc_labels(self, N=1000):
        print 'First %d pmids:' % N
        for i in range(min(N, len(self.pmids))):
            print self.pmids[i]
        print '...'

    # View N wordindices
    def view_word_indices(self, N=100):
        print 'First %d wdidx, widx:' % N
        for i in range(min(N, len(self.widx))):
            print self.wdidx[i], self.widx[i]
        print '...'

    # View N peak
    def view_peak_indices(self, N=100):
        print('Peak Locs Dimensions: %s' % (self.peak_vals.shape))
        print('First %d peak_didx, peak_x, peak_y, peak_z:' % N)
        for i in range(min(N, len(self.widx))):
            print self.peak_didx[i], self.peak_vals[i]

        print('...')

if __name__ == '__main__':
    print('Calling dataset.py as a script')

    gcdat = Dataset('2015Filtered2_1000docs', '../datasets/neurosynth/')
    gcdat.import_all_data()
    gcdat.display_dataset_summary()
else:
    print('Importing dataset.py')
