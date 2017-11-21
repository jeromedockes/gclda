# gclda

This is a Python implementation of the Generalized Correspondence-LDA model (gcLDA).

[![Build Status](https://travis-ci.org/tsalo/gclda.svg?branch=master)](https://travis-ci.org/tsalo/gclda)
[![Coverage Status](https://coveralls.io/repos/github/tsalo/gclda/badge.svg?branch=master)](https://coveralls.io/github/tsalo/gclda?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Generalized Correspondence-LDA Model (GC-LDA)

The gcLDA model is a generalization of the correspondence-LDA model (Blei & Jordan, 2003, "Modeling annotated data"), which is an unsupervised learning model used for modeling multiple data-types, where one data-type describes the other. The gcLDA model was introduced in the following paper:

[Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain](https://timothyrubin.github.io/Files/GCLDA_NIPS_2016_Final_Plus_Supplement.pdf)

where the model was applied for modeling the [Neurosynth](http://neurosynth.org/) corpus of fMRI publications. Each publication in this corpus consists of a set of word tokens and a set of reported peak activation coordinates (x, y and z spatial coordinates corresponding to brain locations).

When applied to fMRI publication data, the gcLDA model identifies a set of T topics, where each topic captures a 'functional region' of the brain. More formally: each topic is associated with (1) a spatial probability distribution that captures the extent of a functional neural region, and (2) a probability distribution over linguistic features that captures the cognitive function of the region.

The gcLDA model can additionally be directly applied to other types of data. For example, Blei & Jordan presented correspondence-LDA for modeling annotated images, where pre-segmented images were represented by vectors of real-valued image features. The code provided here should be directly applicable to these types of data, provided that they are appropriately formatted. Note however that this package has only been tested on the Neurosynth dataset; some modifications may be needed for use with other datasets.

## Installation

Dependencies for this package are: scipy, numpy and matplotlib. If you don't have these installed, the easiest way to do so may be to use [Anaconda](https://www.continuum.io/downloads). Alternatively, [this page](http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/) provides a tutorial on installing them (note that the line "brew install gfortran" now must be replaced by "brew install gcc").

Additionally, some of the example scripts rely on gzip and cPickle (for saving compressed model instances to disk).

This code can be installed as a python package using:

	> python setup.py install

The classes needed to run a gclda model can then be imported into python using:

	> from gclda.dataset import Dataset

	> from gclda.model import Model


## Summary of gclda package

The repository consists of:

- two python classes (contained within the subdirectory 'gclda')
- several scripts and a tutorial that illustrate how to use these classes to train and export a gcLDA model (contained within the subdirectory 'examples')
- formatted versions of the Neurosynth dataset that can be used to train a gclda model (contained within the subdirectory [datasets/neurosynth](datasets/neurosynth))
- some examples of results from trained gcLDA models under different parameter settings (contained within subdirectories of 'example_results')

## Dataset formatting

The `Dataset` class requires four .txt files containing all dataset features that the gcLDA model needs to operate. Please see the example datasets in the [datasets/neurosynth](datasets/neurosynth) subdirectory, for examples of properly formatted data. For additional details about these files, please see [README.txt](documentation/README.txt) in the [documentation](documentation) subdirectory.

## Tutorial usage examples

For a simple tutorial illustrating usage of the gclda package, see the following file:
- [examples/tutorial_gclda.py](examples/tutorial_gclda.py)

This tutorial demonstrates how to (1) build a `Dataset` object (using a small subset of the Neurosynth dataset), (2) train a gcLDA `Model` on the `Dataset` object, and (3) export figures illustrating the trained `Model` to files for viewing.

There is also a version of this same tutorial in the following Jupyter notebook:
- [examples/tutorial_gclda_notebook.ipynb](examples/tutorial_gclda_notebook.ipynb)

## Code usage examples

For additional examples of how to use the code, please see the following scripts in the 'examples' subdirectory:

- [script_run_gclda.py](examples/script_run_gclda.py): Illustrates how to build a dataset object from a version of the Neurosynth dataset, and then train a gcLDA model (using the dataset object and several hyper-parameter settings that get passed to the model constructor).
- [script_export_gclda_figs.py](examples/script_export_gclda_figs.py): Illustrates how to export model data and png files illustrating each topic from a trained gcLDA model object.
- [script_predict_holdout_data.py](examples/script_predict_holdout_data.py): Illustrates how to compute the log-likelihood for a hold-out dataset.

Note that these scripts operate on the following version of the Neurosynth dataset: "2015Filtered2_TrnTst1P1", which is a training dataset from which a subset document data has been removed for testing (the test-data is in the dataset: "2015Filtered2_TrnTst1P2"). The complete Neurosynth dataset, without any test-data removed, is the version labeled "2015Filtered2".

Additional details about the gcLDA code, gcLDA hyper-parameter settings, and about these scripts are provided in the [README.txt](documentation/README.txt) in the [documentation](documentation) subdirectory, as well as in the comments of the [script_run_gclda.py](examples/script_run_gclda.py) file. Note that all three models presented in the source paper ('no subregions', 'unconstrained subregions' and 'constrained subregions') can be trained by modifying the model hyper-parameters appropriately.

## Example results for trained models

Results for some example trained models (including .png files illustrating all topics for the models) are included in the 'example_results' subdirectories.

## Using alternative spatial distributions

As described in our paper, the gcLDA model allows one to associate topics with any valid probability distribution for modeling the observed 'x' data. The package currently has the ability to train gcLDA models using Gaussian mixture models with any number of components, as well as Gaussian mixture models with spatial constraints. If you wish to modify the code to train a model using an alternative distribution, you will need to modify the following methods in `model.py`: (1) `_update_regions` (2) `_get_peak_probs`, as well as the lines of the (3) `__init__` method which allocate memory for storing the distributional parameters.

## Citing the code and data

To cite this module directly from the code, please use [DueCredit](https://github.com/duecredit/duecredit). For example, if you have a script named `run_gclda.py` that uses functions from this package, simply run:

	> python -m duecredit run_gclda.py

This will print a DueCredit report with relevant citations to your terminal. It will also create a file (`.duecredit.p`) containing those citations in the folder you run the script from.

If you want to compile citations by hand, please cite the following paper if you wish to reference this code:

- Timothy N Rubin, Oluwasanmi Koyejo, Michael N Jones, Tal Yarkoni (2016). [Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain](http://papers.nips.cc/paper/6274-generalized-correspondence-lda-models-gc-lda-for-identifying-functional-regions-in-the-brain).

Additionally, the following paper demonstrates a variety of cool applications for gcLDA models trained on Neurosynth (such as "brain decoding"):

- Timothy N Rubin, Oluwasanmi Koyejo, Krzysztof J Gorgolewski, Michael N Jones, Russell A Poldrack, Tal Yarkoni (2017). [Decoding brain activity using a large-scale probabilistic functional-anatomical atlas of human cognition](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005649). *PLOS Computational Biology*, *13*, e1005649.

To reference any of the datasets contained in this repository, or Neurosynth itself:

- Tal Yarkoni, Russell A. Poldrack, Thomas E. Nichols, David C. Van Essen, and Tor D. Wager (2011). "[Large-scale automated synthesis of human functional neuroimaging data](https://www.nature.com/nmeth/journal/v8/n8/full/nmeth.1635.html)." *Nature Methods*, *8*, 665-670.

Additionally, the complete Neurosynth datasets can be accessed at http://github.com/neurosynth/neurosynth-data (note however that those datasets need to be reformatted in order to make them work with the gclda package).

For additional details about Neurosynth please visit [neurosynth.org](http://neurosynth.org/).

## Documentation
To generate documentation files:
```
sphinx-apidoc --separate -M -f -o doc/source/ gclda/ gclda/due.py gclda/version.py gclda/tests/
make html
```
