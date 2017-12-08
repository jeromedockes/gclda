Introduction
============

The gcLDA model is a generalization of the correspondence-LDA model (Blei &
Jordan, 2003, "Modeling annotated data"), which is an unsupervised learning
model used for modeling multiple data-types, where one data-type describes the
other. The gcLDA model was introduced in the following paper:

- `Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain <https://timothyrubin.github.io/Files/GCLDA_NIPS_2016_Final_Plus_Supplement.pdf>`_

where the model was applied for modeling the `Neurosynth <http://neurosynth.org/>`_
corpus of fMRI publications. Each publication in this corpus consists of a set
of word tokens and a set of reported peak activation coordinates (x, y and z
spatial coordinates corresponding to brain locations).

When applied to fMRI publication data, the gcLDA model identifies a set of T
topics, where each topic captures a 'functional region' of the brain. More
formally: each topic is associated with (1) a spatial probability distribution
that captures the extent of a functional neural region, and (2) a probability
distribution over linguistic features that captures the cognitive function of
the region.

The gcLDA model can additionally be directly applied to other types of data.
For example, Blei & Jordan presented correspondence-LDA for modeling annotated
images, where pre-segmented images were represented by vectors of real-valued
image features. The code provided here should be directly applicable to these
types of data, provided that they are appropriately formatted. Note however that
this package has only been tested on the Neurosynth dataset; some modifications
may be needed for use with other datasets.

===============
 Notation
===============

====================================================================  ==================================================================================================================================================================
Notation                                                              Meaning
====================================================================  ==================================================================================================================================================================
:math:`w_{i}`, :math:`x_{i}`                                          The :math:`i` th word token and peak activation token in the corpus, respectively
:math:`N_{w}^{(d)}`,  :math:`N_{x}^{(d)}`                             The number of word tokens and peak activation tokens in document :math:`d`, respectively
:math:`D`                                                             The number of documents in the corpus
:math:`T`                                                             The number of topics in the corpus
:math:`R`                                                             The number of components/subregions in each topic's spatial distribution (subregions model)
:math:`z_{i}`                                                         Indicator variable assigning word token :math:`w_{i}` to a topic
:math:`y_{i}`                                                         Indicator variable assigning activation token :math:`x_{i}` to a topic
:math:`z^{(d)}`, :math:`y^{(d)}`                                      The set of all indicator variables for work tokens and activation tokens in document :math:`d`
:math:`N_{td}^{Y D}`                                                  The number of activation tokens within document :math:`d` that are assigned to topic :math:`t`
:math:`c_{i}`                                                         Indicator variable assigning activation token :math:`y_{i}` to a subregion (subregion models)
:math:`\Lambda^{t}`                                                   Placeholder for all spatial parameters for topic :math:`t`
:math:`\mu_{r}^{(t)}`, :math:`\sigma_{r}^{(t)}`                       Gaussian parameters for topic :math:`t`
:math:`\mu^{(t)}`, :math:`\sigma^{(t)}`                               Gaussian parameters for subregion :math:`r` in topic :math:`t`(subregion models)
:math:`\phi^{(t)}`                                                    Multinomial distribution over word types for topic :math:`t`
:math:`\phi_{w}^{(t)}`                                                Probability of word type :math:`w` given topic :math:`t`
:math:`\theta^{(d)}`                                                  Multinomial distribution over topics for document :math:`d`
:math:`\theta_{t}^{(d)}`                                              Probability of topic :math:`t` given document :math:`d`
:math:`\pi^{(t)}`                                                     Multinomial distribution over subregions for topic :math:`t` (subregion models)
:math:`\pi_{r}^{(t)}`                                                 Probability of subregion :math:`r` given topic :math:`t` (subregion models)
:math:`\beta`, :math:`\alpha`, :math:`\gamma`                         Model hyperparameters
:math:`\delta`                                                        Model hyperparameter (subregion models)
====================================================================  ==================================================================================================================================================================
