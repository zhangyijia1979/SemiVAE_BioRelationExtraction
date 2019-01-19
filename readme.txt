Semi-supervised VAE model-based biomedical relation extraction
================================================================================

Semi-supervised VAE model-based biomedical relation extraction aims to classify the biomedical relation, such as PPI and DDI, in the given sentences, which is developed with Keras using Tensorflow as the backend. This package contains a demo implementation of Semi-supervised VAE described in the paper "Exploring semi-supervised variational autoencoders for biomedical relation extraction." This is research software, provided as is without express or implied warranties etc. see licence.txt for more details. We have tried to make it reasonably usable and provided help options, but adapting the system to new environments or transforming a corpus to the format used by the system may require significant effort. 

The details of related files are described as follows:

Data: ChemProt folder contains ChemProt Corpus. DDI folder contains DDI-Extraction2013 corpus. PPI folder contains BioInfer corpus.

Sourcecode: the folder that contains the source code and proprocessed DDI corpus.


============================ QUICKSTART ========================================

The demo implementation has tested on Keras 2.0.2, python 3.5 and Tensorflow 1.8.0.

User can use semi_vae_bioRE.py to automatic extract DDIs from the processed pkl files.

