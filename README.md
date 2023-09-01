# PLS_ImagingGenetics
Public Repository for the paper 'Identifying the joint signature of brain atrophy and gene variant scores in Alzheimer's Disease' to appear in the Journal of Biomedical Informatics

<img src="./images/Pipeline.png">

This repository contains the code related to two papers.

If you use this work, please cite:

*Identifying the joint signature of brain atrophy and gene variant scores in Alzheimer's Disease*

by Federica Cruciani, Antonino Aparo, Lorenza Brusini, Carlo Combi, Silvia F. Storti, Rosalba Giugno, Gloria Menegaz, Ilaria Boscolo Galazzo.

[[paper]](To appear)

BibTeX citation: [To appear]
```bibtex
```

# Installation

Errors may arise while using this code if your python environment does not contain the dependencies listed in the [`environment.yml`](./environment.yml) file. 

## Data preprocessing
### Imaging
The [`preprocessing/imaging`](https://github.com/ggbioing/mcvae/tree/master/examples/mcvae) contains the necessary file to reproduce the ADNI T1-weighted images preprocessing from data cleaning to region-based thickness and volume feature extraction.

<img src="./images/workflow_T1.png">

The tools needed for this step are:
* FSL version 6.0
* FreeSurfer version 7.0 
* Ants


### Genetics
TODO
<img src="./images/workflow_gen.png">

## PLS modeling
TODO

## PLS explainability and validation

TODO




