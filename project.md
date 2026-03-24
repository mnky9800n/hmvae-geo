# variational autoencoder for the lily database

This project is about building a variational autoencoder to reconstruct the boreholes in the lily database. the lily database is found in the following link:

paper: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GC011287
zenodo data: 

1. https://zenodo.org/records/8408297
2. https://zenodo.org/records/10001855

The goal here is to use the following variables:

1. RGB values
2. bulk density
3. magnetic susceptibility
4. NGR

to calculate embeddings that can then be used to zero-shot predict all of the other variables in the dataset.

A paper draft concerning this effort can be found here:

an-unsupervised-embeddings-model-for-ocean-crust-lithology/

A core component of the zero shot prediction is to maximize the return of this. A missing component in the repository currently is a monte-carlo exploration of what maximizes zero shot prediction.
