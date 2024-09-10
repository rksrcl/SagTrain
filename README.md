This repo includes the files I used for training with the Sagittarius model (https://www.nature.com/articles/s42256-023-00679-5) having adapted the Conditional Variational Autoencoder for predicting Drosophila (fruit fly) gene profiles at various time periods to output a zero-inflated negative binomial distribution instead of Gaussian.


The Drosophila datasets for these experiments is provided here: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190147.


Most of the model training can be found at SagTrain/training/training_and_eval.
