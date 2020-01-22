# bnn-blog-experiments
For testing whether Bayesian NN priors are generalization-agnostic...or nah.

Requires Python 3 & Tensorflow.

Run: `python svhn_good_or_bad.py good` to train on just SVHN train set, or `python svhn_good_or_bad.py bad` to train on the SVHN train set and a corrupted version of the test set.

Run: `python svhn_is_posterior_real.py` to train a BNN and also a regular neural network on the same data.
