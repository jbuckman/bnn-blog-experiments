# ga-prior-or-nah
For testing whether Bayesian NN priors are generalization-agnostic...or nah.

Requires Python 3 & Tensorflow.

Run: `python svhn_good_or_bad.py good` to train on just SVHN train set, or `python svhn_good_or_bad.py bad` to train on the SVHN train set and a corrupted version of the test set.
