""" This is the package where we keep the Machine Learning models.

So far we have different modules in it:

- model: module where the base class Model is defined. \
    Every new Machine Learning model should inherit from it. \
    It is an abstract class that provides the basic methods \
    for training and making predictions.
- shallow_networks: module where we keep the shallow networks models, \
    such as the basic LogisticRegression, or one hidden layer neural network.
- deep_networks: module for the deeper neural networks, like Recurrent \
    Neural Nets or Convolutionnal ones.
"""

from nlp.models.model import Model
from nlp.models.deep_networks import *
from nlp.models.shallow_networks import *
