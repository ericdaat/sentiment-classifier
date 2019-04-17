""" Module containing the root Model class that every new model \
    must inherit from.

The Model class has the following attributes:

 - model: the ML model, so far built using Keras
 - tokenizer: responsible for mapping words into indices

The Model class implements the following methods:

 - build_model: builds the model
 - train: trains the model
 - save: saves the model weights & tokenizer
 - predict: predicts on sentences
 - _make_training_data: a private method that creates the train/test \
    matrices from a Reader object
"""
import os
from config import logger
from abc import abstractmethod, ABC
import tensorflow as tf


class Model(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.tokenizer = None
        self.model = None

        logger.info("Initializing {0}".format(self.name))

    def _make_training_data(self, reader):
        """ Method for preparing the training matrices.

        This function fits the tokenizer and creates train/test matrices.

        Args:
            reader (nlp.reader.Reader): a Reader instance that contains \
                the data to train the model on.

        Returns:
            x_train (np.ndarray)
            x_test (np.ndarray)
            y_train (np.ndarray)
            y_test (np.ndarray)

        """
        self.tokenizer.fit(reader.train_data["review"])

        x_train = self.tokenizer.transform(reader.train_data["review"])
        x_test = self.tokenizer.transform(reader.test_data["review"])

        y_train = reader.train_data["label"].values
        y_test = reader.test_data["label"].values

        return x_train, x_test, y_train, y_test

    def save(self, filepath):
        """Save the model weights and tokenizer

        Args:
            filepath (str): Path where to store the model.
        """

        os.makedirs(filepath, exist_ok=True)

        model_filepath = os.path.join(
            filepath,
            "{0}_model.pkl".format(self.name)
        )

        tokenizer_filepath = os.path.join(
            filepath,
            "{0}_tokenizer.pkl".format(self.name)
        )

        self.model.save(model_filepath)
        self.tokenizer.save(tokenizer_filepath)

    def load(self, filepath):
        """ Load the model weights and tokenizer

        Args:
            filepath (str): Path where to load the model.
        """

        model_filepath = os.path.join(
            filepath,
            "{0}_model.pkl".format(self.name)
        )

        tokenizer_filepath = os.path.join(
            filepath,
            "{0}_tokenizer.pkl".format(self.name)
        )

        self.model = tf.keras.models.load_model(model_filepath)
        self.tokenizer = self.tokenizer.load(tokenizer_filepath)

    @abstractmethod
    def build_model(self, input_shape):
        """ Method for building the model.

        Args:
            input_shape (int): Size of the input

        Returns:
            model (keras.Models): a keras model, to be compiled and trained
        """
        pass

    @abstractmethod
    def train(self, reader, filepath):
        """ Method for training the model. Must be implemented by
        the subclasses.

        Args:
            reader (nlp.reader.Reader): a Reader instance that contains \
                the data to train the model on.
            filepath (str): path to where the model will be stored

        Returns:
            None

        """
        pass

    def predict(self, texts, preprocessing_function):
        """ Predict on a sentence

        Args:
            texts (np.ndarray): the texts to predict on
            preprocessing_function: a preprocessing function, \
                from nlp.preprocessing module.

        Returns:
            cleaned_texts(list): the cleaned texts
        """
        if not (self.tokenizer and self.model):
            raise Exception("Model not trained")

        if isinstance(texts[0], str):
            cleaned_texts = [preprocessing_function(s) for s in texts]
        elif isinstance(texts[0], list):
            cleaned_texts = [preprocessing_function(s[0]) for s in texts]
        else:
            raise Exception("Wrong input kind for texts")

        cleaned_and_tokenized_texts = self.tokenizer.transform(cleaned_texts)
        predictions = self.model.predict(cleaned_and_tokenized_texts)

        return predictions
