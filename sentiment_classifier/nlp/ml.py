""" Module for Machine Learning models.

This module hosts the Machine Learning models. Every model subclasses a
Model abstract class that has the following attributes:

 - model: the ML model, so far built using Keras
 - tokenizer: responsible for mapping words into indices

The Model class implements the following methods:

 - train: trains the model
 - save: saves the model weights & tokenizer
 - predict: predicts on sentences
 - _make_training_data: a private method that creates the train/test
 matrices from a Reader object
"""
import os
from abc import abstractmethod, ABC
from keras import layers, models
from nlp.tokenizer import KerasTokenizer

from nlp.utils import load_word_vectors
from nlp.preprocessing import clean_text


class Model(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.tokenizer = None
        self.model = None

    def _make_training_data(self, reader):
        """ Method for preparing the training matrices.

        This function fits the tokenizer and creates train/test matrices.

        Args:
            reader (nlp.reader.Reader): a Reader instance that contains
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

        self.model = models.load_model(model_filepath)
        self.tokenizer = self.tokenizer.load(tokenizer_filepath)

    @abstractmethod
    def train(self, reader, filepath):
        """ Method for training the model. Must be implemented by
        the subclasses.

        Args:
            reader (nlp.reader.Reader): a Reader instance that contains
            the data to train the model on.
            filepath (str): path to where the model will be stored

        Returns:
            None

        """
        pass

    def predict(self, texts):
        """ Predict on a sentence

        Args:
            texts (np.ndarray): the texts to predict on

        Returns:
            cleaned_texts(list): the cleaned texts
        """
        if not (self.tokenizer and self.model):
            raise Exception("Model not trained")

        if isinstance(texts[0], str):
            cleaned_texts = [clean_text(s) for s in texts]
        elif isinstance(texts[0], list):
            cleaned_texts = [clean_text(s[0]) for s in texts]
        else:
            raise Exception("Wrong input kind for texts")

        cleaned_and_tokenized_texts = self.tokenizer.transform(cleaned_texts)
        predictions = self.model.predict(cleaned_and_tokenized_texts)

        return predictions


class LogisticRegression(Model):
    """ Linear Model that works on one hot word encoding.
    Basic but works pretty well on simple sentences.
    """
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.tokenizer = KerasTokenizer(
            pad_max_len=None,
            lower=True
        )

    def train(self, reader, filepath):
        x_train, x_test, y_train, y_test = self._make_training_data(reader)

        i = layers.Input(shape=(x_train.shape[1],))
        h = layers.Dense(units=1, activation="sigmoid")(i)
        self.model = models.Model(inputs=[i], outputs=[h])

        self.model.compile(loss="binary_crossentropy",
                           optimizer="sgd",
                           metrics=["binary_accuracy"])

        self.model.fit(x=x_train,
                       y=y_train,
                       validation_data=(x_test, y_test),
                       epochs=5)

        self.save(filepath)


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.tokenizer = KerasTokenizer(
            pad_max_len=1000,
            lower=False
        )

    def train(self, reader, filepath):
        x_train, x_test, y_train, y_test = self._make_training_data(reader)
        word_vectors = load_word_vectors(filepath="data/wiki-news-300d-1M.vec",
                                         word_index=self.tokenizer.tokenizer.word_index,
                                         vector_size=300)

        embedding_layer = layers.Embedding(input_dim=word_vectors.shape[0],
                                           output_dim=word_vectors.shape[1],
                                           weights=[word_vectors],
                                           trainable=False)

        i = layers.Input(shape=(x_train.shape[1],))
        text_embedding = embedding_layer(i)
        convs = []

        for layer_params in [(10, 2), (10, 3), (10, 4)]:
            conv = layers.Conv1D(filters=layer_params[0],
                                 kernel_size=layer_params[1],
                                 activation="relu")(text_embedding)
            conv = layers.GlobalMaxPooling1D()(conv)
            convs.append(conv)

        concat = layers.concatenate(convs)
        hidden = layers.Dropout(0.5)(concat)
        output = layers.Dense(1, activation="sigmoid")(hidden)

        self.model = models.Model(inputs=[i], outputs=[output])

        self.model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

        self.model.fit(x=x_train,
                       y=y_train,
                       validation_data=(x_test, y_test),
                       epochs=5)

        self.save(filepath)
