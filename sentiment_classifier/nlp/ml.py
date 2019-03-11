""" Module for Machine Learning models.

Here we define the Machine Learning models that we use
for detecting the sentiment on a given sentence.
Each model inherits from the `Model` abstract class
and must implement the required methods.
"""
import os
import pickle
from abc import abstractmethod, ABC
from keras import layers, models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nlp.utils import load_word_vectors
from nlp.preprocessing import clean_text


class Model(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.tokenizer = None
        self.model = None

    def save(self):
        """ Saves the model weights and tokenizer

        Returns:
            None
        """
        self.model.save(os.path.join("bin", "{0}_model.pkl".format(self.name)))

        with open(os.path.join("bin", "{0}_tokenizer.pkl".format(self.name)), "wb") as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        """ Load the model weights and tokenizer

        Returns:
            None
        """
        self.model = models.load_model(os.path.join("bin", "{0}_model.pkl".format(self.name)))

        with open(os.path.join("bin", "{0}_tokenizer.pkl".format(self.name)), "rb") as f:
            self.tokenizer = pickle.load(f)

    @abstractmethod
    def train(self, reader):
        """ Method for training the model. Must be implemented by
        the subclasses.

        Args:
            reader (nlp.reader.Reader): a Reader instance that contains
            the data to train the model on.

        Returns:
            None

        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def predict(self, sentence):
        """ Predict on a sentence

        Args:
            sentence (np.ndarray): the sentence to predict on

        Returns:
            cleaned_sentence (list): the cleaned sentence
        """
        if not (self.tokenizer and self.model):
            raise Exception("Model not trained")

        cleaned_sentence = [clean_text(sentence[0])]

        return cleaned_sentence


class LogisticRegression(Model):
    """ Linear Model that works on one hot word encoding.
    Basic but works pretty well on simple sentences.
    """
    def __init__(self):
        super(LogisticRegression, self).__init__()

    def _make_training_data(self, reader):
        self.tokenizer = Tokenizer(lower=False, filters="\t\n")
        self.tokenizer.fit_on_texts(reader.train_data["review"])

        x_train = self.tokenizer.texts_to_matrix(reader.train_data["review"])
        y_train = reader.train_data["label"].values

        x_test = self.tokenizer.texts_to_matrix(reader.test_data["review"])
        y_test = reader.test_data["label"].values

        return x_train, x_test, y_train, y_test

    def train(self, reader):
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

        self.save()
        
    def predict(self, sentence):
        sentence = super(LogisticRegression, self).predict(sentence)
        sentence = self.tokenizer.texts_to_matrix(sentence)

        return self.model.predict(sentence)[0][0]


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()

    def _make_training_data(self, reader):
        self.tokenizer = Tokenizer(lower=False, filters="\t\n")
        self.tokenizer.fit_on_texts(reader.train_data["review"])

        x_train = self.tokenizer.texts_to_sequences(reader.train_data["review"])
        x_train = pad_sequences(x_train, maxlen=1000)
        y_train = reader.train_data["label"].values

        x_test = self.tokenizer.texts_to_sequences(reader.test_data["review"])
        x_test = pad_sequences(x_test, maxlen=1000)
        y_test = reader.test_data["label"].values

        return x_train, x_test, y_train, y_test

    def train(self, reader):
        x_train, x_test, y_train, y_test = self._make_training_data(reader)
        word_vectors = load_word_vectors(filepath="data/wiki-news-300d-1M.vec",
                                         word_index=self.tokenizer.word_index,
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
        output = layers.Dense(2, activation="softmax")(hidden)

        self.model = models.Model(inputs=[i], outputs=[output])

        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

        self.model.fit(x=x_train,
                       y=y_train,
                       validation_data=(x_test, y_test),
                       epochs=5)

        self.save()

    def predict(self, sentence):
        sentence = super(CNN, self).predict(sentence)
        sentence = self.tokenizer.texts_to_sequences(sentence)
        sentence = pad_sequences(sentence, 1000)

        return self.model.predict(sentence)[0][1]
