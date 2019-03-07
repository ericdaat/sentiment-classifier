import os
from keras.preprocessing.text import Tokenizer
from keras import layers, models
import pickle
import numpy as np
from abc import abstractmethod, ABC


class Model(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        self.tokenizer = None
        self.model = None

    def save(self):
        self.model.save(os.path.join("bin", "{0}_model.pkl".format(self.name)))

        with open(os.path.join("bin", "{0}_tokenizer.pkl".format(self.name)), "wb") as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        self.model = models.load_model(os.path.join("bin", "{0}_model.pkl".format(self.name)))

        with open(os.path.join("bin", "{0}_tokenizer.pkl".format(self.name)), "rb") as f:
            self.tokenizer = pickle.load(f)

    @abstractmethod
    def train(self, reader):
        pass

    @abstractmethod
    def predict(self, texts):
        if not (self.tokenizer and self.model):
            raise Exception("Model not trained")


class LogisticRegression(Model):
    def __init__(self):
        super(LogisticRegression, self).__init__()

    def train(self, reader):
        self.tokenizer = Tokenizer(lower=False, filters='\t\n')
        self.tokenizer.fit_on_texts(reader.train_data['review'])

        x_train = self.tokenizer.texts_to_matrix(reader.train_data['review'])
        y_train = reader.train_data['label']

        x_test = self.tokenizer.texts_to_matrix(reader.test_data['review'])
        y_test = reader.test_data['label']

        i = layers.Input(shape=(x_train.shape[1],))
        h = layers.Dense(units=1, activation='sigmoid')(i)
        self.model = models.Model(inputs=[i], outputs=[h])

        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['binary_accuracy'])

        self.model.fit(x=x_train,
                       y=y_train,
                       validation_data=(x_test, y_test),
                       epochs=5)

        self.save()

    def predict(self, texts):
        super(LogisticRegression, self).predict(texts)
        texts = self.tokenizer.texts_to_matrix(texts)

        return self.model.predict(texts)
