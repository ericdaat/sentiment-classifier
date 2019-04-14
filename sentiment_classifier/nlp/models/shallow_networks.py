""" Code for shallow neural networks models.
"""

import keras
import tensorflow as tf
from sentiment_classifier.nlp.models import Model
from sentiment_classifier.nlp.tokenizer import KerasTokenizer
from sentiment_classifier.nlp.utils import load_word_vectors


class ExampleModel(Model):
    def __init__(self):
        super(ExampleModel, self).__init__()

        self.tokenizer = KerasTokenizer(
            pad_max_len=512,
            lower=True
        )

    def build_model(self, input_shape):
        word_vectors = load_word_vectors(
            filepath="./data/wiki-news-300d-1M.vec",
            word_index=self.tokenizer.tokenizer.word_index,
            vector_size=300
        )

        model = keras.Sequential([
            keras.layers.Embedding(
                word_vectors.shape[0],
                word_vectors.shape[1],
                weights=[word_vectors],
                trainable=False
            ),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        return model

    def train(self, reader, filepath):
        x_train, x_test, y_train, y_test = self._make_training_data(reader)

        self.model = self.build_model(input_shape=x_train.shape[1])

        self.model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

        self.model.summary()

        print("\nTraining")

        self.model.fit(x=x_train,
                       y=y_train,
                       validation_split=0.1,
                       epochs=5)

        print("\nEvaluate on test data")

        self.model.evaluate(x_test, y_test, metrics=["accuracy"])

        self.save(filepath)
