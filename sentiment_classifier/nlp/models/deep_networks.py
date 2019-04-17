""" Code for deep neural networks models.
"""

import os
import tensorflow as tf
from sentiment_classifier.nlp.models import Model
from sentiment_classifier.nlp.tokenizer import KerasTokenizer
from sentiment_classifier.nlp.utils import load_word_vectors


class BiLSTM(Model):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.tokenizer = KerasTokenizer(
            pad_max_len=512,
            lower=False
        )

    def build_model(self, input_shape):
        word_vectors = load_word_vectors(
            filepath="data/wiki-news-300d-1M.vec",
            word_index=self.tokenizer.tokenizer.word_index,
            vector_size=300
        )

        word_vectors = load_word_vectors(
            filepath="./data/wiki-news-300d-1M.vec",
            word_index=self.tokenizer.tokenizer.word_index,
            vector_size=300
        )

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                word_vectors.shape[0],
                word_vectors.shape[1],
                weights=[word_vectors],
                trainable=False
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        return model

    def train(self, reader, filepath):
        x_train, x_test, y_train, y_test = self._make_training_data(reader)

        self.model = self.build_model(input_shape=x_train.shape[1])

        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        callbacks_list = [
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join("logs", self.name)
            ),
        ]

        self.model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
            epochs=5,
            callbacks=callbacks_list
        )

        self.save(filepath)
