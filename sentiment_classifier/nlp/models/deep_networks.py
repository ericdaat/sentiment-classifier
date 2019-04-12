""" Code for deep neural networks models.

So far we have:

- CNN: Implementation of the paper "Convolutional Neural Networks \
    for Sentence Classification" by Yoon Kim.
"""

import os
from nlp.models import Model
from nlp.tokenizer import KerasTokenizer
from nlp.utils import load_word_vectors
from keras import layers, models, callbacks


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.tokenizer = KerasTokenizer(
            pad_max_len=200,
            lower=False
        )

    def train(self, reader, filepath):
        x_train, x_test, y_train, y_test = self._make_training_data(reader)

        word_vectors = load_word_vectors(
            filepath="data/wiki-news-300d-1M.vec",
            word_index=self.tokenizer.tokenizer.word_index,
            vector_size=300
        )

        embedding_layer = layers.Embedding(
            input_dim=word_vectors.shape[0],
            output_dim=word_vectors.shape[1],
            weights=[word_vectors],
            trainable=False
        )

        i = layers.Input(shape=(x_train.shape[1],))
        text_embedding = embedding_layer(i)
        convs = []

        for layer_params in [(5, 3), (5, 4)]:
            conv = layers.Conv1D(
                filters=layer_params[0],
                kernel_size=layer_params[1],
                activation="relu"
            )(text_embedding)

            conv = layers.GlobalMaxPooling1D()(conv)
            convs.append(conv)

        concat = layers.concatenate(convs)
        output = layers.Dense(1, activation="sigmoid")(concat)

        self.model = models.Model(inputs=[i], outputs=[output])

        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        callbacks_list = [
            callbacks.TensorBoard(log_dir=os.path.join("logs", self.name)),
        ]

        self.model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
            epochs=5,
            callbacks=callbacks_list
        )

        self.save(filepath)
