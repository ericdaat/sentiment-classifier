""" Code for shallow neural networks models.

So far we have:
 - LogisticRegression: Basic Logistic Regression model, that serves \
    as baseline.
"""

from nlp.models import Model
from nlp.tokenizer import KerasTokenizer
from keras import layers, models


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
