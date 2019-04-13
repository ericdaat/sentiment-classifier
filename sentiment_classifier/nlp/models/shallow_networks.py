""" Code for shallow neural networks models.

So far we have:
 - LogisticRegression: Basic Logistic Regression model, that serves \
    as baseline.
"""

from sentiment_classifier.nlp.models import Model
from sentiment_classifier.nlp.tokenizer import KerasTokenizer
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

    def build_model(self, input_shape):
        i = layers.Input(shape=(input_shape,))
        h = layers.Dense(units=1, activation="sigmoid")(i)

        model = models.Model(inputs=[i], outputs=[h])

        return model

    def train(self, reader, filepath):
        x_train, x_test, y_train, y_test = self._make_training_data(reader)

        self.model = self.build_model(input_shape=x_train.shape[1])

        self.model.compile(loss="binary_crossentropy",
                           optimizer="sgd",
                           metrics=["binary_accuracy"])

        self.model.fit(x=x_train,
                       y=y_train,
                       validation_data=(x_test, y_test),
                       epochs=5)

        self.save(filepath)
