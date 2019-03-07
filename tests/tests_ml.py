import unittest
import numpy as np
from nlp import ml, reader, preprocessing
from keras.preprocessing.text import Tokenizer
from keras.models import Model


class TestLogisticRegression(unittest.TestCase):
    model = ml.LogisticRegression()

    def test_init(self):
        self.assertEqual(TestLogisticRegression.model.model, None)
        self.assertEqual(TestLogisticRegression.model.tokenizer, None)

    def test_train(self):
        imdb = reader.IMDBReader()
        imdb.load_dataset("data/aclImdb",
                          limit=100,
                          preprocessing_function=preprocessing.clean_text)

        TestLogisticRegression.model.train(reader=imdb)

        self.assertIsInstance(TestLogisticRegression.model.tokenizer, Tokenizer)
        self.assertIsInstance(TestLogisticRegression.model.model, Model)

    def test_load(self):
        TestLogisticRegression.model.load()

        self.assertIsInstance(TestLogisticRegression.model.tokenizer, Tokenizer)
        self.assertIsInstance(TestLogisticRegression.model.model, Model)

    def test_predict(self):
        self.assertIsInstance(
            TestLogisticRegression.model.predict([["hi there"]]),
            np.ndarray)
        self.assertEqual(
            TestLogisticRegression.model.predict([["hi there"], ["how are you"]]).shape,
            (2, 1))
