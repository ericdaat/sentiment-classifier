import unittest
import numpy as np
from nlp import ml, reader, preprocessing
from keras.preprocessing.text import Tokenizer
from keras.models import Model


class CommonTests(unittest.TestCase):
    __test__ = False
    model = None

    @classmethod
    def setUpClass(cls):
        if cls is CommonTests:
            raise unittest.SkipTest()

    def test_init(self):
        self.assertEqual(self.model.model, None)
        self.assertEqual(self.model.tokenizer, None)

    def test_train(self):
        imdb = reader.IMDBReader()
        imdb.load_dataset("data/aclImdb",
                          limit=100,
                          preprocessing_function=preprocessing.clean_text)

        self.model.train(reader=imdb)

        self.assertIsInstance(self.model.tokenizer, Tokenizer)
        self.assertIsInstance(self.model.model, Model)

    def test_load(self):
        self.model.load()

        self.assertIsInstance(self.model.tokenizer, Tokenizer)
        self.assertIsInstance(self.model.model, Model)

    def test_predict(self):
        self.model.load()
        pred = self.model.predict([["hi there"], ["how are you"]])
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (2, 1))


class TestLogisticRegression(CommonTests):
    __test__ = True
    model = ml.LogisticRegression()
