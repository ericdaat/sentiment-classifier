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

    def test_pipeline(self):
        imdb = reader.IMDBReader()
        imdb.load_dataset("data/aclImdb",
                          limit=10,
                          preprocessing_function=preprocessing.clean_text)

        self.model.train(reader=imdb)

        self.assertIsInstance(self.model.tokenizer, Tokenizer)
        self.assertIsInstance(self.model.model, Model)

        pred = self.model.predict([["hi there"], ["how are you"]])
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (2, 1))

        self.model.load()

        self.assertIsInstance(self.model.tokenizer, Tokenizer)
        self.assertIsInstance(self.model.model, Model)


class TestLogisticRegression(CommonTests):
    model = ml.LogisticRegression()


class TestCNN(CommonTests):
    model = ml.CNN()
