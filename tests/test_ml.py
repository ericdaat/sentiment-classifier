import unittest
import numpy as np
from nlp import reader, preprocessing, tokenizer
from nlp.models import shallow_networks, deep_networks
from keras.models import Model
from config import TEST_MODEL_FILEPATH


class CommonTests(unittest.TestCase):
    __test__ = False
    model = None

    @classmethod
    def setUpClass(cls):
        if cls is CommonTests:
            raise unittest.SkipTest()

    def test_init(self):
        self.assertEqual(self.model.model, None)
        self.assertIsInstance(self.model.tokenizer, tokenizer.BaseTokenizer)

    def test_pipeline(self):
        imdb = reader.IMDBReader()
        imdb.load_dataset(
            limit=10,
            preprocessing_function=preprocessing.clean_text
        )

        self.model.train(
            reader=imdb,
            filepath=TEST_MODEL_FILEPATH
        )

        self.assertIsInstance(self.model.tokenizer, tokenizer.BaseTokenizer)
        self.assertIsInstance(self.model.model, Model)

        pred = self.model.predict(
            texts=[["hi there"], ["how are you"]],
            preprocessing_function=preprocessing.clean_text
        )

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (2, 1))

        self.model.load(
            filepath=TEST_MODEL_FILEPATH
        )

        self.assertIsInstance(self.model.tokenizer, tokenizer.BaseTokenizer)
        self.assertIsInstance(self.model.model, Model)


class TestLogisticRegression(CommonTests):
    model = shallow_networks.LogisticRegression()


class TestCNN(CommonTests):
    model = deep_networks.CNN()
