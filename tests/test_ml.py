import unittest
import numpy as np
from sentiment_classifier.nlp import reader, preprocessing, tokenizer
from sentiment_classifier.nlp.models import ExampleModel
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
        imdb = reader.IMDBReader(path="./data/aclImdb")
        imdb.load_dataset(
            limit=10,
            preprocessing_function=preprocessing.clean_text
        )

        self.model.train(
            reader=imdb,
            filepath=TEST_MODEL_FILEPATH
        )

        def predict():
            pred = self.model.predict(
                texts=[["hi there"], ["how are you"]],
                preprocessing_function=preprocessing.clean_text
            )

            return pred

        pred = predict()
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (2, 1))

        self.model.load(filepath=TEST_MODEL_FILEPATH)

        pred = predict()
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (2, 1))


class TestExampleModel(CommonTests):
    model = ExampleModel()
