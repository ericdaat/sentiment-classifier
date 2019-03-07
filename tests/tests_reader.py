import unittest
import pandas as pd
from sentiment_classifier.nlp import reader


class TestImdb(unittest.TestCase):
    imdb = reader.IMDBReader()

    def test_init(self):
        self.assertEqual(TestImdb.imdb.train_data, None)
        self.assertEqual(TestImdb.imdb.test_data, None)

    def test_read_folder(self):
        limit = 5
        TestImdb.imdb.load_dataset("data/aclImdb", limit=limit)

        # test type
        self.assertIsInstance(TestImdb.imdb.train_data, pd.DataFrame)
        self.assertIsInstance(TestImdb.imdb.test_data, pd.DataFrame)

        # test shape
        self.assertEqual(TestImdb.imdb.train_data.shape, (2*limit, 2))
        self.assertEqual(TestImdb.imdb.test_data.shape, (2*limit, 2))

        # test columns
        self.assertListEqual(TestImdb.imdb.train_data.columns.tolist(),
                             ["review", "label"])
        self.assertListEqual(TestImdb.imdb.test_data.columns.tolist(),
                             ["review", "label"])
