import unittest
import pandas as pd
from nlp import reader
from nlp.preprocessing import clean_text


class TestImdb(unittest.TestCase):
    imdb = reader.IMDBReader()

    def test_init(self):
        self.assertEqual(TestImdb.imdb.train_data, None)
        self.assertEqual(TestImdb.imdb.test_data, None)

    def test_read_folder(self):
        limit = 5

        for preprocessing_function in [None, clean_text]:
            TestImdb.imdb.load_dataset(
                limit=limit,
                preprocessing_function=preprocessing_function
            )

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
