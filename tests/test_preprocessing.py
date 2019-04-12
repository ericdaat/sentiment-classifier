import unittest
from nlp import preprocessing


class TestCleanText(unittest.TestCase):
    def test_html(self):
        self.assertEqual(preprocessing.clean_text("<p>hi</p>"), "hi")
        self.assertEqual(preprocessing.clean_text("<br>hi</br>"), "hi")

    def test_split_punctuation(self):
        self.assertEqual(preprocessing.clean_text("hi!"), "hi !")
