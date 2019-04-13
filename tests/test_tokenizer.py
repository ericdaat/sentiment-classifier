import unittest
import numpy as np
from sentiment_classifier.nlp import tokenizer

texts = [
    "Hi there I feel very good today",
    "I am in an excellent mood"
]


class TestKerasTokenizer(unittest.TestCase):
    def test_non_sequence(self):
        t = tokenizer.KerasTokenizer(pad_max_len=None)
        t.fit(texts)

        results = t.transform(["Hi"])

        self.assertIsInstance(
            results,
            np.ndarray
        )

    def test_sequence(self):
        t = tokenizer.KerasTokenizer(pad_max_len=10)
        t.fit(texts)
        results = t.transform(["Hi"])

        self.assertIsInstance(
            results,
            np.ndarray
        )
