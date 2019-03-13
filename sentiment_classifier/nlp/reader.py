"""
We are using the IMDB Large Movie Reviews dataset from Stanford AI.

It provides 50,000 reviews on movies, splitted half-half in train/test
and labelled as positive or negative.
We provide an abstract class Reader that we can subclass for each dataset.

We do this to standardise the dataset loading, and make it easy to use multiple
datasets in the rest of the code with a common interface.

The IMDBReader class implements all the code needed to load the IMDB dataset.
"""
import os
import pandas as pd
from glob import glob
import io
from abc import ABC, abstractmethod


class Reader(ABC):
    def __init__(self):
        self.train_data = None
        self.test_data = None

    @abstractmethod
    def load_dataset(self, path, limit=None, preprocessing_function=None):
        pass


class IMDBReader(Reader):
    def __init__(self):
        super(IMDBReader, self).__init__()

    def _read_folder(self, path, label, limit, preprocessing_function):
        texts = []
        files = glob(os.path.join(path, "*.txt"))
        if limit:
            files = files[:limit]

        for i, file in enumerate(files):
            with io.open(file, "r", encoding="utf8") as f:
                text = f.read()
                if preprocessing_function:
                    text = preprocessing_function(text)
                texts.append((text, label))

        return texts

    def _concat_and_shuffle_dataset(self, pos, neg):
        df = pd.concat([pd.DataFrame(pos),
                        pd.DataFrame(neg)])\
                .sample(frac=1)\
                .reset_index(drop=True)
        df.columns = ['review', 'label']

        return df

    def load_dataset(self, path, limit=None, preprocessing_function=None):
        train_pos = self._read_folder(os.path.join(path, "train", "pos"),
                                      1, limit, preprocessing_function)
        train_neg = self._read_folder(os.path.join(path, "train", "neg"),
                                      0, limit, preprocessing_function)
        test_pos = self._read_folder(os.path.join(path, "test", "pos"),
                                     1, limit, preprocessing_function)
        test_neg = self._read_folder(os.path.join(path, "test", "neg"),
                                     0, limit, preprocessing_function)

        train_data = self._concat_and_shuffle_dataset(train_pos, train_neg)
        test_data = self._concat_and_shuffle_dataset(test_pos, test_neg)

        self.train_data = train_data
        self.test_data = test_data
