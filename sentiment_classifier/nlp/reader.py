import os
import pandas as pd
from glob import glob
import logging


class IMDBReader(object):
    def __init__(self, path):
        self.train_data = None
        self.test_data = None

        train_data, test_data = self._load_dataset(path)
        self.train_data = train_data
        self.test_data = test_data

    def _read_folder(self, path, label):
        texts = []
        for file in glob(os.path.join(path, "*.txt")):
            with open(file, 'r') as f:
                text = f.read()
                texts.append((text, label))
            break
                
        return texts

    def _concat_and_shuffle_dataset(self, pos, neg):
        df = pd.concat([pd.DataFrame(pos),
                        pd.DataFrame(neg)])\
                .sample(frac=1)\
                .reset_index(drop=True)
        df.columns = ['review', 'label']

        return df

    def _load_dataset(self, path):
        train_pos = self._read_folder(os.path.join(path, "train", "pos"), 1)
        train_neg = self._read_folder(os.path.join(path, "train", "neg"), 0)
        test_pos = self._read_folder(os.path.join(path, "test", "pos"), 1)
        test_neg = self._read_folder(os.path.join(path, "test", "neg"), 0)

        train_data = self._concat_and_shuffle_dataset(train_pos, train_neg)
        test_data = self._concat_and_shuffle_dataset(test_pos, test_neg)

        return train_data, test_data
