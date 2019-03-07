import os
import pandas as pd
from glob import glob
import io


class IMDBReader(object):
    def __init__(self):
        self.train_data = None
        self.test_data = None

    def _read_folder(self, path, label, limit):
        texts = []
        files = glob(os.path.join(path, "*.txt"))
        if limit:
            files = files[:limit]

        for i, file in enumerate(files):
            with io.open(file, "r", encoding="utf8") as f:
                text = f.read()
                texts.append((text, label))

        return texts

    def _concat_and_shuffle_dataset(self, pos, neg):
        df = pd.concat([pd.DataFrame(pos),
                        pd.DataFrame(neg)])\
                .sample(frac=1)\
                .reset_index(drop=True)
        df.columns = ['review', 'label']

        return df

    def load_dataset(self, path, limit=None):
        train_pos = self._read_folder(os.path.join(path, "train", "pos"), 1, limit)
        train_neg = self._read_folder(os.path.join(path, "train", "neg"), 0, limit)
        test_pos = self._read_folder(os.path.join(path, "test", "pos"), 1, limit)
        test_neg = self._read_folder(os.path.join(path, "test", "neg"), 0, limit)

        train_data = self._concat_and_shuffle_dataset(train_pos, train_neg)
        test_data = self._concat_and_shuffle_dataset(test_pos, test_neg)

        self.train_data = train_data
        self.test_data = test_data
