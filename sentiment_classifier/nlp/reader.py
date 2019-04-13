"""
We are using the IMDB Large Movie Reviews dataset from Stanford AI.

It provides 50,000 reviews on movies, splitted half-half in train/test \
    and labelled as positive or negative.

We provide an abstract class Reader that we can subclass for each dataset.

We do this to standardise the dataset loading, and make it easy to use \
    multiple datasets in the rest of the code with a common interface.

The IMDBReader class implements all the code needed to load the IMDB dataset.
"""
import os
import pandas as pd
from glob import glob
import io
from abc import ABC, abstractmethod


class Reader(ABC):
    def __init__(self, path):
        self.train_data = None
        self.test_data = None
        self.path = path

    @abstractmethod
    def load_dataset(self, path, limit=None, preprocessing_function=None):
        pass


class IMDBReader(Reader):
    def __init__(self, path):
        super(IMDBReader, self).__init__(path)

    def _read_folder(self, path, label, limit, preprocessing_function):
        """ Read the data from the IMDB dataset folder.

        The data can come from train/test and pos/neg folders. It is also \
            possible to add a limit to avoid reading all the files \
            (useful whendebugging). We can also add a preprocessing_function \
            from the nlp.preprocessing module.

        Args:
            path (str): path to the folder
            label (int): label of the folder (1/0 for pos/neg)
            limit (int): maximum number of files to load
            preprocessing_function: preprocessing function, from \
                nlp.preprocessing module

        Returns:
            list: list of texts, as str.
        """

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
        """Concatenate pos and neg examples and shuffle them.

        Args:
            pos (list): List of positive examples
            neg (list): List of negative examples

        Returns:
            pd.DataFrame: Merged dataframe, with the texts and their labels
        """

        concat_df = pd.concat([pd.DataFrame(pos), pd.DataFrame(neg)])
        concat_df = concat_df.sample(frac=1).reset_index(drop=True)

        concat_df.columns = ['review', 'label']

        return concat_df

    def load_dataset(self, limit=None, preprocessing_function=None):
        """ Load the IMDB dataset.

        This function can also:
         - preprocess using a custom function
         - set a maximum number of files to load

        Args:
            limit (int, optional): Defaults to None. \
                Max number of files to load.
            preprocessing_function (optional): Defaults to None. \
                Function for preprocessing the texts. \
                No preprocessing by default.
        """

        train_pos = self._read_folder(
            path=os.path.join(self.path, "train", "pos"),
            label=1,
            limit=limit,
            preprocessing_function=preprocessing_function
        )
        train_neg = self._read_folder(
            path=os.path.join(self.path, "train", "neg"),
            label=0,
            limit=limit,
            preprocessing_function=preprocessing_function
        )
        test_pos = self._read_folder(
            path=os.path.join(self.path, "test", "pos"),
            label=1,
            limit=limit,
            preprocessing_function=preprocessing_function
        )
        test_neg = self._read_folder(
            path=os.path.join(self.path, "test", "neg"),
            label=0,
            limit=limit,
            preprocessing_function=preprocessing_function
        )

        train_data = self._concat_and_shuffle_dataset(train_pos, train_neg)
        test_data = self._concat_and_shuffle_dataset(test_pos, test_neg)

        self.train_data = train_data
        self.test_data = test_data
