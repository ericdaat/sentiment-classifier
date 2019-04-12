import pickle
from abc import ABC, abstractmethod
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class BaseTokenizer(ABC):
    def __init__(self):
        self.tokenizer = None

    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.tokenizer = pickle.load(f)

        return self


class KerasTokenizer(BaseTokenizer):
    def __init__(self, pad_max_len, lower=False, filters="\t\n"):
        self.tokenizer = Tokenizer(lower=lower, filters=filters)
        self.pad_max_len = pad_max_len

    def fit(self, train_data):
        self.tokenizer.fit_on_texts(train_data)

    def transform(self, data):
        if not self.pad_max_len:
            x = self.tokenizer.texts_to_matrix(data)
        else:
            x = self.tokenizer.texts_to_sequences(data)
            x = pad_sequences(x, maxlen=self.pad_max_len)

        return x
