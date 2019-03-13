import logging
from nlp import ml, reader, preprocessing


def main():
    logging.basicConfig(level=logging.INFO)

    imdb = reader.IMDBReader()
    imdb.load_dataset("data/aclImdb",
                      limit=None,
                      preprocessing_function=preprocessing.clean_text)

    for model in [ml.LogisticRegression(), ml.CNN()]:
        model.train(reader=imdb)


if __name__ == "__main__":
    main()
