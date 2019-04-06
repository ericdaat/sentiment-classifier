import logging
from nlp import ml, reader, preprocessing
from config import PROD_MODEL_FILEPATH


def main():
    logging.basicConfig(level=logging.INFO)

    imdb = reader.IMDBReader()
    imdb.load_dataset(
        limit=None,
        preprocessing_function=preprocessing.clean_text
    )

    for model in [ml.LogisticRegression(), ml.CNN()]:
        model.train(reader=imdb, filepath=PROD_MODEL_FILEPATH)


if __name__ == "__main__":
    main()
