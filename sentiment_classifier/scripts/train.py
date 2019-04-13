import logging
from nlp import reader, preprocessing
from nlp.models import LogisticRegression, CNN
from config import PROD_MODEL_FILEPATH


def main():
    logging.basicConfig(level=logging.INFO)

    imdb = reader.IMDBReader(path="./data/aclImdb")
    imdb.load_dataset(
        limit=None,
        preprocessing_function=preprocessing.clean_text
    )

    models_to_train = [
        # LogisticRegression(),
        CNN()
    ]

    for model in models_to_train:
        model.train(reader=imdb, filepath=PROD_MODEL_FILEPATH)


if __name__ == "__main__":
    main()
