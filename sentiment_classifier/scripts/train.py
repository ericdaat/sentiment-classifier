""" Script to train the classifiers.

Example usage:
    .. code-block:: bash

        python sentiment_classifier/scripts/train.py --models ExampleModel BiLSTM

"""

import argparse
from config import PROD_MODEL_FILEPATH, TEST_MODEL_FILEPATH
from sentiment_classifier.nlp import reader, preprocessing
from sentiment_classifier.nlp import models


def parse_arguments():
    """ Parse arguments from command line.

    Returns:
        argparse.ArgumentParser: Parser object, \
            with the arguments as attributes.
    """

    parser = argparse.ArgumentParser(description="Train classifiers")

    parser.add_argument(
        "--models",
        type=str,
        dest="models",
        required=True,
        help="models to train, separated by space."
    )

    parser.add_argument(
        "--limit",
        type=int,
        dest="limit",
        default=None,
        help="maximum number of texts to load. Defaults to None."
    )

    parser.add_argument(
        "--debug",
        type=bool,
        dest="debug",
        default=False,
        help="Toggle debug mode. Defaults to False."
    )

    args = parser.parse_args()

    return args


def main():
    """ Function in charge of training.

    1. Parse arguments from the command line
    2. Instanciate the requested models
    3. Train them
    """

    args = parse_arguments()

    imdb = reader.IMDBReader(path="./data/aclImdb")
    imdb.load_dataset(
        limit=args.limit,  # train on the full dataset
        preprocessing_function=preprocessing.clean_text
    )

    model_names = args.models.strip()

    for model_name in model_names.split(" "):
        model_class = getattr(models, model_name)
        model_instance = model_class()

        save_path = TEST_MODEL_FILEPATH if args.debug else PROD_MODEL_FILEPATH
        model_instance.train(reader=imdb, filepath=save_path)


if __name__ == "__main__":
    main()
