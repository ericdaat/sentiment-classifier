# Sentiment Classifier

[![CircleCI](https://circleci.com/gh/ericdaat/sentiment-classifier/tree/master.svg?style=svg)](https://circleci.com/gh/ericdaat/sentiment-classifier/tree/master)
[![Documentation Status](https://readthedocs.org/projects/sentiment-classifier-api/badge/?version=latest)](https://sentiment-classifier-api.readthedocs.io/en/latest/?badge=latest)

## About

The goal of this project was to create a sentiment classifier
API that could use various models and datasets.

It is written in Python and uses the following libraries:

- Flask: for the API
- Tensorflow & Keras: for Machine Learning

For more details about the project, you can refer to
[these slides](https://github.com/ericdaat/slides/blob/master/sentiment_classifier_api.pdf).

So far we are only using the [IMDB large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/). But we plan to use more datasets later on.

## Installation

Here are the required steps to get started with the API:

- Clone the repository

- Download the IMDB dataset and place it in the data folder.
We use pre-trained word embeddings from FastText, so you might
want to download them to the data folder as well:

  - [Link to the IMDB Large Movie Review dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
  - [Link to the FastText embeddings](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)

- Create a virtual environment, and install the requirements
from `requirements.txt` file

- Add "sentiment_classifier" to your `PYTHONPATH`:

``` text
export PYTHONPATH=.:$PYTHONPATH
```

- Train the models by running:

``` text
python sentiment_classifier/scripts/train.py
```

- Run the API:

``` text
python sentiment_classifier/api/wsgi.py
```

- Test the API:

``` python
import requests

r = requests.post(
  "http://localhost:8000/api/classify",
  json={"text": "I love it"}
)
```

## Getting Started

Make sure to checkout this notebook to better understand how the code works: [Example Model Notebook](./sentiment_classifier/notebooks/example_model.ipynb)

You can also refer to the [documentation](https://sentiment-classifier-api.readthedocs.io/en/latest/?badge=latest).
