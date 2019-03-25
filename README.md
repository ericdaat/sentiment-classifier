# Sentiment Classifier

## About

The goal of this project was to create a sentiment classifier
API that could use various models and datasets.

It is written in Python and uses the following libraries:
- Flask: for the API
- Keras: for the Machine Learning Models

For more details about the project, you can refer to [these slides](https://github.com/ericdaat/slides/blob/master/sentiment_classifier_api.pdf).

## Installation

Here are the required steps to get started with the API:

- Clone the repository

- Download the IMDB dataset and place it in the data folder.
We use pre-trained word embeddings from FastText, so you might 
want to download them to the data folder as well:
  * [Link to the IMDB Large Movie Review dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
  * [Link to the FastTest embeddings](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)
  
- Create a virtual environment, and install the requirements
from `requirements.txt` file

- Add "sentiment_classifier" to your `PYTHONPATH`:

``` text
export PYTHONPATH=sentiment_classifier:$PYTHONPATH
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
r = requests.post("http://localhost:8000/api/classify",
                  json={"text": "I love it"})
```
