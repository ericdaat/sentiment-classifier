FROM python:latest

RUN mkdir -p /opt/sentiment-classifier/data
WORKDIR /opt/sentiment-classifier

# RUN wget -nv http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# RUN wget -nv https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
COPY data/aclImdb_v1.tar.gz data/
COPY data/wiki-news-300d-1M.vec.zip data/
