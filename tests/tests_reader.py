from sentiment_classifier.nlp import reader

imdb = reader.IMDBReader("data/aclImdb")

print(imdb.train_data)