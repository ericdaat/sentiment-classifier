import io
import numpy as np


def load_word_vectors(filepath, word_index, vector_size):
    """ Load word embeddings from a file.

    Args:
        filepath (str): path to the embedding file
        word_index (dict): word indices from the keras Tokenizer
        vector_size (int): embedding dimension, must match the \
            trained word vectors

    Returns:
        embedding_matrix (np.ndarray): a matrix of size \
            (len(word_index) * vector_size) that assigns each word \
            to its learned embedding.

    """
    embedding_matrix = np.zeros((len(word_index) + 1, vector_size))

    fin = io.open(filepath, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())

    for line in fin:
        tokens = line.rstrip().split(" ")
        if tokens[0] in word_index:
            w = word_index[tokens[0]]
            embedding_matrix[w] = np.fromiter(map(float, tokens[1:]), "float")

    return embedding_matrix
