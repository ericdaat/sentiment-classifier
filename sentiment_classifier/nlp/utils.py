import io
import numpy as np


def load_word_vectors(fname, word_index, vector_size):
    embedding_matrix = np.zeros((len(word_index) + 1, vector_size))

    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())

    for line in fin:
        tokens = line.rstrip().split(" ")
        if tokens[0] in word_index:
            w = word_index[tokens[0]]
            embedding_matrix[w] = np.fromiter(map(float, tokens[1:]), "float")

    return embedding_matrix
