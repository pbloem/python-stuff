from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
from nltk import FreqDist
import numpy as np
import os
import datetime

"""


BAsed on https://github.com/ChunML/seq2seq/blob/master/seq2seq_utils.py

"""

# Sentence limit. Useful for debugging
LIMIT = None

def load_data(source, dist, max_len=100, vocab_size=10000):

    # Reading raw text from source and destination files
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    f = open(dist, 'r')
    y_data = f.read()
    f.close()

    print('raw data read')

    if LIMIT is not None:
        X_data = X_data[:LIMIT]
        y_data = y_data[:LIMIT]

    # Splitting raw text into array of sequences
    X = [text_to_word_sequence(x)[::-1] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]
    y = [text_to_word_sequence(y) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]

    # Creating the vocabulary set with the most common words (leaving room for PAD, START, UNK)
    dist = FreqDist(np.hstack(X))
    X_vocab = dist.most_common(vocab_size - 3)
    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common(vocab_size - 3)

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    X_ix_to_word = [word[0] for word in X_vocab]
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word.insert(0, '<PAD>')
    X_ix_to_word.insert(1, '<START>')
    X_ix_to_word.insert(2, '<UNK>')

    # Creating the word-to-index dictionary from the array created above
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}

    # Converting each word to its index value
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['<UNK>']

    y_ix_to_word = [word[0] for word in y_vocab]

    X_ix_to_word.insert(0, '<PAD>')
    y_ix_to_word.insert(1, '<START>')
    y_ix_to_word.insert(2, '<UNK>')

    y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}

    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]
            else:
                y[i][j] = y_word_to_ix['<UNK>']

    return X, len(X_vocab)+2, X_word_to_ix, X_ix_to_word, \
           y, len(y_vocab)+2, y_word_to_ix, y_ix_to_word

def process_data(word_sentences, max_len, word_to_ix):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences

