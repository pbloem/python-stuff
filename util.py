from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Input, Layer
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
import keras.utils
import keras.backend as K

from nltk import FreqDist
import numpy as np
import os, sys
import datetime
from keras.preprocessing import sequence

from scipy.misc import logsumexp

"""

Based on https://github.com/ChunML/seq2seq/blob/master/seq2seq_utils.py

"""

EXTRA_SYMBOLS = ['<PAD>', '<START>', '<UNK>', '<EOS>']

def load_data(source, dist, vocab_size=10000, limit=None):

    # Reading raw text from source and destination files
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    f = open(dist, 'r')
    y_data = f.read()
    f.close()

    print('raw data read')

    if limit is not None:
        X_data = X_data[:limit]
        y_data = y_data[:limit]

    # Splitting raw text into array of sequences
    X = [text_to_word_sequence(x) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 ]
    y = [text_to_word_sequence(y) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 ]

    # Creating the vocabulary set with the most common words (leaving room for PAD, START, UNK)
    dist = FreqDist(np.hstack(X))
    X_vocab = dist.most_common(vocab_size - len(EXTRA_SYMBOLS))
    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common(vocab_size - len(EXTRA_SYMBOLS))

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    X_ix_to_word = [word[0] for word in X_vocab]
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word = EXTRA_SYMBOLS + X_ix_to_word

    # Creating the word-to-index dictionary from the array created above
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}

    # print(X_word_to_ix['<PAD>'])
    # print(X_word_to_ix['the'])
    # print(X_word_to_ix['session'])
    # print(X_word_to_ix['resumption'])

    # Converting each word to its index value
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['<UNK>']

    # for s in range(3):
    #     print('___ ', ' '.join(X_ix_to_word[id] for id in X[s]))

    y_ix_to_word = [word[0] for word in y_vocab]
    y_ix_to_word = EXTRA_SYMBOLS + y_ix_to_word

    y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}

    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]
            else:
                y[i][j] = y_word_to_ix['<UNK>']

    return X, len(X_vocab)+2, X_word_to_ix, X_ix_to_word, \
           y, len(y_vocab)+2, y_word_to_ix, y_ix_to_word

def load_sentences(source, vocab_size=10000, limit=None):

    # Reading raw text from source and destination files
    f = open(source, 'r')
    X_data = f.read()
    f.close()

    print('raw data read')

    if limit is not None:
        X_data = X_data[:limit]

    # Splitting raw text into array of sequences
    X = [text_to_word_sequence(x) for x in X_data.split('\n') if len(x) > 0]

    # Creating the vocabulary set with the most common words (leaving room for PAD, START, UNK)
    dist = FreqDist(np.hstack(X))
    X_vocab = dist.most_common(vocab_size - len(EXTRA_SYMBOLS))

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    X_ix_to_word = [word[0] for word in X_vocab]
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word = EXTRA_SYMBOLS + X_ix_to_word

    # Creating the word-to-index dictionary from the array created above
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}

    # print(X_word_to_ix['<PAD>'])
    # print(X_word_to_ix['the'])
    # print(X_word_to_ix['session'])
    # print(X_word_to_ix['resumption'])

    # Converting each word to its index value
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['<UNK>']

    return X, len(X_vocab)+2, X_word_to_ix, X_ix_to_word

def load_char_data(source, limit=None, length=None):

    # Reading raw text from source and destination files
    f = open(source, 'r')
    x_data = f.read()
    f.close()

    print('raw data read')

    if limit is not None:
        x_data = x_data[:limit]

    # Splitting raw text into array of sequences
    if length is None:
        x = [list(line) for line in x_data.split('\n') if len(line) > 0]
    else:
        x = [list(chunk) for chunk in chunks(x_data, length)]

    # Creating the vocabulary set with the most common words (leaving room for PAD, START, UNK)
    chars = set()
    for line in x:
        for char in line:
            chars.add(char)

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    ix_to_char = list(chars)
    # Adding the word "ZERO" to the beginning of the array
    ix_to_char = ['<PAD>', '<START>', '<UNK>'] + ix_to_char

    # Creating the word-to-index dictionary from the array created above
    char_to_ix = {word:ix for ix, word in enumerate(ix_to_char)}

    # Converting each word to its index value
    for i, sentence in enumerate(x):
        for j, word in enumerate(sentence):
            if word in char_to_ix:
                x[i][j] = char_to_ix[word]
            else:
                x[i][j] = char_to_ix['<UNK>']



    return x, len(ix_to_char), char_to_ix, ix_to_char

def process_data(word_sentences, max_len, word_to_ix):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences


def batch_pad(x, batch_size, min_length=3, add_eos=False):
    """
    Takes a list of integer sequences, sorts them by lengths and pads them so that sentences in each batch have the
    same length.

    :param x:
    :return: A list of tensors containing equal-length sequences padded to the length of the longest sequence in the batch
    """

    x = sorted(x, key=lambda l : len(l))

    if add_eos:
        eos = EXTRA_SYMBOLS.index('<EOS>')
        x = [sent + [eos,] for sent in x]

    batches = []

    start = 0
    while start < len(x):
        end = start + batch_size
        if end > len(x):
            end = len(x)

        batch = x[start:end]

        mlen = max([len(l) for l in batch])

        if mlen >= min_length:
            batch = sequence.pad_sequences(batch, maxlen=mlen, dtype='int32', padding='post', truncating='post')

            batches.append(batch)

        start += batch_size


    print('max length per batch: ', [max([len(l) for l in batch]) for batch in batches])
    return batches

def to_categorical(batch, num_classes):
    """
    Converts a batch of length-padded integer sequences to a one-hot encoded sequence
    :param batch:
    :param num_classes:
    :return:
    """

    b, l = batch.shape

    out = np.zeros((b, l, num_classes))

    for i in range(b):
        seq = batch[0, :]
        out[i, :, :] = keras.utils.to_categorical(seq, num_classes=num_classes)

    return out

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def sample(preds, temperature=1.0):
    """
    Sample an index from a probability vector

    :param preds:
    :param temperature:
    :return:
    """

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

def sample_logits(preds, temperature=1.0):
    """
    Sample an index from a logit vector

    :param preds:
    :param temperature:
    :return:
    """
    preds = np.asarray(preds).astype('float64')

    if temperature == 0.0:
        return np.argmax(preds)

    preds = preds / temperature
    preds = preds - logsumexp(preds)

    choice = np.random.choice(len(preds), 1, p=np.exp(preds))

    return choice

class KLLayer(Layer):

    """
    Identity transform layer that adds KL divergence
    to the final model loss.

    During training, call
            K.set_value(kl_layer.weight, new_value)
    to scale the KL loss term.

    based on:
    http://tiao.io/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    """

    def __init__(self, weight = None, *args, **kwargs):
        self.is_placeholder = True
        self.weight = weight
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss((1.0 if self.weight is None else self.weight) * K.mean(kl_batch), inputs=inputs)

        return inputs

class Sample(Layer):
    """
    Performs sampling step
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var, eps = inputs

        z = K.exp(.5 * log_var) * eps + mu

        return z

    def compute_output_shape(self, input_shape):
        shape_mu, _, _ = input_shape
        return shape_mu