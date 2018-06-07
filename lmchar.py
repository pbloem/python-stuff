import keras

import keras.backend as K
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import \
    Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, Input, Reshape, LSTM, Embedding, RepeatVector,\
    TimeDistributed, Bidirectional, Concatenate, Lambda, SpatialDropout1D, Softmax
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

from keras.utils import multi_gpu_model

import tensorflow as tf

from sklearn import datasets

from tqdm import tqdm
import math, sys, os, random
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, MaxPooling2D, UpSampling2D, Flatten, Cropping2D
from keras.models import Model, Sequential
from keras.engine.topology import Layer
from keras.utils import to_categorical

import util

INDEX_FROM = 3
CHECK = 5

def sample(preds, temperature=0.1):
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

def generate_seq(
        model : Sequential,
        lstm : LSTM,
        seed, numchars, size):

    lstm.reset_states()

    tokens = []

    for s in seed:

        soh = util.to_categorical(np.asarray([[s]]), numchars)
        model.predict(soh)
        tokens.append(s)

    # generate a fixed number of words
    for _ in range(size):

        toh = util.to_categorical(np.asarray([[tokens[-1]]]), numchars)
        next_probs = model.predict(toh)
        next_token = sample(next_probs[0, 0, :])

        tokens.append(next_token)

    return tokens

def sparse_loss(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def go(options):
    lstm_hidden = options.lstm_capacity

    print('devices', device_lib.list_local_devices())

    if options.task == 'europarl':

        dir = options.data_dir
        x, numchars, char_to_ix, ix_to_char = \
            util.load_char_data(dir+os.sep+'europarl-v8.fi-en.en')

        x_max_len = max([len(sentence) for sentence in x])

        print('max sequence length ', x_max_len)
        print(len(ix_to_char), ' distinct characters')

        x = util.batch_pad(x, options.batch)

        def decode(seq):
            return ''.join(ix_to_char[id] for id in seq)

    else:
        raise Exception('Dataset name not recognized.')

    print('Data Loaded.')

    print(sum([b.shape[0] for b in x]), ' sentences loaded')

    for i in range(3):
        batch = random.choice(x)
        print(batch[0, :])
        print(decode(batch[0, :]))

    ## Define model
    input = Input(shape=(None, numchars))

    tohidden   = Dense(lstm_hidden)
    decoder_lstm = LSTM(lstm_hidden, return_sequences=True)
    fromhidden = Dense(numchars)

    h = TimeDistributed(tohidden)(input)
    h = decoder_lstm(h)
    out = TimeDistributed(fromhidden)(h)

    model = Model(input, out)

    opt = keras.optimizers.Adam(lr=options.lr)

    model.compile(opt, 'categorical_crossentropy')
    model.summary()

    epochs = 0
    while epochs < options.epochs:

        for batch in tqdm(x):
            n = batch.shape[0]

            batch_shifted = np.concatenate([np.ones((n, 1)), batch], axis=1)  # prepend start symbol
            batch_shifted = util.to_categorical(batch_shifted, numchars)

            batch_out = np.concatenate([batch, np.zeros((n, 1))], axis=1)     # append pad symbol
            batch_out = util.to_categorical(batch_out, numchars)     # output to one-hots

            model.train_on_batch(batch_shifted, batch_out)

        epochs += options.out_every

        gen_length = 1

        # Copy the decoder LSTM to a stateful one
        stateful_lstm = LSTM(lstm_hidden, input_dim=lstm_hidden, input_length=gen_length, batch_size=1, stateful=True, return_sequences=True)
        stateful_lstm.build((1, gen_length, lstm_hidden))
        stateful_lstm.set_weights(decoder_lstm.get_weights())

        tohidden_copy = Dense(lstm_hidden, batch_input_shape=(1, gen_length, numchars))
        tohidden_copy.build((1, gen_length, numchars))
        tohidden_copy.set_weights(tohidden.get_weights())

        generator_model = Sequential([
            tohidden_copy,
            stateful_lstm,
            fromhidden,
            Softmax()
        ])

        # show samples for some sentences from random batches
        for i in range(CHECK):
            b = random.choice(x)

            if b.shape[1] > 20:
                seed = b[0,:20]
            else:
                seed = b[0, :]

            seed = np.insert(seed, 0, 1)
            gen = generate_seq(generator_model, stateful_lstm, seed, numchars, 120)

            print('seed   ', decode(seed))
            print('out    ', decode(gen))

            print()

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="embedding_size",
                        help="Size of the word embeddings on the input layer.",
                        default=300, type=int)

    parser.add_argument("-o", "--output-every",
                        dest="out_every",
                        help="Output every n epochs.",
                        default=1, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="Batch size",
                        default=32, type=int)

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task",
                        default='imdb', type=str)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-L", "--lstm-hidden-size",
                        dest="lstm_capacity",
                        help="LSTM capacity",
                        default=256, type=int)

    parser.add_argument("-m", "--max_length",
                        dest="max_length",
                        help="Max length",
                        default=None, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)