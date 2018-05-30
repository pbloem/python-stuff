import keras

import keras.backend as K
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import \
    Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, Input, Reshape, LSTM, Embedding, RepeatVector,\
    TimeDistributed
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

from keras.utils import multi_gpu_model

import tensorflow as tf

from sklearn import datasets

import math, sys
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, MaxPooling2D, UpSampling2D, Flatten, Cropping2D
from keras.models import Model, Sequential


INDEX_FROM = 3
CHECK = 5

def decode(seq):

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}

    return ' '.join(id_to_word[id] for id in seq)

def sparse_loss(y_true, y_pred):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

def go(options):
    max_sequence_length = 500
    embedding_length = 32
    top_words = 10000
    lstm_hidden = 256

    print('devices', device_lib.list_local_devices())

    # Load only training sequences
    (x, _), _ = imdb.load_data(num_words=top_words)

    x = sequence.pad_sequences(x, maxlen=max_sequence_length)

    encoder = Sequential()
    encoder.add(Embedding(top_words, embedding_length, input_length=max_sequence_length))
    encoder.add(LSTM(lstm_hidden))
    encoder.add(Dense(options.hidden))

    encoder.summary()

    decoder = Sequential()
    decoder.add(RepeatVector(max_sequence_length, input_shape=(options.hidden,)))
    decoder.add(LSTM(lstm_hidden, return_sequences=True))
    decoder.add(TimeDistributed(Dense(top_words)))

    decoder.summary()

    auto = Sequential()
    auto.add(encoder)
    auto.add(decoder)

    auto.summary()

    if options.num_gpu is not None:
        auto = multi_gpu_model(auto, gpus=options.num_gpu)

    opt = keras.optimizers.RMSprop(lr=options.lr)

    auto.compile(opt, keras.losses.sparse_categorical_crossentropy)

    epochs = 0
    while epochs < options.epochs:
        auto.fit(x, x[:, :, None],
                epochs=options.out_every,
                batch_size=options.batch,
                validation_split=1/6,
                shuffle=True)
        epochs += options.out_every

        sub = x[:CHECK, :]
        out = auto.predict(sub)
        y = np.argmax(out, axis=-1)

        for i in range(CHECK):
            print('in   ',  decode(x[i, :]))
            print('out   ', decode(y[i, :]))
            print()

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

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

    parser.add_argument("-H", "--hidden-size",
                        dest="hidden",
                        help="Latent vector size",
                        default=64, type=int)

    parser.add_argument("-g", "--num-gpu",
                        dest="num_gpu",
                        help="How many GPUs to use",
                        default=None, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)