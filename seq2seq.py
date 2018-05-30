import keras

import keras.backend as K
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import \
    Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, Input, Reshape, LSTM, Embedding, RepeatVector,\
    TimeDistributed, Bidirectional
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

from keras.utils import multi_gpu_model

import tensorflow as tf

from sklearn import datasets

import math, sys, os
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, MaxPooling2D, UpSampling2D, Flatten, Cropping2D
from keras.models import Model, Sequential
from keras.engine.topology import Layer

import util

INDEX_FROM = 3
CHECK = 5

class KLLayer(Layer):

    """
    Identity transform layer that adds KL divergence
    to the final model loss.

    http://tiao.io/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

class Sample(Layer):

    """
    Performs sampling step

    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        eps = Input(tensor=K.random_normal(shape=K.shape(mu) ))

        z = K.exp(.5 * log_var) * eps + mu

        return z

    def compute_output_shape(self, input_shape):
        shape_mu, _ = input_shape
        return shape_mu


def decode_imdb(seq):

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
    slength = options.max_length
    embedding_length = 32
    top_words = options.top_words
    lstm_hidden = options.lstm_capacity

    print('devices', device_lib.list_local_devices())

    if options.task == 'europarl':

        dir = options.data_dir
        x, x_vocab_len, x_word_to_ix, x_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = \
            util.load_data(dir+os.sep+'europarl-v8.fi-en.en', dir+os.sep+'europarl-v8.fi-en.fi', top_words)

        # Finding the length of the longest sequence
        x_max_len = max([len(sentence) for sentence in x])
        y_max_len = max([len(sentence) for sentence in y])

        # Padding zeros to make all sequences have a same length with the longest one
        X = sequence.pad_sequences(x, maxlen=x_max_len, dtype='int32')
        y = sequence.pad_sequences(y, maxlen=y_max_len, dtype='int32')

        def decode(seq):
            return ' '.join(x_ix_to_word[id] for id in seq)

    else:
        # Load only training sequences
        (x, _), _ = imdb.load_data(num_words=top_words)

        x = sequence.pad_sequences(x, maxlen=slength, padding='pre', truncating='post')

        decode = decode_imdb

    print('Data Loaded. Size ', x.shape)

    input = Input(shape=(slength, ))
    h = Embedding(top_words, embedding_length, input_length=slength)(input)
    print(h)
    h = Bidirectional(LSTM(lstm_hidden))(h)
    zmean = Dense(options.hidden)(h)
    zlsigma = Dense(options.hidden)(h)

    encoder = Model(input, (zmean, zlsigma))

    decoder = Sequential()
    decoder.add(RepeatVector(slength, input_shape=(options.hidden,)))
    decoder.add(LSTM(lstm_hidden, return_sequences=True))
    decoder.add(TimeDistributed(Dense(top_words)))

    decoder.summary()

    input = Input(shape=(slength, ))
    h = encoder(input)
    h = KLLayer()(h) # computes the KL loss and stores it for later
    h = Sample()(h)  # implements the reparam trick
    out = decoder(h)

    auto = Model(input, out)

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

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task",
                        default='imdb', type=str)


    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-H", "--hidden-size",
                        dest="hidden",
                        help="Latent vector size",
                        default=64, type=int)

    parser.add_argument("-L", "--lstm-hidden-size",
                        dest="lstm_capacity",
                        help="LSTM capacity",
                        default=256, type=int)


    parser.add_argument("-g", "--num-gpu",
                        dest="num_gpu",
                        help="How many GPUs to use",
                        default=None, type=int)

    parser.add_argument("-m", "--max_length",
                        dest="max_length",
                        help="Max length",
                        default=500, type=int)

    parser.add_argument("-w", "--top_words",
                        dest="top_words",
                        help="Top words",
                        default=10000, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)