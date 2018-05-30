# Simple, quick test of truncated backpropagation through time, with pure tensorflow
#
# Keras can do TBPTT, but only in a weird combination with batching, so it doesn't seem like it'll work in a full batch
# setting.
#
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/keras lst
#

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import matplotlib.pyplot as plt

import tensorflow as tf

STEPS = 1000
BATCH = 10
HIDDEN = 128

numpy.random.seed(7)

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

lengths = []
for x in X_train:
    lengths.append(len(x))

num_units = 200
num_layers = 1
dropout = tf.placeholder(tf.float32)

lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN)

# Initial state of the LSTM memory.
initial_state = state = tf.zeros([BATCH, lstm.state_size])

for i in range(1000):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state