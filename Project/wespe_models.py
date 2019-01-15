import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, GlobalAveragePooling2D
from keras.models import Model, load_model, Sequential
from tensorflow import set_random_seed
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import tensorflow as tf

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import *
from vgg import *


def resblock(in, num):
    model = Sequential()
    model.add(Conv2D(in, 64, 3, padding='SAME', strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer()))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, padding='SAME', strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer()))
    model.add(Activation('relu'))
    return model + in


def discriminator_network(img, var_scope='discriminator', preprocess='gray'):
    model = Sequential()
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):

        model = Sequential()
        # conv layer 1
        model.add(Conv2D(img, 48, 11, strides=4, padding='SAME', name='CONV_1', kernel_initializer=tf.contrib.layers.xavier_initializer()))
        model.add(LeakyReLU(alpha=.001))

        # conv layer 2
        model.add(Conv2D(128, 5, strides=2, padding='SAME', name='CONV_2', kernel_initializer=tf.contrib.layers.xavier_initializer()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))

        # conv layer 3
        model.add(Conv2D(192, 3, strides=1, padding='SAME', name='CONV_3', kernel_initializer=tf.contrib.layers.xavier_initializer()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))

        # conv layer 4
        model.add(Conv2D(192, 3, strides=1, padding='SAME', name='CONV_4', kernel_initializer=tf.contrib.layers.xavier_initializer()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))

        # conv layer 5
        model.add(Conv2D(128, 3, strides=2, padding='SAME', name='CONV_5', kernel_initializer=tf.contrib.layers.xavier_initializer()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))

        # FC layer 1
        model.add(Flatten())
        model.add(Dense(units=1024))
        model.add(LeakyReLU(alpha=.001))

        # FC layer 2
        logits = tf.layers.dense(model, units=1, activation=None)
        probability = tf.nn.sigmoid(logits)
    return logits, probability


def generator_network(image, var_scope='generator'):
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):

        # conv. layer before residual blocks
        b1_in = tf.layers.conv2d(image, 64, 9, strides=1, padding='SAME', name='CONV_1', kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        b1_in = tf.nn.relu(b1_in)

        # residual blocks
        b1_out = resblock(b1_in, 1)
        b2_out = resblock(b1_out, 2)
        b3_out = resblock(b2_out, 3)
        b4_out = resblock(b3_out, 4)

        # conv. layers after residual blocks
        temp = tf.layers.conv2d(b4_out, 64, 3, strides=1, padding='SAME', name='CONV_2', kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.nn.relu(temp)
        temp = tf.layers.conv2d(temp, 64, 3, strides=1, padding='SAME', name='CONV_3', kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.nn.relu(temp)
        temp = tf.layers.conv2d(temp, 64, 3, strides=1, padding='SAME', name='CONV_4', kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.nn.relu(temp)
        temp = tf.layers.conv2d(temp, 3, 1, strides=1, padding='SAME', name='CONV_5', kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        return temp
