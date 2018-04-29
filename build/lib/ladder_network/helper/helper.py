#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:25:15 2018

@author: alexandreboyker
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data

def print_tf_variables(sess):
    
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)

    for var, val in zip(tvars, tvars_vals):
                
        print(var.name, val.shape , '\n')
        
def get_bias(inits, size, name):
    initializer = inits * tf.ones([size])
    return tf.get_variable(name, initializer=initializer)


def get_weight(shape, name):
    initializer = tf.random_normal(shape)
    return tf.get_variable(name, initializer=initializer ) / np.sqrt(shape[0])

def reset_graph():
    
    if 'sess' in globals() and sess:
        sess.close()
        
    tf.reset_default_graph()
    
    

def get_MNIST_data(n_supervised, random_state=None):
    """ returns MNIST data + labels, artificially split between 'supervised' data points and 'unsupervised' data points. Labels of 'unsupervised' data points are used for model validation """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    data = np.concatenate([mnist.train.images, mnist.test.images])
    labels = np.concatenate([mnist.train.labels, mnist.test.labels])
    X_supervised, X_unsupervised, y_supervised, y_unsupervised = train_test_split(data, labels, train_size=n_supervised, random_state=random_state)
    return X_supervised, X_unsupervised, y_supervised, y_unsupervised

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(np.array(y_test), np.array(y_pred))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      
    print("accuracy: {}".format(accuracy_score(y_test, y_pred)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
