#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:02:44 2018

@author: alexandreboyker
"""
import tensorflow as tf

class BatchNorm(object):
    
    def __init__(self, layer_dimensions=None):
        if layer_dimensions is not None:
            
            self.layer_dimensions = layer_dimensions
            # average mean and variance of all layers
  
            self.running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in self.layer_dimensions[1:]]
            self.running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in self.layer_dimensions[1:]]
            self.ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
            self.bn_assigns = []  # this list stores the updates to be made to average mean and variance
            
        
        
    
    def batch_normalization(self, batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
    
    
    
    def update_batch_normalization(self, batch, l):
        "batch normalize + update average mean and variance of layer l"
        mean, var = tf.nn.moments(batch, axes=[0])
        assign_mean = self.running_mean[l-1].assign(mean)
        assign_var = self.running_var[l-1].assign(var)
        self.bn_assigns.append(self.ewma.apply([self.running_mean[l-1], self.running_var[l-1]]))
        with tf.control_dependencies([assign_mean, assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)
