#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 10:02:25 2018

@author: alexandreboyker

"""
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from .batch_norm import BatchNorm
from .helper import get_weight, get_bias, reset_graph
from sklearn.metrics import accuracy_score
class LadderNet(object):
    
    def __init__(self, layer_dimensions=[784, 1000, 500, 250, 250, 250, 10], 
                 denoising_penalties = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10],
                 batch_size=100, num_epochs=1, starter_learning_rate=0.02, decay_after=15, noise_std=.3):
        

        self.layer_dimensions = layer_dimensions # list containing the dimensions of each layer
        self.L = len(self.layer_dimensions) - 1  # depth of the network
        self.num_epochs = num_epochs # number of epochs
        self.starter_learning_rate = starter_learning_rate
        
        self.decay_after = decay_after  # epoch after which to begin learning rate decay
        
        self.batch_size = batch_size 
        self.shapes = list(zip(self.layer_dimensions[:-1], self.layer_dimensions[1:])) # shapes of weights matrices connecting each layer
        self.noise_std = noise_std  # standard deviation of Gaussian noise for corrupted encoder
        self.denoising_penalties = denoising_penalties # penalties on denoising cost functions
        

       

    def _get_weights(self):
        
        weights = {'W': [get_weight(s, "W_"+str(i)) for i,s in enumerate(self.shapes)],  # Encoder weights
           # batch normalization parameter to shift the normalized value
           'beta': [get_bias(0.0, self.layer_dimensions[l+1], "beta_"+str(l)) for l in range(self.L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [get_bias(1.0, self.layer_dimensions[l+1], "gamma_"+str(l)) for l in range(self.L)],
           
           'V' : list([get_weight(s[::-1], "V"+str(i)) for i,s in enumerate(self.shapes)])
            } # weights for decoder
        
        return weights


    def _encoder(self, input_x, noise_std=.3):
        
        """
        Encoder, returns:
            
            -- h: last layer tensor, after activation, this is used to compute predicted values (clean encoder)
            
                and supervised cost (corrupted encoder)
                
            -- tensor_dict: dictionary containing pre-activation tensors z, 
            
                post-activation tensors h, sample mean m and sample variance v for each layer
                
        positional arguments:
            
            -- input_x: tensor of size (self.batch_size + self.num_labelled) * self.layer_dimensions[0] 
                                                                                            |
                                                                                            v
                                                                                        input dimension, 784 for MNIST
                                                                                        
            -- noise_std: float32, standard deviation for Gaussian noise at each layer N(0, noise_std)
                
                        noise_std = 0.0 for clean encoder and noise_std > 0 for corrupted encoder. Default = 0.3
        
        
        """
        h = input_x + tf.random_normal(tf.shape(input_x)) * noise_std  # if corrupted encoder, that is noise_std>0, noise is added
        tensor_dict = {}  
        
        tensor_dict['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        tensor_dict['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        tensor_dict['labeled']['z'][0], tensor_dict['unlabeled']['z'][0] = self.split_lu(h)
        
        for l in range(1, self.L+1):

            tensor_dict['labeled']['h'][l-1], tensor_dict['unlabeled']['h'][l-1] = self.split_lu(h)
            z_pre = tf.matmul(h, self.weights['W'][l-1])  # p
            z_pre_l, z_pre_u = self.split_lu(z_pre)  # split labeled and unlabeled examples
    
            batch_mean, batch_var = tf.nn.moments(z_pre_u, axes=[0])
    
            # if training:
            def training_batch_norm():
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is performed separately
                if noise_std > 0:
                    # Corrupted encoder
                    # batch normalization + noise
                    z = self.join(self.batch_normalization(z_pre_l), self.batch_normalization(z_pre_u, batch_mean, batch_var))
                    z += tf.random_normal(tf.shape(z_pre)) * noise_std
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                    z = self.join(self.update_batch_normalization(z_pre_l, l), self.batch_normalization(z_pre_u, batch_mean, batch_var))
                return z
    
            # else:
            def eval_batch_norm():
                # Evaluation batch normalization
                # obtain average mean and variance and use it to normalize the batch
                mean = self.ewma.average(self.running_mean[l-1])
                var = self.ewma.average(self.running_var[l-1])
                z = self.batch_normalization(z_pre, mean, var)
                return z
    
            # perform batch normalization according to value of boolean "training" placeholder:
            z = tf.cond(self.is_training, training_batch_norm, eval_batch_norm)
    
            if l == self.L:
                
                # use softmax activation in output layer
                h = tf.nn.softmax(self.weights['gamma'][l-1] * (z + self.weights["beta"][l-1]))
            
            else:
                
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self.weights["beta"][l-1])
                
            tensor_dict['labeled']['z'][l], tensor_dict['unlabeled']['z'][l] = self.split_lu(z)
            tensor_dict['unlabeled']['m'][l], tensor_dict['unlabeled']['v'][l] = batch_mean, batch_var  # save mean and variance of unlabeled examples for decoding
            
        tensor_dict['labeled']['h'][l], tensor_dict['unlabeled']['h'][l] = self.split_lu(h)
        return h, tensor_dict
    
    
    def _g_gauss(self, z_c, u, size):
        "gaussian denoising function"
        wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')
    
        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')
    
        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10
    
        z_est = (z_c - mu) * v + mu
        return z_est
    
    def _decoder(self, clean, corr, y, y_c):
        
        """
        Decoder, returns a list containing all unsupervised denoising costs, including the denoising penalties
        
        """
        z_hat = {}
        d_cost = []  # to store the denoising cost of all layers
        for l in range(self.L, -1, -1):
            

            z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
            m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
            
            if l == self.L:
                u = self.unlabeled(y_c)
            else:
                    
                u = tf.matmul(z_hat[l+1], self.weights['V'][l])
                
            u = self.batch_normalization(u)
            z_hat[l] = self._g_gauss(z_c, u, self.layer_dimensions[l])
            z_hat_bn = (z_hat[l] - m) / v
            # append the cost of this layer to d_cost
            d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_hat_bn - z), 1)) / self.layer_dimensions[l]) * self.denoising_penalties[l])
        
        return d_cost
    
    def _build_graph(self):
        """ Build computational graph for the ladder network """
        
        input_x = tf.placeholder(tf.float32, shape=(None, self.layer_dimensions[0]), name='input_x')
        input_y = tf.placeholder(tf.float32, name='input_y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.weights = self._get_weights()
        y_corrupted, corrupted_dict = self._encoder(input_x, self.noise_std)
        
        y_clean, clean_dict = self._encoder(input_x, 0.0) 
        
        denoising_cost_list = self._decoder(clean_dict, corrupted_dict, y_clean, y_corrupted)
        
        
        # calculate total unsupervised cost by adding the denoising cost of all layers
        denoising_cost = tf.add_n(denoising_cost_list)
        
        y_corrupted_labeled = self.labeled(y_corrupted)
        
        # supervised cost
        supervised_cost = -tf.reduce_mean(tf.reduce_sum(input_y*tf.log(y_corrupted_labeled), 1))  
        
        # total loss = supervised cost + denoising_cost
        loss = supervised_cost + denoising_cost  
        
        predi = tf.argmax(y_clean, 1, name ='predictions')
        correct_prediction = tf.equal(predi, tf.argmax(input_y, 1))  # no of correct predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)
        
        self.learning_rate = tf.Variable(self.starter_learning_rate, trainable=False)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
        # add the updates of batch normalization statistics to train_step
        updates_batch_norm = tf.group(*self.bn_assigns)
        with tf.control_dependencies([train_step]):
            train_step = tf.group(updates_batch_norm)
            
        return {'input_x':input_x, 'input_y':input_y, 'train_step':train_step, 
                'accuracy':accuracy, 'loss':loss, 'is_training':self.is_training,
                'predi':predi}
        

    def _create_batch_norm(self):
        """creates batch normamization object """
        bn = BatchNorm(self.layer_dimensions)
        self.batch_normalization = bn.batch_normalization
        self.update_batch_normalization = bn.update_batch_normalization
        self.ewma = bn.ewma
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var
        self.bn_assigns = bn.bn_assigns       
        
    def train(self, X_labelled, X_unlabelled, y_supervised, y_unsupervised=None):
        """
        train the ladder network
        
        positional arguments:
        
            -- X_labelled: labeled data, as a 2 dimensional numpy array
        
            -- X_unlabelled: unlabeled data, as a 2 dimensional numpy array
        
            -- y_supervised: labels related to X_labelled, one-hot encoded
            
        keyword argument:
            
            -- y_unsupervised: labels of the same size as X_unlabelled, only for model validation in an 'artificial' setting, where we have labels for the whole dataset
        
        
            
        """
        
        self.num_labeled = X_labelled.shape[0]
        self.num_examples = X_labelled.shape[0] + X_unlabelled.shape[0]
        self.num_iter = (self.num_examples/self.batch_size) * self.num_epochs  # number of loop iterations
        
        # functions to split and join labelled and unlabelled data points
        self.join = lambda l, u: tf.concat([l, u], 0)
        self.labeled = lambda x: tf.slice(x, [0, 0], [self.num_labeled, -1]) if x is not None else x
        self.unlabeled = lambda x: tf.slice(x, [self.num_labeled, 0], [-1, -1]) if x is not None else x
        self.split_lu = lambda x: (self.labeled(x), self.unlabeled(x))
        
        self._create_batch_norm()
        
        dic = self._build_graph()
        

        with tf.Session() as sess:
      
            input_x = dic['input_x']
            input_y =  dic['input_y']
            train_step = dic['train_step']
            accuracy = dic['accuracy']
            loss = dic['loss']
            is_training = dic['is_training']
            predi = dic['predi']
            # these 3 lines stay together!!
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            
            n_unlablled_samples = X_unlabelled.shape[0]
            
            for i in range(self.num_epochs):
                
                for batch_number in range(0, n_unlablled_samples, self.batch_size):
                    
                    batch_X_unlabelled = X_unlabelled[batch_number:batch_number + self.batch_size]
                    
                    batch_X_labelled = X_labelled
                    batch_X = np.concatenate((batch_X_labelled, batch_X_unlabelled ), axis=0)
                    _, l = sess.run([train_step, loss], feed_dict={input_x: batch_X, input_y: y_supervised, is_training: True})
                    print( "{} iterations: {} out of {} epoch: {} out of {}  loss: {} ".format( str(datetime.now()), batch_number, n_unlablled_samples, i+1, self.num_epochs, l ))

                if y_unsupervised is not None:
                    
                    prediction_list = []
                    validation_batch_size = max([self.batch_size, self.num_labeled])
                    for batch_number in range(0, n_unlablled_samples, validation_batch_size):
    
                        batch_X_unlabelled = X_unlabelled[batch_number:batch_number + validation_batch_size]
                        predi_batch = sess.run(predi, feed_dict={input_x: batch_X_unlabelled, is_training: False})
                        prediction_list += list(predi_batch)
                        
                    print("\nAccuracy on validation data: {}".format(accuracy_score(np.argmax(y_unsupervised,1) ,prediction_list)))

                    if not os.path.exists(os.path.join(os.getcwd(), 'saved_model')):
                        os.makedirs(os.path.join(os.getcwd(), 'saved_model'))
                    saver.save(sess, os.path.join("saved_model","model"))
                    print("Model saved" )
    
    def predict(self, X_test, batch_size_test=1000):
        
        reset_graph()
        predictions = []
        
        # the 4 lines stay together
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join("saved_model","model.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(os.getcwd(), 'saved_model')))
        graph = tf.get_default_graph()

        input_x = graph.get_tensor_by_name("input_x:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        y_pred = graph.get_operation_by_name("predictions").outputs[0]
        
        for batch_number in range(0, X_test.shape[0], batch_size_test):
                
                batch_X = X_test[batch_number:batch_number + batch_size_test]
                prediction_batch = sess.run(y_pred, feed_dict={input_x: batch_X,  is_training: False})
                predictions += list(prediction_batch)
                
        
        return predictions
            


        
            
    
