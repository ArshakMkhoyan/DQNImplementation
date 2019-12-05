import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class DQNAgent:
    def __init__(self, scope, input_shape, output_shape, model_path=None, eps=0.):
        with tf.variable_scope(scope):
            if model_path is None or not os.path.exists(model_path):
                print('Creating a model')
                self.model = keras.models.Sequential(
                    [keras.layers.Conv2D(32, 8, strides=4, input_shape=input_shape, activation='relu',
                                         kernel_initializer=tf.variance_scaling_initializer(scale=2)),
                     keras.layers.Conv2D(64, 4, strides=2, activation='relu',
                                         kernel_initializer=tf.variance_scaling_initializer(scale=2)),
                     keras.layers.Conv2D(64, 3, strides=1, activation='relu',
                                         kernel_initializer=tf.variance_scaling_initializer(scale=2)),
                     keras.layers.Flatten(),
                     keras.layers.Dense(512, activation='relu',
                                        kernel_initializer=tf.variance_scaling_initializer(scale=2)),
                     keras.layers.Dense(output_shape, activation='linear')])
            else:
                print('Loading the model')
                self.model = keras.models.load_model(model_path)
            self.state_ph = tf.placeholder(dtype='float32', shape=(None,) + input_shape)
            self.qvalues = self.get_symb_qvalues(self.state_ph)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        self.eps = eps

    def get_symb_qvalues(self, state):
        """in tf type"""
        qvalues = self.model(state)
        return qvalues

    def get_qvalues(self, state):
        """ in numpy type"""
        sess = tf.get_default_session()
        return sess.run(self.qvalues, {self.state_ph: state})

    def sample_action(self, qvalues):
        """epsilon greedy policy implementation"""
        batch_size, n_actions = qvalues.shape
        chosen_actions = qvalues.argmax(axis=-1)
        random_actions = np.random.choice(n_actions, size=(batch_size,))
        random_mask = np.random.choice([0, 1], size=(batch_size,), p=[self.eps, 1 - self.eps])
        return np.where(random_mask, chosen_actions, random_actions)