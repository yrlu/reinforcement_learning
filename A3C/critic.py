import tensorflow as tf
import tf_utils
import numpy as np


class CriticNetwork(object):


  def __init__(self, state_size, lr, name, n_h1=400, n_h2=300):
    self.state_size = state_size
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    
    self.optimizer = tf.train.AdamOptimizer(lr)
    self.input_s, self.model_variables, self.output_v, self.target_v = self._build_network(self.name)
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model_variables])
    self.loss = tf.reduce_mean(tf.square(self.target_v - self.output_v)) + 0.01*self.l2_loss
    # for training local model and get local gradients
    self.gradients = tf.gradients(self.output_v, self.model_variables)
    self.optimize = self.optimizer.minimize(self.loss)


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None, self.state_size])
    target_v = tf.placeholder(tf.float32, [None])
    with tf.variable_scope(name):
      layer_1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      layer_2 = tf_utils.fc(layer_1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      output_v = tf_utils.fc(layer_2, 1, scope="out")
    model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, model_variables, output_v, target_v


  def get_value(self, state, sess):
    state = np.reshape(state,[-1, self.state_size])
    return sess.run(self.output_v, feed_dict={self.input_s: state})[0]


  def apply_gradients(self, gradients, sess):
    sess.run(self.optimizer.apply_gradients(zip(gradients, self.model_variables)))


  def get_trainable_variables(self):
    return self.model_variables