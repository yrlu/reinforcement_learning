# Deep Deterministic Policy Gradient
#   following paper: Continuous control with deep reinforcement learning
#                   (https://arxiv.org/pdf/1509.02971.pdf)
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import tensorflow as tf
import tf_utils



class ActorNetwork(object):


  def __init__(self, state_size, action_size, lr, n_h1=400, n_h2=300, tau=0.001):
    self.state_size = state_size
    self.action_size = action_size
    self.optimizer = tf.train.AdamOptimizer(lr)
    self.tau = tau

    self.n_h1 = n_h1
    self.n_h2 = n_h2

    self.input_s, self.actor_variables, self.action_values = self._build_network("actor")
    self.input_s_target, self.actor_variables_target, self.action_values_target = self._build_network("actor_target")

    self.action_gradients = tf.placeholder(tf.float32, [None, self.action_size])
    self.actor_gradients = tf.gradients(self.action_values, self.actor_variables, -self.action_gradients)
    self.update_target_op = [self.actor_variables_target[i].assign(tf.mul(self.actor_variables[i], self.tau) + tf.mul(self.actor_variables_target[i], 1 - self.tau)) 
                              for i in range(len(self.actor_variables))]
    self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients, self.actor_variables))


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None, self.state_size])
    with tf.variable_scope(name):
      layer_1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu, 
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      layer_2 = tf_utils.fc(layer_1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      action_values = tf_utils.fc(layer_2, self.action_size, scope="out", activation_fn=tf.nn.tanh,
        initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
    actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, actor_variables, action_values


  def get_action(self, state, sess):
    return sess.run(self.action_values, feed_dict={self.input_s: state})


  def get_action_target(self, state, sess):
    return sess.run(self.action_values_target, feed_dict={self.input_s_target: state})


  def train(self, state, action_gradients, sess):
    sess.run(self.optimize, feed_dict={
        self.input_s: state, 
        self.action_gradients: action_gradients
      })


  def update_target(self, sess):
    sess.run(self.update_target_op)