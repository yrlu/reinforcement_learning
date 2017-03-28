import tensorflow as tf
import tf_utils
import numpy as np


class ActorNetwork(object):


  def __init__(self, state_size, action_size, lr, name, n_h1=400, n_h2=300):
    self.state_size = state_size
    self.action_size = action_size
    self.name = name
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    
    self.optimizer = tf.train.AdamOptimizer(lr)
    self.input_s, self.action, self.advantage, self.model_variables, self.action_prob, self.action_prob_pred = self._build_network(self.name)
    self.l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in self.model_variables])
    self.pg_loss = tf.reduce_mean(-tf.log(self.action_prob_pred) * self.advantage)

    self.loss = self.pg_loss + 0.01 * self.l2_loss
    # for training local model and get local gradients
    self.optimize = self.optimizer.minimize(self.loss)
    self.gradients = tf.gradients(self.loss, self.model_variables)

  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None, self.state_size])
    action = tf.placeholder(tf.int32, [None])
    advantage = tf.placeholder(tf.float32, [None])

    with tf.variable_scope(name):
      layer_1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      layer_2 = tf_utils.fc(layer_1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      action_values = tf_utils.fc(layer_2, self.action_size)
      action_mask = tf.one_hot(action, self.action_size, 1.0, 0.0)
      action_prob = tf.nn.softmax(action_values)
      action_prob_pred = tf.reduce_sum(action_prob * action_mask, 1)

    model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, action, advantage, model_variables, action_prob, action_prob_pred


  def get_action(self, state, sess):
    state = np.reshape(state,[-1, self.state_size])
    pi = sess.run(self.action_prob, feed_dict={self.input_s: state})
    return np.random.choice(range(self.action_size), p=pi[0])


  def apply_gradients(self, gradients, sess):
    sess.run(self.optimizer.apply_gradients(zip(gradients, self.model_variables)))


  def get_trainable_variables(self):
    return self.model_variables