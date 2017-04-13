import tensorflow as tf
import tf_utils
import numpy as np


class AC_Net(object):


  def __init__(self, state_size, action_size, lr, 
              name, n_h1=400, n_h2=300, global_name='global'):

    self.state_size = state_size
    self.action_size = action_size
    self.name = name
    self.n_h1 = n_h1
    self.n_h2 = n_h2

    self.optimizer = tf.train.AdamOptimizer(lr)
    self.input_s, self.input_a, self.advantage, self.target_v, self.policy, self.value, self.action_est, self.model_variables = self._build_network(name)

    self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
    self.entropy_loss = tf.reduce_sum(self.policy * tf.log(self.policy))
    self.policy_loss = tf.reduce_sum(-tf.log(self.action_est) * self.advantage)
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model_variables]) 
    # self.loss = 0.5 * self.value_loss + self.policy_loss + 0.01 * self.entropy_loss
    # self.loss = 0.5 * self.value_loss + self.policy_loss + 0.01 * self.entropy_loss + 0.02*self.l2_loss
    self.loss = 0.5 * self.value_loss + self.policy_loss + 0.1 * self.entropy_loss
    # self.loss = self.value_loss + 0.5 * self.policy_loss + 0.01 * self.entropy_loss
    # self.loss = 0.5 * self.value_loss + self.policy_loss
    self.gradients = tf.gradients(self.loss, self.model_variables)
    if name != global_name:
      self.var_norms = tf.global_norm(self.model_variables)
      # self.grads_global, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
      global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, global_name)
      # self.apply_gradients = self.optimizer.apply_gradients(zip(self.grads_global, global_variables))
      self.apply_gradients = self.optimizer.apply_gradients(zip(self.gradients, global_variables))


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None, self.state_size])
    input_a = tf.placeholder(tf.int32, [None])
    advantage = tf.placeholder(tf.float32, [None])
    target_v = tf.placeholder(tf.float32, [None])

    with tf.variable_scope(name):
      layer_1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      layer_2 = tf_utils.fc(layer_1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      policy = tf_utils.fc(layer_2, self.action_size, activation_fn=tf.nn.softmax,
                              scope="policy", initializer=tf_utils.normalized_columns_initializer(0.01))
      value = tf_utils.fc(layer_2, 1, activation_fn=None,
                              scope="value", initializer=tf_utils.normalized_columns_initializer(1.0))

      action_mask = tf.one_hot(input_a, self.action_size, 1.0, 0.0)
      action_est = tf.reduce_sum(policy * action_mask, 1)

    model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, input_a, advantage, target_v, policy, value, action_est, model_variables


  def get_action(self, state, sess):
    state = np.reshape(state,[-1, self.state_size])
    pi = sess.run(self.policy, feed_dict={self.input_s: state})
    return np.random.choice(range(self.action_size), p=pi[0])


  def predict_policy(self, state, sess):
    state = np.reshape(state,[-1, self.state_size])
    pi = sess.run(self.policy, feed_dict={self.input_s: state})
    return pi[0]


  def predict_value(self, state, sess):
    state = np.reshape(state,[-1, self.state_size])
    return sess.run(self.value, feed_dict={self.input_s: state})


