# Deep Q-learning Agent
# Following paper: Playing Atari with Deep Reinforcement Learning
#     https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License


import gym
import numpy as np
import random
import tensorflow as tf
import tf_utils


class DQNAgent_CNN():
  """
  DQN Agent with convolutional q-network that acts epsilon-greedily.
  """


  def __init__(self,
    lr=0.5, 
    momentum=0.95,
    gamma=0.99,
    state_size=[84,84,4],
    action_size=6,
    scope="dqn",
    ):
    """
    args
      lr                learning rate
      gamma             discount factor
      state_size        input image size
      action_size       network output size
      scope             variable scope
    """
    self.lr = lr
    self.momentum = momentum
    self.gamma = gamma
    self.state_size = state_size
    self.action_size = action_size
    self.scope = scope
    self._build_qnet()


  def _build_qnet(self):
    """
    Build q-network
    """
    with tf.variable_scope(self.scope):
      # input state
      self.state_input = tf.placeholder(shape=[None]+self.state_size, dtype=tf.uint8)
      # input action to generate output mask
      self.action = tf.placeholder(shape=[None], dtype=tf.int32)
      # target_q = tf.add(reward + gamma * max(q(s,a)))
      self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)

      state = tf.to_float(self.state_input)/255.0
      conv1 = tf_utils.conv2d(state, n_kernel=32, k_sz=[8,8], stride=4)
      conv2 = tf_utils.conv2d(conv1, n_kernel=64, k_sz=[4,4], stride=2)
      conv3 = tf_utils.conv2d(conv2, n_kernel=64, k_sz=[3,3], stride=1)

      # Fully connected layers
      flattened = tf_utils.flatten(conv3)
      fc1 = tf_utils.fc(flattened, n_output=512, activation_fn=tf.nn.relu)

      self.q_values = tf_utils.fc(fc1, self.action_size, activation_fn=None)
      # self.q_values = tf.nn.relu(self.q_values)
      
      action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
      q_value_pred = tf.reduce_sum(self.q_values * action_mask, 1)

      self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_q, q_value_pred)))
      # self.optimizer = tf.train.AdamOptimizer(self.lr)
      self.optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=self.momentum)
      self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())


  def learn(self, batch_steps, sess):
    """
    Deep Q-learing: train qnetwork with the input batch

    Args
      batch_steps:    a batch of sampled namedtuple Step, where Step.cur_step and 
                      Step.next_step are of shape {self.state_size}
      sess:           tf session
    Returns 
      batch loss (-1 if input is empty)
    """
    if len(batch_steps) == 0:
      return -1
    next_state_batch = [s.next_step for s in batch_steps]
    # compute target q value
    q_values = sess.run(self.q_values, feed_dict={self.state_input: next_state_batch})
    max_q_values = q_values.max(axis=1)
    target_q = np.array([s.reward + self.gamma*max_q_values[i]*(1-s.done) for i,s in enumerate(batch_steps)])
    target_q = target_q.reshape([len(batch_steps)])
    # minimize TD-error
    cur_state_batch = [s.cur_step for s in batch_steps]
    actions = [s.action for s in batch_steps]
    l, _, = sess.run([self.loss, self.train_op], feed_dict={self.state_input: cur_state_batch,
                                                            self.target_q: target_q,
                                                            self.action: actions})
    return l

  def get_optimal_action(self, state, sess):
    return self.get_action_e(state, sess, 0)

  def get_action_e(self, state, sess, epsilon):
    """
    Epsilon-greedy action with epsilon

    Args
      state           state of size {self.state_size}
      epsilon         with {epsilon} prob to randomly choose an action, 
                      with (1-epsilon) act according to argmax q-value.
                      1 for completely randomized, 0 for optimal action
    Returns
      an action to take given the state
    """
    if np.random.random() < epsilon:
      # act randomly
      return np.random.randint(0, self.action_size)
    else:
      actions = sess.run(self.q_values, feed_dict={self.state_input: [state]})
      return actions.argmax()


  def get_action_values(self, state, sess):
    actions = sess.run(self.q_values, feed_dict={self.state_input: [state]})
    return actions




