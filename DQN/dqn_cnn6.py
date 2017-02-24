# Deep Q-learning agent with q-value approximation
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
    session,
    epsilon=0.5, 
    epsilon_anneal = 0.01,
    end_epsilon=0.1,
    lr=0.5,
    momentum=0.95,
    gamma=0.99,
    state_size=[84,84,4],
    action_size=2,
    scope="dqn",
    ):
    """
    args
      epsilon           exploration rate
      epsilon_anneal    linear decay rate per call of learn() function (iteration)
      end_epsilon       lowest exploration rate
      lr                learning rate
      momentum          momentum
      gamma             discount factor
      state_size        network input size
      action_size       network output size
    """
    self.epsilon = epsilon
    self.epsilon_anneal = epsilon_anneal
    self.end_epsilon = end_epsilon
    self.lr = lr
    self.momentum = momentum
    self.gamma = gamma
    self.state_size = state_size
    self.action_size = action_size
    self.scope = scope
    self.sess = session
    self._build_qnet()

  def _build_qnet(self):
    """
    Build q-network
    """
    with tf.variable_scope(self.scope):
      self.state_input = tf.placeholder(shape=[None]+self.state_size, dtype=tf.uint8)
      self.action = tf.placeholder(shape=[None], dtype=tf.int32)
      self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)

      state = tf.reshape(tf.to_float(self.state_input) / 255.0, [-1, self.state_size[0], self.state_size[1], self.state_size[2]])

      conv1 = tf_utils.conv2d(state, n_kernel=16, k_sz=[8,8], stride=4)
      conv2 = tf_utils.conv2d(conv1, n_kernel=32, k_sz=[4,4], stride=2)
      # conv3 = tf_utils.conv2d(conv2, n_kernel=64, k_sz=[3,3], stride=1)

      # Fully connected layers
      flattened = tf_utils.flatten(conv2)
      fc1 = tf_utils.fc(flattened, n_output=256, activation_fn=tf.nn.relu)
      self.q_values = tf_utils.fc(fc1, self.action_size, activation_fn=None)

      action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
      q_value_pred = tf.reduce_sum(self.q_values * action_mask, 1)

      self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_q, q_value_pred)))
      self.optimizer = tf.train.AdamOptimizer(self.lr)
      # self.optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=self.momentum)
      self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

  def get_action_values(self, state):
    actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
    return actions

  def get_optimal_action(self, state):
    actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
    return actions.argmax()

  def get_action(self, state):
    """
    Epsilon-greedy action

    args
      state           current state      
    returns
      an action to take given the state
    """
    if np.random.random() < self.epsilon:
      # act randomly
      return np.random.randint(0, self.action_size)
    else:
      return self.get_optimal_action(state)

  def epsilon_decay(self):    
    if self.epsilon > self.end_epsilon:
      self.epsilon = self.epsilon - self.epsilon_anneal

  def learn_epoch(self, exprep, num_steps):
    """
    Deep Q-learing: train qnetwork for num_steps, for each step, sample a batch from exprep

    Args
      exprep:         experience replay
      num_steps:      num of steps
    """
    for i in xrange(num_steps):
      self.learn_batch(exprep.sample())

  def learn_batch(self, batch_steps):
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
    q_values = self.sess.run(self.q_values, feed_dict={self.state_input: next_state_batch})

    max_q_values = q_values.max(axis=1)
    # compute target q value
    target_q = np.array([s.reward + self.gamma*max_q_values[i]*(1-s.done) for i,s in enumerate(batch_steps)])
    target_q = target_q.reshape([len(batch_steps)])
    
    # minimize the TD-error
    cur_state_batch = [s.cur_step for s in batch_steps]
    actions = [s.action for s in batch_steps]
    l, _, = self.sess.run([self.loss, self.train_op], feed_dict={ self.state_input: cur_state_batch,
                                                                  self.target_q: target_q,
                                                                  self.action: actions })
    return l

