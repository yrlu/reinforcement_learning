import gym
import numpy as np
import random
import tensorflow as tf


class DQNAgent_CNN():
  """
  DQN Agent with 3 convolution layers q-network that acts epsilon-greedily.
  """


  def __init__(self,
    epsilon=0.5, 
    epsilon_anneal = 0.01,
    end_epsilon=0.1,
    lr=0.5, 
    gamma=0.99, 
    batch_size=64, 
    state_size=[84,84],
    action_size=6,
    mem_size=1e4,
    scope="dqn"
    ):
    """
    args
      sess              session
      epsilon           exploration rate
      epsilon_anneal    linear decay rate per call of learn() function (iteration)
      end_epsilon       lowest exploration rate
      lr                learning rate
      gamma             discount factor
      batch_size        training minibatch size per iteration
      state_size        input image size
      action_size       network output size
      mem_size          max memory size
      scope             variable scope
    """
    self.epsilon = epsilon
    self.epsilon_anneal = epsilon_anneal
    self.end_epsilon = end_epsilon
    self.lr = lr
    self.gamma = gamma
    self.batch_size = batch_size
    self.state_size = state_size
    self.action_size = action_size
    self.mem_size = mem_size
    self.total_steps = 0 
    self.mem = []
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

      state = tf.to_float(self.state_input) / 255.0

      # convolutional layers
      conv1 = tf.contrib.layers.conv2d(
          state, 16, 8, 4, activation_fn=tf.nn.relu)
      conv2 = tf.contrib.layers.conv2d(
          conv1, 32, 4, 2, activation_fn=tf.nn.relu)
      # conv3 = tf.contrib.layers.conv2d(
          # conv2, 64, 3, 1, activation_fn=tf.nn.relu)

      # Fully connected layers
      flattened = tf.contrib.layers.flatten(conv2)
      fc1 = tf.contrib.layers.fully_connected(flattened, 256)
      self.q_values = tf.contrib.layers.fully_connected(fc1, self.action_size)

      action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
      q_value_pred = tf.reduce_sum(self.q_values * action_mask, 1)

      self.loss = tf.reduce_mean(tf.square(tf.sub(self.target_q, q_value_pred)))
      self.optimizer = tf.train.AdamOptimizer(self.lr)
      self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())


  def get_value(self, s, sess):
    a = self.get_optimal_action(s, sess)
    return self.get_qvalue(s, a)


  def get_qvalue(self, s, a, sess):
    q_values = sess.run(self.q_values, feed_dict={self.state_input: [s]})
    return q_values[0][a]



  def get_optimal_action(self, state, sess):
    actions = sess.run(self.q_values, feed_dict={self.state_input: [state]})
    return actions.argmax()


  def get_action(self, state, sess):
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
      # return np.random.randint(0, self.action_size)
      return self.get_optimal_action(state, sess)


  def _add_episode(self, episode):
    """
    Store episode to memory and check if it reaches the mem_size. 
    If so, drop 20% of the oldest memory

    args
      episode       a list of (current state, action, next state, reward, done)
    """
    if self.epsilon > self.end_epsilon:
      self.epsilon = self.epsilon - self.epsilon_anneal

    for step in episode:
      self.mem.append(step)

    while len(self.mem) > self.mem_size:
      # If memory reaches limit, then drop 20% of the oldest memory
      self.mem = self.mem[int(len(self.mem)/5):]


  def learn(self, episode, train_steps, sess):
    """
    Deep Q-learing:
      - Store episode to the memory
      - Sample minibatch from transitions (last state, action, next state, reward, done) from memory
      - Train q-network (s->{a}) by the sampled transitions

    args
      episode       a list of (current state, action, next state, reward, done)
      train_steps   number of training steps per calling learn()
    """
    self._add_episode(episode)

    if len(self.mem) > self.batch_size:
      
      for i in xrange(train_steps):
        self.total_steps = self.total_steps + 1

        samples = random.sample(self.mem, self.batch_size)

        # s[2] is next state
        q_values = sess.run(self.q_values, feed_dict={self.state_input: [s[2] for s in samples]})
        max_q_values = q_values.max(axis=1)

        # compute target q value
        target_q = np.array([samples[i][3] + self.gamma*max_q_values[i]*(1-samples[i][4]) for i in xrange(len(samples))])
        target_q = target_q.reshape([self.batch_size])

        # minimize the TD-error
        l, _, = sess.run([self.loss, self.train_op], feed_dict={
                                                            self.state_input: [s[0] for s in samples],
                                                            self.target_q: target_q,
                                                            self.action: [s[1] for s in samples]
                                                          })

        if self.total_steps % 1000 == 0:
          # print loss
          print l
