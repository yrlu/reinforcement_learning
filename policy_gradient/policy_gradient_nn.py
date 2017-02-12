# Policy Gradient Agent 
#   - policy approximation with fully connected neural network
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


class PolicyGradientNNAgent():

  def __init__(self,
    epsilon=0.5, 
    epsilon_anneal = 0.01,
    end_epsilon=0.1,
    lr=0.5, 
    gamma=0.99, 
    state_size=4,
    action_size=2,
    n_hidden_1=20,
    n_hidden_2=20,
    batch_size=64,
    mem_size=1e4,
    scope="pg"
    ):
    """
    args
      epsilon           exploration rate
      epsilon_anneal    linear decay rate per call of learn() function (iteration)
      end_epsilon       lowest exploration rate
      lr                learning rate
      gamma             discount factor
      state_size        network input size
      action_size       network output size
    """
    self.epsilon = epsilon
    self.epsilon_anneal = epsilon_anneal
    self.end_epsilon = end_epsilon
    self.lr = lr
    self.gamma = gamma
    self.state_size = state_size
    self.action_size = action_size
    self.total_steps = 0
    self.n_hidden_1 = n_hidden_1
    self.n_hidden_2 = n_hidden_2
    self.scope = scope
    self.mem_size = mem_size
    self.mem = []
    self.batch_size = batch_size

    self._build_policy_net()



  def _build_policy_net(self):
    """Build policy network"""
    with tf.variable_scope(self.scope):
      # input state
      self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
      # input action to generate output mask
      self.action = tf.placeholder(tf.int32, [None])
      # G_t
      self.target = tf.placeholder(tf.float32, [None])

      n_hidden_1 = self.n_hidden_1
      n_hidden_2 = self.n_hidden_2

      self.weights = {
        'h1': tf.Variable(tf.random_normal([self.state_size, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, self.action_size]))
      }

      self.biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([self.action_size]))
      }

      layer_1 = tf.add(tf.matmul(self.state_input, self.weights['h1']), self.biases['b1'])
      layer_1 = tf.nn.relu(layer_1)

      layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
      layer_2 = tf.nn.relu(layer_2)

      self.action_values = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])
      action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
      self.action_value_pred = tf.reduce_sum(self.action_values * action_mask, 1)

      # self.loss = tf.reduce_mean(tf.square(tf.sub(self.target_q, q_value_pred)))
      self.loss = -tf.log(self.action_value_pred) * self.target
      self.optimizer = tf.train.RMSPropOptimizer(self.lr, 0.99, 0.0, 1e-6)
      # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
      self.train_op = self.optimizer.minimize(
      self.loss, global_step=tf.contrib.framework.get_global_step())


  def get_action(self, state, sess):
    """
    Epsilon-greedy action
    args
      state           current state      
    returns
      an action to take given the state
    """
    if np.random.random() < self.epsilon:
      return np.random.randint(0, self.action_size)
    else:
      pi = self.get_policy(state, sess)
      return np.random.choice(range(self.action_size), p=pi)


  def get_policy(self, state, sess):
    """returns policy as probability distribution of actions"""
    pi = sess.run(self.action_values, feed_dict={self.state_input: [state]})    
    pi = [np.exp(p) for p in pi[0]]
    z = sum(pi)
    pi = [p/z for p in pi]
    return pi


  def _add_episode(self, episode):
    """
    Store episode to memory and check if it reaches the mem_size. 
    If so, drop 20% of the oldest memory
    args
      episode       a list of (current state, action, next state, reward, done)
    """
    if self.epsilon > self.end_epsilon:
      self.epsilon = self.epsilon - self.epsilon_anneal

    self.mem.append(episode)
    # for t in xrange(len(episode)):
    #   self.total_steps = self.total_steps + 1
    #   target = sum([self.gamma**i * r for i, (s, a, s1, r, d) in enumerate(episode[t:])])
    #   state, action, next_state, reward, done = episode[t]
    #   self.mem.append([state, action, target])

    while len(self.mem) > self.mem_size:
      # If memory reaches limit, then drop 20% of the oldest memory
      self.mem = self.mem[int(len(self.mem)/5):]


  def learn(self, episode, sess, train_steps=100):
    """
    args
      episode       a list of (current state, action, next state, reward)
    """
    # self._add_episode(episode)

    # if len(self.mem) > self.batch_size:
    #   for i in xrange(train_steps):
    #     sampled_idx = np.random.choice(len(self.mem), self.batch_size)
    #     samples = random.sample(self.mem, self.batch_size)

    #     # states = [s for s,a,t in samples]
    #     # actions = [a for s,a,t in samples]
    #     # targets = [t for s,a,t in samples]
    #     states = []
    #     actions = []
    #     targets = []
    #     for e in samples:
    #       for t in xrange(len(e)):
    #         self.total_steps = self.total_steps + 1
    #         target = sum([self.gamma**i * r for i, (s, a, s1, r, d) in enumerate(e[t:])])
    #         state, action, next_state, reward, done = e[t]
    #         states.append(state)
    #         actions.append(action)
    #         targets.append(target)

    #     feed_dict = { self.state_input: states, self.target: targets, self.action: actions }
    #     _, loss = sess.run([self.train_op, self.loss], feed_dict)


    states = []
    actions = []
    targets = []
    for t in xrange(len(episode)):
      self.total_steps = self.total_steps + 1
      target = sum([self.gamma**i * r for i, (s, a, s1, r, d) in enumerate(episode[t:])])
      state, action, next_state, reward, done = episode[t]
      # states.append(state)
      # actions.append(action)
      # targets.append(target)
      feed_dict = { self.state_input: [state], self.target: [target], self.action: [action] }
      _, loss = sess.run([self.train_op, self.loss], feed_dict)


