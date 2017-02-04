import gym
import numpy as np
import random
import tensorflow as tf


class DQNAgent():
  """
  DQN Agent with a 2-hidden-layer fully-connected q-network that acts epsilon-greedily.
  """

  def __init__(self,
    epsilon=0.5, 
    epsilon_anneal = 0.01,
    end_epsilon=0.1,
    lr=0.5, 
    gamma=0.99, 
    batch_size=64, 
    state_size=4,
    action_size=2,
    mem_size=1e4,
    n_hidden_1=20,
    n_hidden_2=20
    ):
    """
    args
      epsilon           exploration rate
      epsilon_anneal    linear decay rate per call of learn() function (iteration)
      end_epsilon       lowest exploration rate
      lr                learning rate
      gamma             discount factor
      batch_size         training minibatch size per iteration
      state_size        network input size
      action_size       network output size
      mem_size          max memory size
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
    self.n_hidden_1 = n_hidden_1
    self.n_hidden_2 = n_hidden_2

    self.mem = []
    
    self._build_qnet()
    self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    self.sess.run(tf.global_variables_initializer())


  def _build_qnet(self):
    """
    Build q-network
    """
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.device('/cpu:0'):
      # input state
      self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
      # input action to generate output mask
      self.action = tf.placeholder(tf.int32, [None])
      # target_q = tf.add(reward + gamma * max(q(s,a)))
      self.target_q = tf.placeholder(tf.float32, [None])

      # 2 hidden layers
      # network: [state_size] - [n_hidden_1] - [n_hidden_2] - [action_size]

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

      self.q_values = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])

      action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
      # predicted q(s,a)
      q_value_pred = tf.reduce_sum(self.q_values * action_mask, 1)

      self.loss = tf.reduce_mean(tf.square(tf.sub(self.target_q, q_value_pred)))
      self.optimizer = tf.train.AdamOptimizer(self.lr)
      self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


  def get_value(self, s):
    a = self.get_optimal_action(s)
    return self.get_qvalue(s, a)


  def get_qvalue(self, s, a):
    q_values = self.sess.run(self.q_values, feed_dict={self.state_input: [s]})
    return q_values[0][a]


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


  def learn(self, episode, train_steps):
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
        target_weights = self.sess.run(self.weights)

        sampled_idx = np.random.choice(len(self.mem), self.batch_size)
        samples = random.sample(self.mem, self.batch_size)

        # s[2] is next state
        q_values = self.sess.run(self.q_values, feed_dict={self.state_input: [s[2] for s in samples]})
        max_q_values = q_values.max(axis=1)

        # compute target q value
        target_q = np.array([samples[i][3] + self.gamma*max_q_values[i]*(1-samples[i][4]) for i in xrange(len(samples))])
        target_q = target_q.reshape([self.batch_size])

        # minimize the TD-error
        l, _, = self.sess.run([self.loss, self.train_op], feed_dict={
                                                            self.state_input: [s[0] for s in samples],
                                                            self.target_q: target_q,
                                                            self.action: [s[1] for s in samples]
                                                          })

        if self.total_steps % 1000 == 0:
          # print loss
          print l

