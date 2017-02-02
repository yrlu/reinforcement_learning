import gym
import numpy as np
import random
import tensorflow as tf


LOG_DIR = '/tmp/dqn'


class DQNAgent():
  """
  DQN Agent owns a 2-hidden-layer fully-connected q-network and acts epsilon-greedily.
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
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())


  def _build_qnet(self):
    """
    Build q-network

    returns
      q-network
    """

    self.state_input = tf.placeholder(tf.float32, [None, self.state_size])

    # 2 hidden layers
    # network: [state_size] - n_hidden_1 - n_hidden_2 - [action_size]

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

    self.qnet = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])

    # TBD
    self.reward = tf.placeholder(tf.float32, [None])
    self.action = tf.placeholder(tf.int32, [None])

    # target_q = tf.add(self.reward + self.gamma * tf.reduce_max(self.qnet, 1))
    self.target_q = tf.placeholder(tf.float32, [None])
    action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
    pred_action_value = tf.reduce_sum(self.qnet * action_one_hot, 1)

    self.loss = tf.reduce_mean(tf.square(tf.sub(self.target_q, pred_action_value)))
    self.optimizer = tf.train.AdamOptimizer(self.lr)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
    


  def get_value(self, s):
    a = self.get_optimal_action(s)
    return self.get_qvalue(s, a)


  def get_qvalue(self, s, a):
    q_values = self.sess.run(self.qnet, feed_dict={self.state_input: [s]})
    return q_values[0][a]


  def get_optimal_action(self, state):
    # return np.random.randint(0, self.action_size)
    # st = [state[i] for i in xrange(len(state))]
    actions = self.sess.run(self.qnet, feed_dict={self.state_input: [state]})
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
    If so, randomly drop 20% of the memory

    args
      episode       a list of (current state, action, next state, reward, done)
    """
    if self.epsilon > self.end_epsilon:
      self.epsilon = self.epsilon - self.epsilon_anneal

    for step in episode:
      self.mem.append(step)

    while len(self.mem) > self.mem_size:
      # drop 20% of the memory
      self.mem = self.mem[int(len(self.mem)/5):]
      # np.random.shuffle(self.mem)
      # for i in xrange(int(self.mem_size*0.2)):
        # self.mem.pop()


  def learn(self, episode, train_steps):
    """
    Deep Q-learing:
      - Store episode to the memory
      - Sample minibatch from transitions (last state, action, next state, reward, done) from memory
      - Train q-network (s->{a}) by the sampled transitions

    args
      episode       a list of (current state, action, next state, reward, done)
    """
    self._add_episode(episode)


    if len(self.mem) > self.batch_size:
      

      for i in xrange(train_steps):
        self.total_steps = self.total_steps + 1

        # tf.summary.scalar('loss', self.loss)
        # self.summary = tf.summary.merge_all()
        # self.summary_writer = tf.summary.FileWriter(LOG_DIR, self.sess.graph)

        step_count = []

        target_weights = self.sess.run(self.weights)

        sampled_idx = np.random.choice(len(self.mem), self.batch_size)
        samples = random.sample(self.mem, self.batch_size)

        q_values = self.sess.run(self.qnet, feed_dict={self.state_input: [s[2] for s in samples]})
        
        max_q_values = q_values.max(axis=1)

        target_q = np.array([samples[i][3] + self.gamma*max_q_values[i]*(1-samples[i][4]) for i in xrange(len(samples))])
        # print samples[0], target_q[0]
        target_q = target_q.reshape([self.batch_size])
        # print len(target_q), target_q
        feed_dict = {
          self.state_input: [s[0] for s in samples],
          self.target_q: target_q,
          self.action: [s[1] for s in samples]
        }
        l, _, = self.sess.run([self.loss, self.train_op], 
                                          feed_dict=feed_dict)

        # Write summary for TensorBoard
        if self.total_steps % 1000 == 0:
          print l
          # self.summary_writer.add_summary(summary, self.total_steps)

