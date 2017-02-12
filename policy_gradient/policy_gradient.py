# Policy Gradient Agent with Softmax Policy
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


class PolicyGradientAgent():

  def __init__(self,
    epsilon=0.5, 
    epsilon_anneal = 0.01,
    end_epsilon=0.1,
    lr=0.5, 
    gamma=0.99, 
    state_size=4,
    action_size=2):
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

    # h(s,a,\theta) = \theta.* \phi(s,a)
    self.theta = np.ones([state_size*3 + state_size**2 + 1])


  def _h(self, state, action):
    """
    h(s,a,theta) is numerical preference of the (s,a) with parameter theta
    h(s,a,theta) = theta.*(s,a)
    """
    return np.dot(self.theta, self._get_features(state, action))


  def get_action(self, state):
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
      pi = self.get_policy(state)
      return np.random.choice(range(self.action_size), p=pi)


  def get_policy(self, state):
    """
    args
      state       current state
    returns
      pi(s,a,theta)   policy as probability distribution of actions
    """
    pi = [np.exp(self._h(state, i)) for i in xrange(self.action_size)]
    z = sum(pi)
    pi = [v/z for v in pi]
    return pi

  @staticmethod
  def quadratic_features(s, a):
    """hand designed features"""
    # n_feat = len(s)*2 + len(s)**2 + 1
    return [f for f in s] + [abs(f) for f in s] + [f**2 for f in s] + [f1*f2 for f1 in s for f2 in s] + [a]



  @staticmethod
  def phi_linear(s, a):
    """
    linear features
    """
    return [f for f in s] + [a]

  def _get_features(self, s, a):
    return PolicyGradientAgent.quadratic_features(s,a)


  def learn(self, episode):
    """
    args
      episode       a list of (current state, action, next state, reward)
    """
    if self.epsilon > self.end_epsilon:
      self.epsilon = self.epsilon - self.epsilon_anneal

    for t in xrange(len(episode)):
      G_t = sum([self.gamma**i * r for i, (s, a, s1, r, d) in enumerate(episode[t:])])

      state, action, next_state, reward, done = episode[t]
      pi = self.get_policy(state);
      sum_b = np.sum([np.multiply(p, self._get_features(state, b)) for b,p in enumerate(pi)])
      grad_log_pi = np.subtract(self._get_features(state, action), sum_b)
      self.theta = np.subtract(self.theta, np.multiply(self.lr, grad_log_pi))



