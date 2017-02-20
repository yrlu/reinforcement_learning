# Monte Carlo Agent
# Epsilon-greedy monte carlo agent
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import sys
if "../" not in sys.path:
  sys.path.append("../")
from TD import qlearning
import numpy

class Counter:
  """
  Counter class 
  """

  def __init__(self):
    self.counter = {}

  def add(self, key):
    if key in self.counter:
      self.counter[key] = self.counter[key] + 1
    else:
      self.counter[key] = 1

  def get(self, key):
    if key in self.counter:
      return self.counter[key]
    else:
      return 0


class MonteCarloAgent(qlearning.QLearningAgent):

  def __init__(self, legal_actions_fn, epsilon=0.5, alpha=0.5, gamma=0.9, epsilon_decay=1):
    self.n_s_a = Counter()
    super(MonteCarloAgent, self).__init__(legal_actions_fn, epsilon, alpha, gamma, epsilon_decay)


  @staticmethod
  def compute_G_t(rewards, gamma):
    """
    args
      a list of rewards
    returns 
      a list of cummulated rewards G_t = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + .. + gamma^{T-t-1}*R_{T}
    """
    G_t = [0]*len(rewards)

    for i in xrange(0,len(rewards)):
      G_t[0] = G_t[0] + rewards[i]*(gamma**i)

    for i in xrange(1,len(rewards)):
      G_t[i] = (G_t[i-1] - rewards[i-1])/gamma

    return G_t


  def learn(self, episode):
    """
    args
      episode       a list of (current state, action, next state, reward)
    """
    q_values = self.q_values.copy()
    
    rewards = [r for c, a, n, r in episode]
    G_t = MonteCarloAgent.compute_G_t(rewards, self.gamma)
    for i in xrange(len(episode)):
      c, a, n, r = episode[i]
      # q-state count++
      self.n_s_a.add((c,a))
      # update q-value 
      # notices here I took the max of the weights and self.alpha to ensure it actually 
      # learns some thing from each episode of experience
      q_values[(c,a)] = self.get_qvalue(c,a) + max(1/self.n_s_a.get((c,a)), self.alpha) * (G_t[i] - self.get_qvalue(c,a))

    self.q_values = q_values

    # policy improvement
    policy = self.policy.copy()
    for c, a, n, r in episode:
      legal_actions = self.legal_actions_fn(c)
      s_q_values = [self.get_qvalue(c,a) for a in legal_actions]
      policy[c] = legal_actions[s_q_values.index(max(s_q_values))]  
    self.policy = policy

    self.epsilon = self.epsilon*self.epsilon_decay

