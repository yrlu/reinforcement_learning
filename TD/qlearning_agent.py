# Q-learning Agent
# Model-free Temporal Difference learning
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import agent
import numpy

class QLearningAgent(agent.RLAgent):


  def __init__(self, legal_actions_fn, epsilon=0.5, alpha=0.5, gamma=0.9, epsilon_decay=1):
    """
    args
      legal_actions_fn    takes a state and returns a list of legal actions
      alpha       learning rate
      epsilon     exploration rate
      gamma       discount factor
    """
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon_decay=epsilon_decay
    self.legal_actions_fn = legal_actions_fn

    # map: {(state, action): q-value}
    self.q_values = {}
    # map: {state: action}
    self.policy = {}
    

  def get_value(self, s):
    a = self.get_optimal_action(s)
    return self.get_qvalue(s, a)


  def get_qvalue(self, s, a):
    if (s,a) in self.q_values:
      return self.q_values[(s,a)]
    else:
      # set to 0
      self.q_values[(s,a)] = 0
      return 0

  def _set_qvalue(self, s, a, v):
    self.q_values[(s,a)] = v


  def get_optimal_action(self, state):
    legal_actions = self.legal_actions_fn(state)
    assert len(legal_actions) > 0, "no legal actions"
    if state in self.policy:
      return self.policy[state]
    else:
      # randomly select an action as default and return
      self.policy[state] = legal_actions[numpy.random.randint(0, len(legal_actions))]
      return self.policy[state]

  def get_action(self, state):
    """
    Epsilon-greedy action
    args
      state           current state      
    returns
      an action to take given the state
    """
    legal_actions = self.legal_actions_fn(state)

    assert len(legal_actions) > 0, "no legal actions on state {}".format(state)

    if numpy.random.random() < self.epsilon:
      # act randomly
      return legal_actions[numpy.random.randint(0, len(legal_actions))]
    else:
      if state in self.policy:
        return self.policy[state]
      else:
        # set the first action in the list to default and return
        self.policy[state] = legal_actions[0]
        return legal_actions[0]


  def learn(self, s, a, s1, r, is_done):
    """
    Updates self.q_values[(s,a)] and self.policy[s]
    args
      s         current state
      a         action taken
      s1        next state
      r         reward
      is_done   True if the episode concludes
    """
    # update q value
    if is_done:
      sample = r
    else:
      sample = r + self.gamma*max([self.get_qvalue(s1,a1) for a1 in self.legal_actions_fn(s1)])
    
    q_s_a = self.get_qvalue(s,a)
    q_s_a = q_s_a + self.alpha*(sample - q_s_a)
    self._set_qvalue(s,a,q_s_a)

    # policy improvement
    legal_actions = self.legal_actions_fn(s)
    s_q_values = [self.get_qvalue(s,a) for a in legal_actions]
    self.policy[s] = legal_actions[s_q_values.index(max(s_q_values))]

    self.epsilon = self.epsilon*self.epsilon_decay