# Policy iteration agent
# Model-based learning which requires mdp.
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import agent
import math


class PolicyIterationAgent(agent.Agent):

  def __init__(self, mdp, gamma, iterations=100):
    """
    The constructor performs policy iteration on mdp using dynamic programming
    ---
    args
      mdp:      markov decision process that is required by value iteration agent
      gamma:    discount factor
    """
    self.mdp = mdp
    self.gamma = gamma
    states = mdp.get_states()
    # init values
    self.values = {}
    # policy is a map from state to action
    self.policy = {}

    for s in states:
      if mdp.is_terminal(s):
        self.values[s] = mdp.get_reward(s)
      else:
        self.values[s] = 0
      self.policy[s] = 0

    # useful functions of the mdp:
    #   mdp.get_states()                                      {s}
    #   mdp.get_actions(state)                                {a}
    #   mdp.get_reward(state)                                 R(s, a)
    #   mdp.get_transition_states_and_probs(state, action)    {P(s'|s, a)}
    #   mdp.is_terminal(state)                                True/False

    # estimate values
    for i in range(iterations):
      values_tmp = self.values.copy()
      policy_tmp = self.policy.copy()


      for s in states:
        # policy iteration
        if mdp.is_terminal(s):
          continue

        self.values[s] = sum([P_s1_s_a * (self.mdp.get_reward(s) + self.gamma*values_tmp[s1]) for s1,P_s1_s_a in self.mdp.get_transition_states_and_probs(s, policy_tmp[s])])

        # policy improvement
        actions = [a_s[0] for a_s in mdp.get_actions(s)]
        # print actions
        v_a = [sum([P_s1_s_a * (self.mdp.get_reward(s) + self.gamma*values_tmp[s1]) 
                for s1, P_s1_s_a in self.mdp.get_transition_states_and_probs(s, a)]) for a in actions]
        self.policy[s] = actions[v_a.index(max(v_a))]
        # print s,self.policy[s], v_a.index(max(v_a)), max(v_a), actions[v_a.index(max(v_a))]

  def get_values(self):
    """
    returns
      a dictionary {<state, value>}
    """
    return self.values

  def get_optimal_policy(self):
    """
    returns
      a dictionary {<state, action>}
    """
    states = self.mdp.get_states()
    policy = {}
    for s in states:
      policy[s] = [(self.get_action(s),1)]
    return policy

  def get_action(self, state):
    """
    args
      state    current state
    returns
      an action to take given the state
    """
    return self.policy[state]