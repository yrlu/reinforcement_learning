# Value iteration agent
# Model-based learning which requires mdp. 
# 
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
# 
# MIT License

import agent

class ValueIterationAgent(agent.Agent):

  def __init__(self, mdp, gamma, iterations = 100):
    """
    The constructor build a value model from mdp using dynamic programming
    ---
    args
      mdp:      markov decision process that is required by value iteration agent
      gamma:    discount factor
    """
    self.mdp = mdp
    self.gamma = gamma
    self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
    states = mdp.get_states()
    # init values
    self.values = {}

    for s in states:
      if mdp.is_terminal(s):
        self.values[s] = mdp.get_reward(s)
      else:
        self.values[s] = 0

    # useful functions of the mdp:
    #   mdp.get_states()                                      {s} 
    #   mdp.get_actions(state)                                {a}
    #   mdp.get_reward(state)                                 R(s, a)
    #   mdp.get_transition_states_and_probs(state, action)    {P(s'|s, a)}
    #   mdp.is_terminal(state)                                True/False

    # estimate values
    for i in range(iterations):
      values_tmp = self.values.copy()

      for s in states:
        if mdp.is_terminal(s):
          continue

        actions = [a_s[0] for a_s in mdp.get_actions(s)]
        v_s = []
        for a in actions:
          P_s1sa = mdp.get_transition_states_and_probs(s, a)
          R_sas1 = [mdp.get_reward(s1) for s1 in [p[0] for p in P_s1sa]]
          v_s.append(sum([P_s1sa[s1_id][1]*(mdp.get_reward(s) + gamma*values_tmp[P_s1sa[s1_id][0]]) for s1_id in range(len(P_s1sa))]))
        # V(s) = max_{a} \sum_{s'} P(s'| s, a) (R(s,a,s') + \gamma V(s'))
        self.values[s] = max(v_s)


  def get_values(self):
    """
    returns
      a dictionary {<state, value>}
    """
    return self.values

  


  def get_action(self, state):
    """
      args
        state    current state
      returns
        an action to take given the state
    """
    actions = [a_s[0] for a_s in self.mdp.get_actions(state)]
    v_s = []
    for a in actions:
      P_s1sa = self.mdp.get_transition_states_and_probs(state, a)
      R_sas1 = [self.mdp.get_reward(s1) for s1 in [p[0] for p in P_s1sa]]
      v_s.append(sum([P_s1sa[s1_id][1]*(self.mdp.get_reward(state) + self.gamma*self.values[P_s1sa[s1_id][0]]) for s1_id in range(len(P_s1sa))]))
    a_id = v_s.index(max(v_s))
    return actions[a_id]

import unittest
import gridworld

class ValueIterationAgentTest(unittest.TestCase):
  """
  Unit test for value iteration agent
  """
  def setUp(self):
    grid = [['0','0','0','1'],
            ['0','x','0','-1'],
            ['0','0','0','0']]

    self.grid = grid
    self.gw_deterministic = gridworld.GridWorld(grid, {(0,3),(1,3)}, 1)
    self.gw_non_deterministic = gridworld.GridWorld(grid, {(0,3),(1,3)}, 0.8)

    self.agent = ValueIterationAgent(self.gw_non_deterministic, 0.9, 100)
    self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}

  def test_values(self):
    print 
    self.gw_non_deterministic.display_value_grid(self.agent)

  def test_actions(self):
    print 
    self.gw_non_deterministic.display_policy_grid(self.agent)
    # for i in range(len(self.grid)):
      # print [self.dirs[self.agent.get_action((i,j))] for j in range(len(self.grid[0]))]

if __name__ == '__main__':
  unittest.main()
