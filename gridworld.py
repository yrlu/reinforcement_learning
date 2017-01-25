# Gridworld environment based on mdp.py
# Gridworld provides a basic environment for RL agents to interact with
# 
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
# 
# MIT License

import mdp

class GridWorld(mdp.MDP):
  """
  Grid world environment
  """

  def __init__(self, grid, terminals, trans_prob=1):
    """
    input:
      grid        2-d list of the grid including the reward
      terminals   a set of all the terminal states
      trans_prob  transition probability when given a certain action
    """
    self.height = len(grid)
    self.width = len(grid[0])
    self.terminals = terminals
    self.grid = grid
    self.neighbors = [(0,1),(0,-1),(1,0),(-1,0),(0,0)]
    self.actions =   [  0,    1,    2,    3,    4]
    #                   right,    left,     down,     up
    # self.action_nei = {0: (0,1), 1:(0,-1), 2:(1,0), 3:(-1,0)}

    # If the mdp is deterministic, the transition probability of taken a certain action should be 1
    # otherwise < 1, the rest of the probability are equally spreaded onto other neighboring states.
    self.trans_prob = trans_prob

  def show_grid(self):
    for i in range(len(self.grid)):
      print self.grid[i]


  def get_grid(self):
    return self.grid

  def get_states(self):
    """
    returns
      a list of all states
    """
    return filter(lambda x: self.grid[x[0]][x[1]] != 'x', [(i,j) for i in range(self.height) for j in range(self.width)])

  def get_actions(self, state):
    """
    get all the actions that can be takens on the current state
    returns
      a list of (action, state) pairs
    """
    res = []
    for i in range(len(self.actions)):
      inc = self.neighbors[i]
      a = self.actions[i]
      nei_s = (state[0]+inc[0], state[1]+inc[1])
      if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[1] >= 0 and nei_s[1] < self.width and self.grid[nei_s[0]][nei_s[1]] != 'x':
        res.append((a, nei_s))
    return res

  def get_reward(self, state):
    """
    returns
      the reward on current state
    """
    if not self.grid[state[0]][state[1]] == 'x':
      return float(self.grid[state[0]][state[1]])
    else:
      return 0

  def get_transition_states_and_probs(self, state, action):
    """
    get all the possible transition states and their probabilities with [action] on [state]
    args
      state     (y, x)
      action    int
    returns
      a list of (state, probability) pair
    """
    if self.trans_prob == 1:
      inc = self.neighbors[action]
      nei_s = (state[0]+inc[0], state[1]+inc[1])
      if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[1] >= 0 and nei_s[1] < self.width:
        return [(nei_s, 1)]
      else:
        return [(state, 1)]
    else:
      action_states = self.get_actions(state)
      inc = self.neighbors[action]
      nei_s = (state[0]+inc[0], state[1]+inc[1])
      res = []
      
      if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[1] >= 0 and nei_s[1] < self.width:
        for i in range(len(action_states)):
          if(action_states[i][0] == action):
            res.append((action_states[i][1], self.trans_prob))
          else:
            res.append((action_states[i][1], (1-self.trans_prob)/(len(action_states)-1)))
      else:
        for i in range(len(action_states)):
          res.append((action_states[i][1], 1/len(action_states)))
      return res

  def is_terminal(self, state):
    """
    returns
      True if the [state] is terminal
    """
    if state in self.terminals:
      return True
    else:
      return False



import unittest

class GridWorldTest(unittest.TestCase):
  """
  Unit test for grid world
  """

  def setUp(self):
    # grid = [['0','0','0','0','0'],['0','0','0','0','0'],['10','10','0','0','0']]
    grid = [['0','0','0','0','10'],
            ['0','x','0','0','-10'],
            ['0','0','0','0','0']]
    # grid = [[0,0,0,0,10],
            # [0,0,0,0,-10],
            # [0,0,0,0,0]]
    self.grid = grid
    self.gw_deterministic = GridWorld(grid, {(0,4),(1,4)}, 1)
    self.gw_non_deterministic = GridWorld(grid, {(0,4),(1,4)}, 0.8)

  def test_grid_dims(self):
    self.assertEqual(len(self.gw_deterministic.get_grid()), 3)
    self.assertEqual(len(self.gw_deterministic.get_grid()[0]), 5)

  def test_grid_values(self):
    grid_tmp = self.gw_deterministic.get_grid()
    for i in range(len(grid_tmp)):
      for j in range(len(grid_tmp[0])):
        self.assertEqual(self.grid[i][j], grid_tmp[i][j])

  def test_get_states(self):
    self.assertEqual(len(self.gw_deterministic.get_states()), 14)

  def test_get_actions(self):
    self.assertEqual(len(self.gw_deterministic.get_actions((0,0))), 3);
    self.assertEqual(len(self.gw_deterministic.get_actions((2,0))), 3);
    self.assertEqual(len(self.gw_deterministic.get_actions((2,4))), 3);
    self.assertEqual(len(self.gw_deterministic.get_actions((0,4))), 3);
    self.assertEqual(len(self.gw_deterministic.get_actions((1,0))), 3);
    
  def test_get_reward(self):
    self.assertEqual(self.gw_deterministic.get_reward((0,0)), 0);
    self.assertEqual(self.gw_deterministic.get_reward((0,4)), 10.0);
    self.assertEqual(self.gw_deterministic.get_reward((1,4)), -10.0);

  def test_trans_prob_deter(self):
    self.assertEqual(len(self.gw_deterministic.get_transition_states_and_probs((0,0),0)), 1)
    self.assertEqual(self.gw_deterministic.get_transition_states_and_probs((0,0),0)[0][0], (0,1))
    self.assertEqual(self.gw_deterministic.get_transition_states_and_probs((0,0),0)[0][1], 1)

    self.assertEqual(len(self.gw_deterministic.get_transition_states_and_probs((0,0),1)), 1)
    self.assertEqual(self.gw_deterministic.get_transition_states_and_probs((0,0),1)[0][0], (0,0))
    self.assertEqual(self.gw_deterministic.get_transition_states_and_probs((0,0),1)[0][1], 1)

  def test_trans_prob_non_deter(self):
    self.assertEqual(len(self.gw_non_deterministic.get_transition_states_and_probs((0,0),0)), 3)
    self.assertEqual(self.gw_non_deterministic.get_transition_states_and_probs((0,0),0)[0][0], (0,1))
    self.assertEqual(self.gw_non_deterministic.get_transition_states_and_probs((0,0),0)[0][1], 0.8)
    # print self.gw_non_deterministic.get_transition_states_and_probs((0,0),0)

    self.assertTrue(self.gw_non_deterministic.get_transition_states_and_probs((0,0),0)[1][1]-0.1 < 0.0001)
    self.assertTrue(self.gw_non_deterministic.get_transition_states_and_probs((0,0),0)[2][1]-0.1 < 0.0001)

    self.assertEqual(len(self.gw_non_deterministic.get_transition_states_and_probs((1,0),0)), 3)
    # self.assertEqual(self.gw_non_deterministic.get_transition_states_and_probs((1,0),0)[0][0], (0,1))
    # self.assertEqual(self.gw_non_deterministic.get_transition_states_and_probs((1,0),0)[0][1], 0.8)

  def test_terminals(self):
    self.assertTrue(self.gw_deterministic.is_terminal((0,4)))
    self.assertTrue(self.gw_deterministic.is_terminal((1,4)))

if __name__ == '__main__':
  unittest.main()