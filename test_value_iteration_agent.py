import unittest
import gridworld
import value_iteration_agent


class ValueIterationAgentTest(unittest.TestCase):
  """
  Unit test for value iteration agent
  """

  def setUp(self):
    grid = [['0', '0', '0', '1'],
            ['0', 'x', '0', '-1'],
            ['0', '0', '0', '0']]

    self.grid = grid
    self.gw_deterministic = gridworld.GridWorld(grid, {(0, 3), (1, 3)}, 1)
    self.gw_non_deterministic = gridworld.GridWorld(
        grid, {(0, 3), (1, 3)}, 0.8)

    self.agent = value_iteration_agent.ValueIterationAgent(self.gw_non_deterministic, 0.9, 100)
    self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}

  def test_values(self):
    print
    self.gw_non_deterministic.display_value_grid(self.agent)

  def test_actions(self):
    print
    self.gw_non_deterministic.display_policy_grid(self.agent)
    # for i in range(len(self.grid)):
    # print [self.dirs[self.agent.get_action((i,j))] for j in
    # range(len(self.grid[0]))]

if __name__ == '__main__':
  unittest.main()
