import unittest
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from envs import gridworld
import policy_iteration_agent


class PolicyIterationAgentTest(unittest.TestCase):
  """
  Unit test for policy iteration agent
  """

  def setUp(self):
    grid = [['0', '0', '0', '1'],
            ['0', 'x', '0', '-1'],
            ['0', '0', '0', '0']]

    self.grid = grid
    self.gw_non_deterministic = gridworld.GridWorld(
        grid, {(0, 3), (1, 3)}, 0.8)

    self.agent = policy_iteration_agent.PolicyIterationAgent(
        self.gw_non_deterministic, 0.9, 20)

  def test_show_policy(self):
    print 'Show policy learned by value iteration:'
    self.gw_non_deterministic.display_policy_grid(
        self.agent.get_optimal_policy())

  def test_values(self):
    print 'Show value iteration results:'
    self.gw_non_deterministic.display_value_grid(self.agent.values)


if __name__ == '__main__':
  unittest.main()
