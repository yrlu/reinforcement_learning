import unittest
import gridworld
import qlearning_agent


class QLearningAgentTest(unittest.TestCase):
  """
  Unit test for value iteration agent
  """


  def setUp(self):
    grid = [['0', '0', '0', '1'],
            ['0', 'x', '0', '-1'],
            ['0', '0', '0', '0']]

    self.grid = grid
    self.gw = gridworld.GridWorld(
        self.grid, {(0, 3), (1, 3)}, 0.8)

    self.agent = qlearning_agent.QLearningAgent(self.gw.get_actions, 
                  epsilon=0.1, alpha=0.5, gamma=0.9)
    
    # Training
    episodes = 5000
    for i in range(episodes):
      self.gw.reset((2,0))
      cur_s = self.gw.get_current_state()
      is_done = False
      while not is_done:
        a = self.agent.get_action(cur_s)
        last_state, action, next_state, reward, is_done = self.gw.step(a)
        self.agent.learn(last_state, action, next_state, reward, is_done)
        cur_s = next_state

  def test_show_policy(self):
    # show optimal policy
    opt_policy = self.gw.get_optimal_policy(self.agent)
    self.gw.display_policy_grid(opt_policy)
    self.gw.display_value_grid(self.gw.get_values(self.agent))
    self.gw.display_qvalue_grid(self.gw.get_qvalues(self.agent))

if __name__ == '__main__':
  unittest.main()
