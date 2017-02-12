import unittest
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from envs import gridworld
import monte_carlo


class MonteCarloAgentTest(unittest.TestCase):
  """
  Unit test for monte carlo agent
  """


  def test2(self):
    print 'Test 2 -- Gridworld test'
    grid = [['0', '0', '0', '1'],
            ['0', 'x', '0', '-1'],
            ['0', '0', '0', '0']]

    gw = gridworld.GridWorld(
        grid, {(0, 3), (1, 3)}, 0.8)

    agent = monte_carlo.MonteCarloAgent(gw.get_actions, 
                  epsilon=0.4, gamma=0.9, alpha=0.01, epsilon_decay=1)
    # Training
    episodes = 1000
    for i in range(episodes):
      episode = []
      gw.reset((2,0))
      cur_s = gw.get_current_state()
      is_done = False
      while not is_done:
        a = agent.get_action(cur_s)
        last_state, action, next_state, reward, is_done = gw.step(a)
        episode.append((last_state, action, next_state, reward))
        # agent.learn(last_state, action, next_state, reward, is_done)
        cur_s = next_state
        if is_done:
          agent.learn(episode)
    
    opt_policy = gw.get_optimal_policy(agent)
    gw.display_policy_grid(opt_policy)
    gw.display_value_grid(gw.get_values(agent))
    gw.display_qvalue_grid(gw.get_qvalues(agent))


  def test1(self):
    print 'Test 1 -- test G_t'
    G_t = monte_carlo.MonteCarloAgent.compute_G_t([1,2,3,4], 0.5)
    self.assertEqual(G_t, [3.25,4.5,5,4])

if __name__ == '__main__':
  unittest.main()