import unittest
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from envs import gridworld
import dqn



class DQNAgentTest(unittest.TestCase):
  """
  Unit test for deep q-learning agent
  """

  def test1(self):
    print 'Test 1 -- Regular Case'
    grid = [['0', '0', '0', '1'],
            ['0', 'x', '0', '-1'],
            ['0', '0', '0', '0']]
    gw = gridworld.GridWorld(grid, {(0, 3), (1, 3)}, 0.6)
    def valid_actions():
      return [0, 1, 2, 3, 4]
    agent = dqn.DQNAgent(epsilon=1, epsilon_anneal=0.001, end_epsilon=0.1, 
      lr=0.001, gamma=0.9, batch_size=32, state_size=2, action_size=5, mem_size=1e4,
      n_hidden_1=10, n_hidden_2=10)
    episodes = 5000
    for i in range(episodes):
      if i % 200 == 0:
        print '-------',i,'-------'
        # show optimal policy
        opt_policy = gw.get_optimal_policy(agent)
        gw.display_policy_grid(opt_policy)
        gw.display_value_grid(gw.get_values(agent))
        gw.display_qvalue_grid(gw.get_qvalues(agent))

      gw.reset((2,0))
      cur_s = gw.get_current_state()
      is_done = False
      episode = []
      while not is_done:
        a = agent.get_action(cur_s)
        last_state, action, next_state, reward, is_done = gw.step(a)
        episode.append([[last_state[0], last_state[1]], action, [next_state[0], next_state[1]], reward, is_done])
        # agent.learn(last_state, action, next_state, reward, is_done)
        cur_s = next_state
      agent.learn(episode, 4)

    # show optimal policy
    opt_policy = gw.get_optimal_policy(agent)
    gw.display_policy_grid(opt_policy)
    gw.display_value_grid(gw.get_values(agent))
    gw.display_qvalue_grid(gw.get_qvalues(agent))


if __name__ == '__main__':
  unittest.main()

    
