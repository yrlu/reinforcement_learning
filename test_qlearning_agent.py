import unittest
import gridworld
import qlearning_agent


class QLearningAgentTest(unittest.TestCase):
  """
  Unit test for value iteration agent
  """

  def test2(self):
    print 'Test 1 -- Bridge Crossing Analysis'
    grid = [['x', '-100', '-100', '-100', 'x'],
            ['1', '0',    '0',    '0',   '10'],
            ['x', '-100', '-100', '-100', 'x']]

    gw = gridworld.GridWorld(
        grid, {(1,0), (1,4), 
               (0,1), (0,2), (0,3), 
               (2,1), (2,2), (2,3)}, 0.9)

    agent = qlearning_agent.QLearningAgent(gw.get_actions, 
                  epsilon=0.1, alpha=0.5, gamma=0.9)

    # Training
    episodes = 5000
    for i in range(episodes):
      gw.reset((1,1))
      cur_s = gw.get_current_state()
      is_done = False
      while not is_done:
        a = agent.get_action(cur_s)
        last_state, action, next_state, reward, is_done = gw.step(a)
        agent.learn(last_state, action, next_state, reward, is_done)
        cur_s = next_state
    # show optimal policy
    opt_policy = gw.get_optimal_policy(agent)
    gw.display_policy_grid(opt_policy)
    gw.display_value_grid(gw.get_values(agent))
    gw.display_qvalue_grid(gw.get_qvalues(agent))

  def test1(self):
    print 'Test 1 -- Regular Case'
    grid = [['0', '0', '0', '1'],
            ['0', 'x', '0', '-1'],
            ['0', '0', '0', '0']]

    gw = gridworld.GridWorld(
        grid, {(0, 3), (1, 3)}, 0.8)

    agent = qlearning_agent.QLearningAgent(gw.get_actions, 
                  epsilon=0.2, alpha=0.5, gamma=0.9)
    
    # Training
    episodes = 5000
    for i in range(episodes):
      gw.reset((2,0))
      cur_s = gw.get_current_state()
      is_done = False
      while not is_done:
        a = agent.get_action(cur_s)
        last_state, action, next_state, reward, is_done = gw.step(a)
        agent.learn(last_state, action, next_state, reward, is_done)
        cur_s = next_state

    # show optimal policy
    opt_policy = gw.get_optimal_policy(agent)
    gw.display_policy_grid(opt_policy)
    gw.display_value_grid(gw.get_values(agent))
    gw.display_qvalue_grid(gw.get_qvalues(agent))

if __name__ == '__main__':
  unittest.main()
