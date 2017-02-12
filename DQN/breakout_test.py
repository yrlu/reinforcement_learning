import gym
import sys

print sys.argv

env = gym.make('Breakout-v0')
for i_episode in range(20):
  observation = env.reset()
  done = False
  cum_reward = 0
  t = 0
  while not done:
    t = t + 1
    env.render()
    # print(observation)
    action = env.action_space.sample()
    # action space  0: pause, 1: stay, 2: pause, 3: pause, 4: right, 5: left
    action = 4
    observation, reward, done, info = env.step(action)
    cum_reward = cum_reward + reward
    if done:
      print "Episode finished after {} timesteps, cumulated reward: {}".format(t+1, cum_reward)
      break
