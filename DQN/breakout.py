import gym
import numpy as np
import sys

env = gym.make('Breakout-v0')
for i_episode in range(20):
  print i_episode
  observation = env.reset()
  # cur_state = preprocess_frame(observation)
  done = False
  t = 0
  last_life = 5

  while not done:
    t = t + 1
    env.render()
    # action = agent.get_action(cur_state)
    # action = 4
    action = env.action_space.sample()
    # print env.action_space.n
    # print action
    # 1: stay, 4: right 5: left
    # action = 3

    action = int(sys.stdin.readline())
    # print line

    obs, reward, done, info = env.step(action)
    # next_state = preprocess_frame(obs)

    if reward == 0:
      reward = info['ale.lives'] - last_life
      last_life = info['ale.lives']

    if reward > 0:
      reward = 1
    elif reward < 0:
      reward = -1

    if reward != 0:
      print reward, info

    if done:
      print reward, info
      print("Episode finished after {} timesteps".format(t+1))
      break