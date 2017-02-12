import gym
import numpy as np
import dqn_cnn
import matplotlib.pyplot as plt
import time

from skimage.transform import resize
from skimage.color import rgb2gray



# ACTIONS = {0:1, 1:4, 2:5}
NUM_EPISODES = 2000
MAX_STEPS = 300
FAIL_PENALTY = -1
EPSILON = 1
EPSILON_DECAY = 0.001
END_EPSILON = 0.1
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
MEM_SIZE = 1e4
ENV_NAME = 'Breakout-v0'
STEP_PER_EPOCH = 200
RECORD = False
KTH_FRAME = 4

BATCH_SIZE = 64
IMAGE_SIZE = [84, 84]




def preprocess_frame(frame, size=IMAGE_SIZE):
  # print frame[0]
  return np.uint8(resize(rgb2gray(frame), (size[0], size[1])) * 255)
  # return resize(rgb2gray(frame) (size[0], size[1]))



def train(agent, env, history, num_episodes=NUM_EPISODES):
  for i in xrange(NUM_EPISODES):
    if i % 100:
      print "Episode {}".format(i + 1)
    obs = env.reset()
    cur_state = preprocess_frame(obs)
    episode = []
    done = False
    t = 0
    last_life = 5
    while not done:
      t = t + 1
      action = agent.get_action(cur_state)
      obs, reward, done, info = env.step(action)
      next_state = preprocess_frame(obs)
      if reward == 0:
        reward = info['ale.lives'] - last_life
        last_life = info['ale.lives']

      if reward > 0:
        reward = 1
      elif reward < 0:
        reward = -1

      if done:
        reward = FAIL_PENALTY
        episode.append([cur_state, action, next_state, reward, done])
        print("Episode finished after {} timesteps".format(t + 1))
        history.append(t + 1)
        break
      if t % KTH_FRAME == 0:
        episode.append([cur_state, action, next_state, reward, done])
      cur_state = next_state
    agent.learn(episode, STEP_PER_EPOCH)
  return agent, history


env = gym.make(ENV_NAME)
if RECORD:
  env = wrappers.Monitor(env, '/tmp/breakout-experiment-1', force=True)

#     epsilon=0.5, 
#     epsilon_anneal = 0.01,
#     end_epsilon=0.1,
#     lr=0.5, 
#     gamma=0.99, 
#     batch_size=64, 
#     state_size=[84,84,4],
#     action_size=2,
#     mem_size=1e4,

agent = dqn_cnn.DQNAgent_CNN(epsilon=EPSILON, epsilon_anneal=EPSILON_DECAY, end_epsilon=END_EPSILON, 
      lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, batch_size=BATCH_SIZE, state_size=IMAGE_SIZE, 
      action_size=6, mem_size=MEM_SIZE)


history = []
agent, history = train(agent, env, history)
print history


avg_reward = [np.mean(history[i*10:(i+1)*10]) for i in xrange(int(len(history)/10))]
f_reward = plt.figure(1)
plt.plot(np.linspace(0, len(history), len(avg_reward)), avg_reward)
plt.ylabel('Rewards')
f_reward.show()
print 'press enter to continue'
raw_input()
plt.close()

plt.ion()
plt.figure()

env = gym.make('Breakout-v0')
# for i_episode in range(20):
while True:
  observation = env.reset()
  cur_state = preprocess_frame(observation)
  done = False
  t = 0
  last_life = 5

  while not done:
    t = t + 1
    # if t % 4 == 0:
    #   img = preprocess_frame(observation)
    #   plt.imshow(img)
    #   plt.show()
    #   plt.pause(0.0001)

    # raw_input()
    # viewer = ImageViewer(img)
    # viewer.show()
    env.render()

    action = agent.get_action(cur_state)
    obs, reward, done, info = env.step(action)
    next_state = preprocess_frame(obs)

    # action = agent.get_optimal_action(cur_state)
    # observation, reward, done, info = env.step(action)
    if reward == 0:
      reward = info['ale.lives'] - last_life
      last_life = info['ale.lives']

    if reward > 0:
      reward = 1
    elif reward < 0:
      reward = -1

    if reward != 0:
      print reward, info
    # print info
    # print len(observation),len(observation[0]),len(observation[0][0])
    if done:
      print reward, info
      print("Episode finished after {} timesteps".format(t+1))
      break
