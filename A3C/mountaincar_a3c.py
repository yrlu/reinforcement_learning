import ac_net
import worker
import tensorflow as tf
import gym
import numpy as np
import time
import threading
import multiprocessing
import matplotlib.pyplot as plt



# env = gym.make('MountainCar-v0')
# print(env.observation_space.shape[0])
# print(env.action_space.n)



# env = gym.make('MountainCar-v0')
# env._max_episode_steps = 10000
# for i_episode in range(100):
#   cum_reward = 0
#   observation = env.reset()
#   for t in range(10000):
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     cum_reward += reward
#     # print observation, action, reward, done, info
#     if done:
#       print("Episode finished after {} timesteps, cumulated reward: {}".format(t+1, cum_reward))
#       cum_reward = 0
#       break




DEVICE = 'cpu'
ENV_NAME = 'MountainCar-v0'
env = gym.make('MountainCar-v0')
STATE_SIZE = env.observation_space.shape[0] # 2
ACTION_SIZE = env.action_space.n # 3
LEARNING_RATE = 0.0001
GAMMA = 0.99
T_MAX = 200
NUM_WORKERS = multiprocessing.cpu_count()
# NUM_WORKERS = 4
NUM_EPISODES = 500
MAX_STEPS = 10000



N_H1 = 300
N_H2 = 300

tf.reset_default_graph()

history = []

with tf.device('/{}:0'.format(DEVICE)):
  sess = tf.Session()
  global_model = ac_net.AC_Net(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, 'global', n_h1=N_H1, n_h2=N_H2)
  workers = []
  for i in xrange(NUM_WORKERS):
    env = gym.make(ENV_NAME)
    env._max_episode_steps = MAX_STEPS
    workers.append(worker.Worker(env, 
      state_size=STATE_SIZE, action_size=ACTION_SIZE, 
      worker_name='worker_{}'.format(i), global_name='global', 
      lr=LEARNING_RATE, gamma=GAMMA, t_max=T_MAX, sess=sess, 
      history=history, n_h1=N_H1, n_h2=N_H2))

  sess.run(tf.global_variables_initializer())

  for worker in workers:
    worker_work = lambda: worker.work(NUM_EPISODES)
    t = threading.Thread(target=worker_work)
    t.start()

while(len(history) < NUM_EPISODES*NUM_WORKERS * 0.9):
  time.sleep(5)


def plot_curve(history, smooth=10):
  window = smooth
  avg_reward = [np.mean(history[i*window:(i+1)*window]) for i in xrange(int(len(history)/window))]
  f_reward = plt.figure(1)
  plt.plot(np.linspace(0, len(history), len(avg_reward)), avg_reward)
  plt.ylabel('Rewards')
  f_reward.show()
  print 'press enter to continue'
  raw_input()



plot_curve(history, 20)

