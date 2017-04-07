import ac_net
import worker
import tensorflow as tf
import gym
import numpy as np
import time
import threading
import multiprocessing
import matplotlib.pyplot as plt


DEVICE = 'cpu'
STATE_SIZE = 4
ACTION_SIZE = 2
# LEARNING_RATE = 0.00002
LEARNING_RATE = 0.00001
GAMMA = 0.99
T_MAX = 5
NUM_WORKERS = multiprocessing.cpu_count()
NUM_EPISODES = 1000

N_H1 = 300
N_H2 = 300

tf.reset_default_graph()

history = []
# with tf.Graph().as_default() as g:
with tf.device('/{}:0'.format(DEVICE)):
  # g = tf.Graph()
  # with g.as_default():
  sess = tf.Session()
  global_model = ac_net.AC_Net(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, 'global', n_h1=N_H1, n_h2=N_H2)
  workers = []
  for i in xrange(NUM_WORKERS):

    workers.append(worker.Worker(gym.make('CartPole-v0'), 
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



plot_curve(history, 10)
















      