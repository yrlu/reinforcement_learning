import tensorflow as tf
import gym
import time
import threading
# import multiprocessing
import argparse
import ac_net
import worker


parser = argparse.ArgumentParser(description=None)
parser.add_argument('-d', '--device', default='cpu', type=str, help='choose device: cpu/gpu')
parser.add_argument('-e', '--episodes', default=20000, type=int, help='number of episodes')
parser.add_argument('-w', '--workers', default=8, type=int, help='number of workers')
parser.add_argument('-l', '--log_dir', default='mountaincar_logs', type=str, help='log directory')
args = parser.parse_args()
print(args)



DEVICE = args.device
ENV_NAME = 'MountainCar-v0'
env = gym.make('MountainCar-v0')
STATE_SIZE = env.observation_space.shape[0] # 2
ACTION_SIZE = env.action_space.n # 3
LEARNING_RATE = 0.0001
GAMMA = 0.99
T_MAX = 5
# NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = args.workers
NUM_EPISODES = args.episodes
MAX_STEPS = 10000
LOG_DIR = args.log_dir


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
      history=history, n_h1=N_H1, n_h2=N_H2, logdir=LOG_DIR))

  sess.run(tf.global_variables_initializer())

  for worker in workers:
    worker_work = lambda: worker.work(NUM_EPISODES)
    t = threading.Thread(target=worker_work)
    t.start()


