'''Example of A3C running on MountainCar environment'''
import argparse
import time
import threading
import tensorflow as tf
import gym
# import multiprocessing

import ac_net
import worker


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-d', '--device', default='cpu', type=str, help='choose device: cpu/gpu')
PARSER.add_argument('-e', '--episodes', default=20000, type=int, help='number of episodes')
PARSER.add_argument('-w', '--workers', default=8, type=int, help='number of workers')
PARSER.add_argument('-l', '--log_dir', default='mountaincar_logs', type=str, help='log directory')
ARGS = PARSER.parse_args()
print ARGS


DEVICE = ARGS.device
ENV_NAME = 'MountainCar-v0'
ENV = gym.make('MountainCar-v0')
STATE_SIZE = ENV.observation_space.shape[0]  # 2
ACTION_SIZE = ENV.action_space.n  # 3
LEARNING_RATE = 0.0001
GAMMA = 0.99
T_MAX = 5
# NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = ARGS.workers
NUM_EPISODES = ARGS.episodes
MAX_STEPS = 10000
LOG_DIR = ARGS.log_dir


N_H1 = 300
N_H2 = 300


def main():
  '''Example of A3C running on MountainCar environment'''
  tf.reset_default_graph()

  history = []

  with tf.device('/{}:0'.format(DEVICE)):
    sess = tf.Session()
    global_model = ac_net.AC_Net(
        STATE_SIZE,
        ACTION_SIZE,
        LEARNING_RATE,
        'global',
        n_h1=N_H1,
        n_h2=N_H2)
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

    for workeri in workers:
      worker_work = lambda: workeri.work(NUM_EPISODES)
      thread = threading.Thread(target=worker_work)
      thread.start()


if __name__ == "__main__":
  main()
