import actor
import critic
import worker
import tensorflow as tf
import gym
import numpy as np
import time
import threading
from threading import Lock


LOCAL_ACTOR_LEARNING_RATE = 0.0001
LOCAL_CRITIC_LEARNING_RATE = 0.001

MAX_STEPS = 300

STATE_SIZE = 4
ACTION_SIZE = 2


class A3C(object):


  def __init__(self, env_name, graph, name='a3c', gamma=0.99, n_workers=8):
    self.name = name
    self.env_name = env_name
    self.gamma = gamma
    self.n_workers = n_workers
    self.global_episodes = 0
    self.graph = graph
    self.sess = tf.Session()
    self.lock = Lock()

    self.global_actor = actor.ActorNetwork(STATE_SIZE, ACTION_SIZE, LOCAL_ACTOR_LEARNING_RATE, self.name+"_global_actor")
    self.global_critic = critic.CriticNetwork(STATE_SIZE, LOCAL_CRITIC_LEARNING_RATE, self.name+"_global_critic")
    self._init_workers()

    self.sess.run(tf.global_variables_initializer())

    self.env = gym.make(self.env_name)
    self.finished = 0


  def _init_workers(self):
    self.workers =[]
    # with tf.get_default_grap  h().as_default():
    for i in range(self.n_workers):
      env_i = gym.make(self.env_name)
      worker_i = worker.Worker(env_i, self.name+"_worker_"+str(i), self.global_actor, self.global_critic, 
        self.global_episodes, self.gamma, self.lock, self.sess)
      self.workers.append(worker_i)


  def _work(self, i, num_episodes):
    # with tf.get_default_graph().as_default():
    with self.graph.as_default():
      worker_i = self.workers[i]
      for i in xrange(num_episodes):
        worker_i.run_episode(300)
      self.finished += 1
      print "finished: {}".format(self.finished)


  def train(self, num_episodes=20):
    for i in range(self.n_workers):
      t = threading.Thread(target=self._work, args=(i, num_episodes)) 
      t.start()

    while not self.finished == self.n_workers:
      time.sleep(5)
    print "finished training!"


  def evaluate(self):
    rewards = []
    for i in range(50):
      state = self.env.reset()
      cum_reward = 0
      for t in xrange(MAX_STEPS):
        self.env.render()
        action = self.global_actor.get_action(state, self.sess)
        state, reward, done, info = self.env.step(action)
        cum_reward += reward
        if done:
          break
      print("Episode {} finished after {} timesteps, cum_reward: {}".format(i, t + 1, cum_reward))
      rewards.append(cum_reward)
    print "avg reward: {}".format(np.average(rewards))
    return rewards



