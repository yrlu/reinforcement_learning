import gym
import numpy as np
import tensorflow as tf

import os
import sys
import pickle
import matplotlib.pyplot as plt

import dqn_cnn6
import exp_replay
from exp_replay import Step


ACTIONS = {0:4, 1:5}
# ACTIONS = {0:1, 1:4, 2:5}
NUM_EPISODES = int(sys.argv[2])
DEVICE = sys.argv[1]
FAIL_PENALTY = -1
EPSILON = 1
EPSILON_DECAY = 0.001
END_EPSILON = 0.1
LEARNING_RATE = 2e-5
# LEARNING_RATE = 0.00025
MOMENTUM=0.95
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 32
KTH_FRAME = 1
IMAGE_SIZE = [84, 84, KTH_FRAME]
MEM_SIZE = 1e5
START_MEM = MEM_SIZE/20
ENV_NAME = 'Breakout-v0'
EPOCH_SIZE = 100

TEST_EVERY_NUM_EPISODES = 40
TEST_N_EPISODES = 10
# SAVE_EVERY_NUM_EPISODES = 500

DISPLAY = False

# MODEL_DIR = '/tmp/breakout-experiment-6'
# MODEL_PATH = '/tmp/breakout-experiment-6/model'
# MEMORY_PATH = '/tmp/breakout-experiment-6/memory.p'
# LOG_PATH = '/tmp/breakout-experiment-log-6'

class StateProcessor():


  def __init__(self):
    # Build the Tensorflow graph
    with tf.variable_scope("state_processor"):
      self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
      self.output = tf.image.rgb_to_grayscale(self.input_state)
      self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
      self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      self.output = tf.squeeze(self.output)


  def process(self, sess, state):
    """
    Args:
        sess: A Tensorflow session object
        state: A [210, 160, 3] Atari RGB State

    Returns:
        A processed [84, 84, 1] state representing grayscale values.
    """
    return sess.run(self.output, { self.input_state: state })


def test(agent, exprep, sp, env, sess):
  print 'testing ...'
  rewards = []
  for i in xrange(TEST_N_EPISODES):
    cum_reward = 0
    obs = env.reset()
    cur_frame = sp.process(sess, obs)
    action = 0
    done = False
    last_life = 5
    t = 1
    while not done:
      t = t + 1
      if DISPLAY:
        env.render()
      obs, reward, done, info = env.step(ACTIONS[action])
      cum_reward = cum_reward + reward
      next_frame = sp.process(sess, obs)

      # process reward: if lost life: -1, if hit the ball: +2, if still living: +1  (avoid 0 rewards)
      if reward == 0:
        reward = info['ale.lives'] - last_life
        last_life = info['ale.lives']
      if reward == 0:
        reward = 1
      else:
        if reward > 0:
          reward = 2
        elif reward < 0:
          reward = -1
      if done:
        reward = FAIL_PENALTY

      exprep.add_step(Step(cur_step=cur_frame, action=action, next_step=next_frame, reward=reward, done=done))
      cur_frame = next_frame
      if (t % KTH_FRAME ==0):
        action = agent.get_optimal_action(exprep.get_last_state())
    rewards.append(cum_reward)
    print 'test episode {}, reward: {}'.format(i, cum_reward)
    last_state = exprep.get_last_state()
    # print agent.get_action_values(last_state), agent.get_optimal_action(last_state)
  print '{} episodes average rewards with optimal policy: {}'.format(TEST_N_EPISODES, np.average(rewards))
  return np.average(rewards)


def train(agent, exprep, sp, env, sess):
  for i in xrange(NUM_EPISODES):
    obs = env.reset()
    cur_state = sp.process(sess, obs)
    action = 0
    done = False
    last_life = 5
    t = 1
    cum_reward = 0
    while not done:
      t = t + 1
      if DISPLAY:
        env.render()
      if exprep.total_steps % 10000 == 0:
        print '--total_steps: {}--'.format(exprep.total_steps)
      obs, reward, done, info = env.step(ACTIONS[action])
      cum_reward = cum_reward + reward
      next_state = sp.process(sess, obs)

      # process reward: if lost life: -1, if hit the ball: +2, if still living: +1  (avoid 0 rewards)
      if reward == 0:
        reward = info['ale.lives'] - last_life
        last_life = info['ale.lives']
      if reward == 0:
        reward = 1
      else:
        if reward > 0:
          reward = 2
        elif reward < 0:
          reward = -1
      if done:
        reward = FAIL_PENALTY

      exprep.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
      
      
      cur_state = next_state
      if (t % KTH_FRAME ==0):
        action = agent.get_action(exprep.get_last_state())
    agent.epsilon_decay()
    agent.learn_epoch(exprep, EPOCH_SIZE)
    print("Episode {} finished after {} timesteps, cumulative rewards: {} ".format(i, t + 1, cum_reward))
    print agent.get_action_values(exprep.get_last_state()), agent.get_optimal_action(exprep.get_last_state())
    print 'epsilon: {}'.format(agent.epsilon)

    if i % TEST_EVERY_NUM_EPISODES == 0:
      test(agent, exprep, sp, env, sess)


env = gym.make(ENV_NAME)
exprep = exp_replay.ExpReplay(mem_size=MEM_SIZE, start_mem=START_MEM, state_size=IMAGE_SIZE[:2], kth=KTH_FRAME, batch_size=BATCH_SIZE)
tf.reset_default_graph()
sp = StateProcessor()

sess = tf.Session()
with tf.device('/{}:0'.format(DEVICE)):
  agent = dqn_cnn6.DQNAgent_CNN(session=sess, epsilon=EPSILON, epsilon_anneal=EPSILON_DECAY, end_epsilon=END_EPSILON, 
        lr=LEARNING_RATE, momentum=MOMENTUM, gamma=DISCOUNT_FACTOR, state_size=IMAGE_SIZE, 
        action_size=len(ACTIONS))
sess.run(tf.initialize_all_variables())
train(agent, exprep, sp, env, sess)
