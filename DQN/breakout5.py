import gym
import numpy as np
import tensorflow as tf

import os
import sys
import pickle
import matplotlib.pyplot as plt

import dqn_cnn5
import exp_replay
from exp_replay import Step



ACTIONS = {0:4, 1:5}
# ACTIONS = {0:1, 1:4, 2:5}
NUM_EPISODES = int(sys.argv[2])
DEVICE = sys.argv[1]
FAIL_PENALTY = -1
EPSILON = 1
EPSILON_DECAY = 1e-6
END_EPSILON = 0.1
# LEARNING_RATE = 2e-5
LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 32
KTH_FRAME = 4
IMAGE_SIZE = [84, 84, KTH_FRAME]
MEM_SIZE = 1e5
ENV_NAME = 'Breakout-v0'
STEP_PER_EPOCH = 100
RECORD = False
TRAIN_EVERY_NUM_EPISODES = 1
TEST_EVERY_NUM_EPISODES = 40
TEST_N_EPISODES = 10
SAVE_EVERY_NUM_EPISODES = 500

DISPLAY = False

MODEL_DIR = '/tmp/breakout-experiment-5'
MODEL_PATH = '/tmp/breakout-experiment-5/model'
MEMORY_PATH = '/tmp/breakout-experiment-5/memory.p'


plt.ion()


def save_model(sess, saver, exprep, model_dir=MODEL_DIR, model_path=MODEL_PATH, memory_path=MEMORY_PATH):
  saver.save(sess, MODEL_PATH)
  pickle.dump(exprep.mem, open(MEMORY_PATH, "wb"))
  print 'Saved model'

def restore_model(sess, saver, exprep, model_dir=MODEL_DIR, model_path=MODEL_PATH, memory_path=MEMORY_PATH):
  if os.path.isdir(MODEL_DIR):
    saver.restore(sess, MODEL_PATH)
    exprep.mem = pickle.load(open(MEMORY_PATH,"rb"))
    print 'Restored model'
  else:
    print 'Model directory not found. Created model directory'
    os.makedirs(MODEL_DIR)


class BreakoutStateProcessor():
  """A working state processor from:
  https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Deep%20Q%20Learning%20Solution.ipynb
  """

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


def test(agent, env, sess, exprep, sp, num_episodes=TEST_N_EPISODES):
  print 'testing ...'
  rewards = []
  for i in xrange(num_episodes):
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
      # main testing procedure
      # print action
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
      # add step to exprep
      exprep.add_step(Step(cur_step=cur_frame, action=action, next_step=next_frame, reward=reward, done=done))
      cur_frame = next_frame
      if (t % KTH_FRAME ==0):
        action = agent.get_optimal_action(exprep.get_last_state(), sess)
    rewards.append(cum_reward)
    print 'test episode {}, reward: {}'.format(i, cum_reward)
    last_state = exprep.get_last_state()
    print agent.get_action_values(last_state ,sess), agent.get_optimal_action(last_state, sess)
  print '{} episodes average rewards with optimal policy: {}'.format(num_episodes, np.average(rewards))
  return np.average(rewards)

def train(agent, env, sess, exprep, sp, saver, num_episodes=NUM_EPISODES):
  total_steps = 1
  epsilon = EPSILON
  for i in xrange(NUM_EPISODES):
    obs = env.reset()
    cur_frame = sp.process(sess, obs)
    action = 0
    done = False
    t = 1
    last_life = 5
    cum_reward = 0
    while not done:
      # counter updates
      t = t + 1
      total_steps = total_steps + 1
      if total_steps % 10000 == 0:
        print 'total_steps: {}'.format(total_steps)
      if DISPLAY:
        env.render()
      # main training procedure
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
      # add step to exprep
      exprep.add_step(Step(cur_step=cur_frame, action=action, next_step=next_frame, reward=reward, done=done))
      cur_frame = next_frame
      agent.learn(exprep.sample(BATCH_SIZE), sess)
      # choose action every kth step:
      if (t % KTH_FRAME ==0):
        action = agent.get_action_e(exprep.get_last_state(), sess, epsilon)
      # update epsilon
      if epsilon > END_EPSILON:
        epsilon = epsilon - EPSILON_DECAY

    # for monitoring use
    print("Episode {} finished after {} timesteps, cumulated reward: {}".format(i, t, cum_reward))
    last_state = exprep.get_last_state()
    print agent.get_action_values(last_state,sess), agent.get_optimal_action(last_state, sess)
    print len(exprep.mem)
    print epsilon
    # save model
    if (i+1) % SAVE_EVERY_NUM_EPISODES == 0:
      save_model(sess, saver, exprep)
    if i % TEST_EVERY_NUM_EPISODES == 0:
      test(agent, env, sess, exprep, sp)


env = gym.envs.make(ENV_NAME)
tf.reset_default_graph()
sp = BreakoutStateProcessor()
exprep = exp_replay.ExpReplay(mem_size=MEM_SIZE, state_size=IMAGE_SIZE[:2], kth=KTH_FRAME)

with tf.Session() as sess:
  with tf.device('/{}:0'.format(DEVICE)):
    agent = dqn_cnn5.DQNAgent_CNN(lr=LEARNING_RATE, 
                                  gamma=DISCOUNT_FACTOR,
                                  state_size=IMAGE_SIZE,
                                  action_size=len(ACTIONS),
                                  scope="dqn")
  sess.run(tf.initialize_all_variables())
  saver = tf.train.Saver()
  restore_model(sess, saver, exprep)
  agent = train(agent, env, sess, exprep, sp, saver)
  save_model(sess, saver, exprep)









