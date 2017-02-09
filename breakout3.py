import gym
import numpy as np
import tensorflow as tf
import dqn_cnn2
import os
import sys

# action space  0: pause, 1: stay, 2: pause, 3: pause, 4: right, 5: left
ACTIONS = {0:1, 1:4, 2:5}
NUM_EPISODES = 1000
FAIL_PENALTY = -1
EPSILON = 1
EPSILON_DECAY = 0.001
END_EPSILON = 0.3
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
MEM_SIZE = 1e5
ENV_NAME = 'Breakout-v0'
STEP_PER_EPOCH = 100
RECORD = False
KTH_FRAME = 4
TRAIN_EVERY_NUM_EPISODES = 1
TEST_EVERY_NUM_EPISODES = 40
TEST_N_EPISODES = 10

BATCH_SIZE = 64
IMAGE_SIZE = [84, 84]

DISPLAY = False

MODEL_DIR = '/tmp/breakout-experiment-3'
MODEL_PATH = '/tmp/breakout-experiment-3/model'



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


def test(agent, env, sess, num_episodes=TEST_N_EPISODES):
  print 'testing ...'
  rewards = []
  for i in xrange(num_episodes):
    cum_reward = 0
    obs = env.reset()
    cur_state = sp.process(sess, obs)
    action = 0
    done = False
    while not done:
      action = agent.get_optimal_action(cur_state, sess)
      obs, reward, done, info = env.step(ACTIONS[action])
      cum_reward = cum_reward + reward
      cur_state = sp.process(sess, obs)
      # print reward, done, info
      if done:
        rewards.append(cum_reward)
	print 'test episode {}, reward: {}'.format(i, cum_reward)
  	break
  # print rewards
  print '{} episodes average rewards with optimal policy: {}'.format(num_episodes, np.average(rewards))


def train(agent, env, history, sess, num_episodes=NUM_EPISODES):
  for i in xrange(num_episodes):
    obs = env.reset()
    cur_state = sp.process(sess, obs)

    action = 0
    episode = []
    done = False
    t = 0
    last_life = 5
    cum_reward = 0
    while not done:
      t = t + 1
      # action = env.action_space.sample()
      action = agent.get_action(cur_state,sess)
      # print action
      if DISPLAY:
        # action = agent.get_optimal_action(cur_state, sess)
        # print agent.get_action_dist(cur_state,sess), agent.get_optimal_action(cur_state, sess)
        env.render()
        # if action != 0:
          # print '~0'
        # while action==0:
          # action = env.action_space.sample()
      # else:
        # action = agent.get_action(cur_state,sess)
      # print len(cur_state), len(cur_state[0]), cur_state[15:-15][15:-15], action
      obs, reward, done, info = env.step(ACTIONS[action])
      cum_reward = cum_reward + reward
      # next_state = sp.process(sess, obs)
	

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
        episode.append([cur_state, action, sp.process(sess, obs), reward, done])
        print("Episode finished after {} timesteps, cumulated reward: {}".format(t+1, cum_reward))
        history.append(t + 1)
        break
      if t % KTH_FRAME == 0:
        next_state = sp.process(sess, obs)
        episode.append([cur_state, action, next_state, reward, done])
        cur_state = next_state

    print agent.get_action_dist(cur_state,sess), agent.get_optimal_action(cur_state, sess)
    agent.add_episode(episode)
    if i % TRAIN_EVERY_NUM_EPISODES == 0:
      print 'train at episode {}'.format(i)
      agent.learn(STEP_PER_EPOCH, sess)
    if i % TEST_EVERY_NUM_EPISODES == 0:
      test(agent, env, sess)
  return agent, history


env = gym.envs.make(ENV_NAME)

tf.reset_default_graph()
sp = StateProcessor()


with tf.Session() as sess:
  with tf.device('/{}:0'.format(sys.argv[1])):
    agent = dqn_cnn2.DQNAgent_CNN(epsilon=EPSILON, epsilon_anneal=EPSILON_DECAY, end_epsilon=END_EPSILON, 
      lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, batch_size=BATCH_SIZE, state_size=IMAGE_SIZE, 
      action_size=len(ACTIONS), mem_size=MEM_SIZE)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  if os.path.isdir(MODEL_DIR):
    saver.restore(sess, MODEL_PATH)
    print 'restored model'
  else:
    os.makedirs(MODEL_DIR)

  history = []
  agent, history = train(agent, env, history, sess)
  print history

  saver.save(sess, MODEL_PATH)
  print 'saved model'
