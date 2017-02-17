import gym
import numpy as np
import tensorflow as tf
import dqn_cnn4
import os
import sys
import pickle
import matplotlib.pyplot as plt


ACTIONS = {0:4, 1:5}
# ACTIONS = {0:1, 1:4, 2:5}
NUM_EPISODES = int(sys.argv[2])
DEVICE = sys.argv[1]
FAIL_PENALTY = -1
EPSILON = 1
EPSILON_DECAY = 0.001
END_EPSILON = 0.1
LEARNING_RATE = 2e-5
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
KTH_FRAME = 2
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



def test(agent, env, sess, sp, num_episodes=TEST_N_EPISODES):
  print 'testing ...'
  rewards = []
  for i in xrange(num_episodes):
    cum_reward = 0
    obs = env.reset()
    cur_frame = sp.process(sess, obs)
    action = 0
    done = False
    episode = [] # records all states
    t = 1
    while not done:
      t = t + 1
      if DISPLAY:
        env.render()
      episode.append([cur_frame, action])
      if (t % KTH_FRAME ==0) and (t > KTH_FRAME):
        last_s = [s for s, a in episode[-KTH_FRAME:]];
        last_s = np.reshape(last_s, [IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]]);
        # pick an actoin every k frame
        action = agent.get_optimal_action(last_s,sess)
      obs, reward, done, info = env.step(ACTIONS[action])
      cum_reward = cum_reward + reward
      cur_frame = sp.process(sess, obs)
    rewards.append(cum_reward)
    print 'test episode {}, reward: {}'.format(i, cum_reward)
  print '{} episodes average rewards with optimal policy: {}'.format(num_episodes, np.average(rewards))
  return np.average(rewards)


def train(agent, env, sess, sp, saver, num_episodes=NUM_EPISODES):
  history = []
  test_res = []
  for i in xrange(num_episodes):
    # save model
    if (i+1) % SAVE_EVERY_NUM_EPISODES == 0:
      saver.save(sess, MODEL_PATH)
      pickle.dump(agent.mem, open(MEMORY_PATH, "wb"))
      print 'saved model!'

    obs = env.reset()
    cur_frame = sp.process(sess, obs)

    action = 0
    episode = [] # records all states
    episode_train = [] # records step every k frames
    done = False
    t = 1
    last_life = 5
    cum_reward = 0
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

      episode.append([cur_frame, action,  next_frame, reward, done])
      cur_frame = next_frame
      
      
      if (t % KTH_FRAME ==0) and (t > KTH_FRAME*2):
        last_s = [s for s, a, s1, r, d in episode[-KTH_FRAME*2:-KTH_FRAME]];
        last_s = np.reshape(last_s, [IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]]);
        next_s = [s for s, a, s1, r, d in episode[-KTH_FRAME:]];
        next_s = np.reshape(next_s, [IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]]);
        last_r = sum([r for s, a, s1, r, d in episode[-KTH_FRAME*2:-KTH_FRAME]]);
        last_a = episode[-KTH_FRAME*2][1];

        # record step every k frames
        episode_train.append([last_s, last_a, next_s, last_r, done])

        # pick an actoin every k frame
        action = agent.get_action(next_s,sess)

    # for monitoring use
    print("Episode finished after {} timesteps, cumulated reward: {}".format(t, cum_reward))
    print agent.get_action_dist(episode_train[-1][0],sess), agent.get_optimal_action(episode_train[-1][0], sess)

    agent.add_episode(episode_train)
    history.append(t)

    # training and testing
    if i % TRAIN_EVERY_NUM_EPISODES == 0:
      print 'train at episode {}'.format(i)
      agent.learn(STEP_PER_EPOCH, sess)
    if i % TEST_EVERY_NUM_EPISODES == 0:
      test_res.append(test(agent, env, sess, sp))
  return agent, history, test_res


env = gym.envs.make(ENV_NAME)
tf.reset_default_graph()
sp = StateProcessor()

with tf.Session() as sess:
  with tf.device('/{}:0'.format(DEVICE)):
    agent = dqn_cnn4.DQNAgent_CNN(epsilon=EPSILON, epsilon_anneal=EPSILON_DECAY, end_epsilon=END_EPSILON, 
      lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, batch_size=BATCH_SIZE, state_size=IMAGE_SIZE, 
      action_size=len(ACTIONS), mem_size=MEM_SIZE)
  sess.run(tf.initialize_all_variables())
  saver = tf.train.Saver()
  if os.path.isdir(MODEL_DIR):
    saver.restore(sess, MODEL_PATH)
    agent.mem = pickle.load(open(MEMORY_PATH,"rb"))
    print 'restored model'
  else:
    os.makedirs(MODEL_DIR)
  # training
  agent, history, test_res = train(agent, env, sess, sp, saver)
  print history
  print test_res
  # save model
  saver.save(sess, MODEL_PATH)
  print 'saved model'
