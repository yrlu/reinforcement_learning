# DDPG Pendulum-v0 example
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import tensorflow as tf
import numpy as np
from ddpg import DDPG
from actor import ActorNetwork
from critic import CriticNetwork
from exp_replay import ExpReplay
from exp_replay import Step
from ou import OUProcess
import matplotlib.pyplot as plt
import sys
import gym

DEVICE = sys.argv[1]
NUM_EPISODES = int(sys.argv[2])
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
MEM_SIZE = 1000000

STATE_SIZE = 3
ACTION_SIZE = 1
BATCH_SIZE = 64
MAX_STEPS = 10000
FAIL_PENALTY = 0
ACTION_RANGE = 1
EVALUATE_EVERY = 10


def train(agent, env, sess):
  for i in xrange(NUM_EPISODES):
    cur_state = env.reset()
    cum_reward = 0
    cum_l = 0
    if (i % EVALUATE_EVERY) == 0:
      print '====evaluation===='
    for t in xrange(MAX_STEPS):
      if (i % EVALUATE_EVERY) == 0:
        env.render()
        action = agent.get_action(cur_state, sess)[0]
      else:
        # decaying noise
        action = agent.get_action_noise(cur_state, sess, rate=(NUM_EPISODES-i)/NUM_EPISODES)[0]
      next_state, reward, done, info = env.step(action)
      if done:
        cum_reward += reward
        agent.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
        print("Episode {} finished after {} timesteps, cum_reward: {}".format(i, t + 1, cum_reward))
        print action
        yield cum_reward
        break
      cum_reward += reward
      agent.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
      cur_state = next_state
      if t == MAX_STEPS - 1:
        print("Episode {} finished after {} timesteps, cum_reward: {}".format(i, t + 1, cum_reward))
        print action
        yield cum_reward
      l = agent.learn_batch(sess)
      if not l is None:
        cum_l += l

    print 'cum_l: {}'.format(cum_l)


env = gym.make('Pendulum-v0')
actor = ActorNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE, optimizer=tf.train.AdamOptimizer(ACTOR_LEARNING_RATE), tau=TAU)
critic = CriticNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE, optimizer=tf.train.AdamOptimizer(CRITIC_LEARNING_RATE), tau=TAU)
noise = OUProcess(ACTION_SIZE)
exprep = ExpReplay(mem_size=MEM_SIZE, start_mem=10000, state_size=[STATE_SIZE], kth=-1, batch_size=BATCH_SIZE)

sess = tf.Session()
with tf.device('/{}:0'.format(DEVICE)):
  agent = DDPG(actor=actor, critic=critic, exprep=exprep, noise=noise, action_bound=env.action_space.high)
sess.run(tf.initialize_all_variables())

history = [c_r for c_r in train(agent, env, sess)]

# plot
window = 2
avg_reward = [np.mean(history[i*window:(i+1)*window]) for i in xrange(int(len(history)/window))]
f_reward = plt.figure(1)
plt.plot(np.linspace(0, len(history), len(avg_reward)), avg_reward)
plt.ylabel('Cumulative rewards')
f_reward.show()

print 'press enter to continue'
raw_input()


