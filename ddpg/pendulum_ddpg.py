# DDPG Pendulum-v0 example
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import tensorflow as tf
import numpy as np
import argparse
from ddpg import DDPG
from actor import ActorNetwork
from critic import CriticNetwork
from exp_replay import ExpReplay
from exp_replay import Step
from ou import OUProcess
import matplotlib.pyplot as plt
import sys
import gym
from gym import wrappers


parser = argparse.ArgumentParser(description=None)
parser.add_argument('-d', '--device', default='cpu', type=str, help='choose device: cpu/gpu')
parser.add_argument('-e', '--episodes', default=300, type=int, help='number of episodes')
parser.add_argument('-l', '--log_dir', default='/tmp/pendulum-log-0', type=str, help='log directory')
args = parser.parse_args()
print(args)


DEVICE = args.device
NUM_EPISODES = args.episodes
LOG_DIR=args.log_dir

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




def summarize(cum_reward, i, summary_writer):
  summary = tf.Summary()
  summary.value.add(tag="cumulative reward", simple_value=cum_reward)
  summary_writer.add_summary(summary, i)
  summary_writer.flush()


def train(agent, env, sess):
  for i in xrange(NUM_EPISODES):
    cur_state = env.reset()
    cum_reward = 0
    # tensorboard summary
    summary_writer = tf.summary.FileWriter(LOG_DIR+'/train', graph=tf.get_default_graph())

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
        summarize(cum_reward, i, summary_writer)
        break
      cum_reward += reward
      agent.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
      cur_state = next_state
      if t == MAX_STEPS - 1:
        print("Episode {} finished after {} timesteps, cum_reward: {}".format(i, t + 1, cum_reward))
        print action
        summarize(cum_reward, i, summary_writer)
      agent.learn_batch(sess)


env = gym.make('Pendulum-v0')
# env = wrappers.Monitor(env, '/tmp/pendulum-experiment-0', force=True)

actor = ActorNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE, lr=ACTOR_LEARNING_RATE, tau=TAU)
critic = CriticNetwork(state_size=STATE_SIZE, action_size=ACTION_SIZE, lr=CRITIC_LEARNING_RATE, tau=TAU)
noise = OUProcess(ACTION_SIZE)
exprep = ExpReplay(mem_size=MEM_SIZE, start_mem=10000, state_size=[STATE_SIZE], kth=-1, batch_size=BATCH_SIZE)

sess = tf.Session()
with tf.device('/{}:0'.format(DEVICE)):
  agent = DDPG(actor=actor, critic=critic, exprep=exprep, noise=noise, action_bound=env.action_space.high)
sess.run(tf.initialize_all_variables())

train(agent, env, sess)
