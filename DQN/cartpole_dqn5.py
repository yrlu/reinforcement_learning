import gym
import sys
import tensorflow as tf
import dqn5
import numpy
import exp_replay
from exp_replay import Step


DEVICE = sys.argv[1]
NUM_EPISODES = int(sys.argv[2])
MAX_STEPS = 300
FAIL_PENALTY = -100
EPSILON = 1
EPSILON_DECAY = 0.01
END_EPSILON = 0.1
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9
BATCH_SIZE = 32
MEM_SIZE = 1e4
STATE_SIZE = [4]
ACTIONS = {0:0, 1:1}


def train(agent, exprep, env, sess):
  epsilon = EPSILON
  for i in xrange(NUM_EPISODES):
    cur_state = env.reset()
    for t in xrange(MAX_STEPS):
      action = agent.get_action_e(cur_state, sess, epsilon)
      next_state, reward, done, info = env.step(action)
      if done:
        reward = FAIL_PENALTY
        exprep.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
        print("Episode {} finished after {} timesteps".format(i, t + 1))
        break
      exprep.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
      cur_state = next_state
      if t == MAX_STEPS - 1:
        history.append(t + 1)
        print("Episode {} finished after {} timesteps".format(i, t + 1))
      samples = exprep.sample(BATCH_SIZE)
      # if len(samples)>0:
        # print samples[0].reward, samples[0].action, samples[0].done
      agent.learn(samples, sess)
    if epsilon > END_EPSILON:
        epsilon = epsilon - EPSILON_DECAY


env = gym.make('CartPole-v0')
exprep = exp_replay.ExpReplay(mem_size=MEM_SIZE, state_size=STATE_SIZE, kth=1)

with tf.Session() as sess:
  with tf.device('/{}:0'.format(DEVICE)):
    agent = dqn5.DQNAgent(lr=LEARNING_RATE, 
                          gamma=DISCOUNT_FACTOR,
                          state_size=STATE_SIZE,
                          action_size=len(ACTIONS),
                          scope="dqn",
                          n_hidden_1=10,
                          n_hidden_2=10)
  sess.run(tf.initialize_all_variables())
  train(agent, exprep, env, sess)


