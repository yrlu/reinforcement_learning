import gym
from gym import wrappers
import reinforce_w_baseline
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf

NUM_EPISODES = 200
MAX_STEPS = 300
FAIL_PENALTY = -100
LEARNING_RATE = 0.002 
DISCOUNT_FACTOR = 0.9
TRAIN_EVERY_NUM_EPISODES = 1
EPOCH_SIZE = 1
MEM_SIZE = 100

RECORD = False


def train(agent, env, sess, num_episodes=NUM_EPISODES):
  history = []
  for i in xrange(NUM_EPISODES):
    if i % 100:
      print "Episode {}".format(i + 1)
    cur_state = env.reset()
    episode = []
    for t in xrange(MAX_STEPS):
      action = agent.get_action(cur_state, sess)
      next_state, reward, done, info = env.step(action)
      if done:
        reward = FAIL_PENALTY
        episode.append([cur_state, action, next_state, reward, done])
        print("Episode finished after {} timesteps".format(t + 1))
        print agent.get_policy(cur_state, sess)
        history.append(t + 1)
        break
      episode.append([cur_state, action, next_state, 1, done])
      cur_state = next_state
      if t == MAX_STEPS - 1:
        history.append(t + 1)
        print("Episode finished after {} timesteps".format(t + 1))
    if i % TRAIN_EVERY_NUM_EPISODES == 0:
      print 'train at episode {}'.format(i)
      agent.learn(episode, sess, EPOCH_SIZE)
  return agent, history


agent = reinforce_w_baseline.PolicyGradientNNAgent(lr=LEARNING_RATE,
                                          gamma=DISCOUNT_FACTOR,
                                          state_size=4,
                                          action_size=2,
                                          n_hidden_1=10,
                                          n_hidden_2=10)


env = gym.make('CartPole-v0')


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  agent, history = train(agent, env, sess)


window = 10
avg_reward = [numpy.mean(history[i*window:(i+1)*window]) for i in xrange(int(len(history)/window))]
f_reward = plt.figure(1)
plt.plot(numpy.linspace(0, len(history), len(avg_reward)), avg_reward)
plt.ylabel('Rewards')
plt.xlabel('Episodes')
f_reward.show()
print 'press enter to continue'
raw_input()

