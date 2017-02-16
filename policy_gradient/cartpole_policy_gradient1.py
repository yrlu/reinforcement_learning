import gym
from gym import wrappers
import policy_gradient
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf

NUM_EPISODES = 5000
MAX_STEPS = 200
FAIL_PENALTY = -1
EPSILON=0.5
EPSILON_ANNEAL = 0.01
END_EPSILON=0.1
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
TRAIN_EVERY_NUM_EPISODES = 1
EPOCH_SIZE = 1
MEM_SIZE = 100

RECORD = False

def train(agent, env, num_episodes=NUM_EPISODES):
  history = []
  for i in xrange(NUM_EPISODES):
    if i % 100:
      print "Episode {}".format(i + 1)
    cur_state = env.reset()
    episode = []
    for t in xrange(MAX_STEPS):
      action = agent.get_action(cur_state)
      next_state, reward, done, info = env.step(action)
      if done:
        reward = FAIL_PENALTY
        episode.append([cur_state, action, next_state, reward, done])
        print("Episode finished after {} timesteps".format(t + 1))
        print agent.get_policy(cur_state)
        history.append(t + 1)
        break
      episode.append([cur_state, action, next_state, 1, done])
      cur_state = next_state
      if t == MAX_STEPS - 1:
        history.append(t + 1)
        print("Episode finished after {} timesteps".format(t + 1))
    # agent.add_episode(episode)
    if i % TRAIN_EVERY_NUM_EPISODES == 0:
      print 'train at episode {}'.format(i)
      agent.learn(episode)
  return agent, history


agent = policy_gradient.PolicyGradientAgent(epsilon=EPSILON,
                                          epsilon_anneal=EPSILON_ANNEAL,
                                          end_epsilon=END_EPSILON,
                                          lr=LEARNING_RATE,
                                          gamma=DISCOUNT_FACTOR,
                                          state_size=4,
                                          action_size=2)


env = gym.make('MountainCar-v0')

agent, history = train(agent, env)


avg_reward = [numpy.mean(history[i*100:(i+1)*100]) for i in xrange(int(len(history)/100))]
f_reward = plt.figure(1)
plt.plot(numpy.linspace(0, len(history), len(avg_reward)), avg_reward)
plt.ylabel('Rewards')
f_reward.show()
print 'press enter to continue'
raw_input()

