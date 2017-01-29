import gym
from gym import wrappers
import qlearning_agent
import numpy
import matplotlib.pyplot as plt

NUM_EPISODES=3000
N_BINS = [8, 8, 8, 8]
MAX_STEPS = 200
FAIL_PENALTY = -100
EPSILON=0.5
EPSILON_DECAY=0.99
LEARNING_RATE=0.05
DISCOUNT_FACTOR=0.9

RECORD=True

MIN_VALUES = [-0.5,-2.0,-0.5,-3.0]
MAX_VALUES = [0.5,2.0,0.5,3.0]
BINS = [numpy.linspace(MIN_VALUES[i], MAX_VALUES[i], N_BINS[i]) for i in xrange(4)]


def discretize(obs):
  return tuple([int(numpy.digitize(obs[i], BINS[i])) for i in xrange(4)])


def train(agent, env, history, num_episodes=NUM_EPISODES):
  for i in xrange(NUM_EPISODES):
    if i % 100:
      print "Episode {}".format(i+1)

    obs = env.reset()
    cur_state = discretize(obs)

    for t in xrange(MAX_STEPS):
      action = agent.get_action(cur_state)
      observation, reward, done, info = env.step(action)
      next_state = discretize(observation)

      if done:
        reward = FAIL_PENALTY
        agent.learn(cur_state, action, next_state, reward, done)
        print("Episode finished after {} timesteps".format(t+1))
        history.append(t+1)
        break

      agent.learn(cur_state, action, next_state, reward, done)
      cur_state = next_state

      if t == MAX_STEPS-1:
        history.append(t+1)
        print("Episode finished after {} timesteps".format(t+1))

  return agent, history



env = gym.make('CartPole-v0')
if RECORD:
  env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)

def get_actions(state):
  return [0, 1]

agent=qlearning_agent.QLearningAgent(get_actions, 
                                epsilon=EPSILON, 
                                alpha=LEARNING_RATE, 
                                gamma=DISCOUNT_FACTOR, 
                                epsilon_decay=EPSILON_DECAY)

history = []

agent, history = train(agent, env, history)

if RECORD:
  env.monitor.close()


plt.plot(history)
plt.ylabel('Rewards')
plt.show()


# gym.upload('/tmp/cartpole-experiment-1', api_key='sk_gw4ldruFRAc3gAhzCBxNw')

# while True:
#   obs = env.reset()
#   cur_state = discretize(obs)
#   done = False

#   t = 0
#   while not done:
#     env.render()
#     t = t+1
#     action = agent.get_action(cur_state)
#     observation, reward, done, info = env.step(action)
#     next_state = discretize(observation)
#     if done:
#       reward = FAIL_PENALTY
#       agent.learn(cur_state, action, next_state, reward, done)
#       print("Episode finished after {} timesteps".format(t+1))
#       history.append(t+1)
#       break
#     agent.learn(cur_state, action, next_state, reward, done)
#     cur_state = next_state
