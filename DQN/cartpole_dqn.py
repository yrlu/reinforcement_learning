import gym
from gym import wrappers
import dqn
import numpy
import matplotlib.pyplot as plt


NUM_EPISODES = 200
MAX_STEPS = 300
FAIL_PENALTY = -100
EPSILON = 1
EPSILON_DECAY = 0.01
END_EPSILON = 0.05
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9
BATCH_SIZE = 32
MEM_SIZE = 1e4

RECORD = False

def train(agent, env, history, num_episodes=NUM_EPISODES):
  for i in xrange(num_episodes):
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
        history.append(t + 1)
        break
      episode.append([cur_state, action, next_state, reward, done])
      cur_state = next_state
      if t == MAX_STEPS - 1:
        history.append(t + 1)
        print("Episode finished after {} timesteps".format(t + 1))
    agent.learn(episode, 100)
  return agent, history


env = gym.make('CartPole-v0')
if RECORD:
  env = wrappers.Monitor(env, '/tmp/cartpole-experiment-2', force=True)

agent = dqn.DQNAgent(epsilon=EPSILON, epsilon_anneal=EPSILON_DECAY, end_epsilon=END_EPSILON, 
      lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, batch_size=BATCH_SIZE, state_size=4, 
      action_size=2, mem_size=MEM_SIZE, n_hidden_1=10, n_hidden_2=10)


history = []
agent, history = train(agent, env, history)
print history


if RECORD:
  env.monitor.close()


avg_reward = [numpy.mean(history[i*10:(i+1)*10]) for i in xrange(int(len(history)/10))]
f_reward = plt.figure(1)
plt.plot(numpy.linspace(0, len(history), len(avg_reward)), avg_reward)
plt.ylabel('Episode length')
plt.xlabel('Training episodes')
f_reward.show()
print 'press enter to continue'
raw_input()
plt.close()


# Display:
print 'press ctrl-c to stop'
while True:
  cur_state = env.reset()
  done = False
  t = 0
  episode = []
  while not done:
    env.render()
    t = t+1
    action = agent.get_action(cur_state)
    next_state, reward, done, info = env.step(action)
    if done:
      reward = FAIL_PENALTY
      print("Episode finished after {} timesteps".format(t+1))
      history.append(t+1)
      break
    episode.append([cur_state, action, next_state, reward, done])
    cur_state = next_state
  agent.learn(episode, 100)
