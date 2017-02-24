import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


Step = namedtuple('Step','cur_step action next_step reward done')


MAX_STEPS = 300
STATE_SIZE = 4
ACTION_SIZE = 2
ACTIONS = [0,1]
GAMMA = 0.99
LEARNING_RATE = 1e15

theta = np.ones([STATE_SIZE + 1])

def _phi(s, a):
  return [v for v in s] + [a]

def _h(s, a):
  # print len(_phi(s,a))
  return np.dot(theta, _phi(s,a))

def _policy(s):
  pi = []
  for a in ACTIONS:
    pi.append(np.exp(_h(s,a)))
  pi = np.divide(pi, sum(pi))
  return pi

def _R(episode, t):
  # t = range(len(episode))
  R_t = sum([GAMMA**i * s.reward for i, s in enumerate(episode[t:])])
  return R_t

def _log_pi(s, a):
  pi = _policy(s)
  return _phi(s,a) - np.sum([np.multiply(p, _phi(s, b)) for b,p in enumerate(pi)], axis=0)

def grad_eta(episode):
  grad = []
  for t, step in enumerate(episode):
    grad.append(np.multiply(_log_pi(step.cur_step, step.action), _R(episode, t)))
  return np.sum(grad, axis=0)

env = gym.make('CartPole-v0')

history = []
for i in range(50000):
  cur_state = env.reset()
  action = 1
  episode = []
  grad = []
  for t in xrange(MAX_STEPS):
    # action = env.action_space.sample()
    action_probs = _policy(cur_state)
    action = int(action_probs[0] < action_probs[1])
    # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    # print cur_state, theta, action_probs, action
    next_state, reward, done, info = env.step(action)
    # print '-------'
    # print 'phi:{}'.format(_phi(cur_state, action))
    # print 'h:{}'.format(_h(cur_state, action))
    # print 'policy:{}'.format(_policy(cur_state))
    # print 'log pi:{}'.format(_log_pi(cur_state, action))
    episode.append(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
    cur_state = next_state
    if done:
      history.append(t+1)
      break
  grad.append(grad_eta(episode))
  if (i + 1) % 500 == 0:
    print("episodes {} done. 50 episodes averaged length: {}".format(i, np.average(history[-50:])))
    print np.average(grad, axis=0)
    theta[:4] = theta[:4] + LEARNING_RATE * np.average(grad, axis=0)[:4]
    print theta
  # print episode
  # print 'R_t:{}'.format([_R(episode, t) for t, step in enumerate(episode)])
  # print 'grad_eta:{}'.format(grad_eta(episode))




import matplotlib.pyplot as plt
avg_reward = [np.mean(history[i*100:(i+1)*100]) for i in xrange(int(len(history)/100))]
f_reward = plt.figure(1)
plt.plot(np.linspace(0, len(history), len(avg_reward)), avg_reward)
plt.ylabel('Rewards')
f_reward.show()
print 'press enter to continue'
raw_input()