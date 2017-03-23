# Deep Deterministic Policy Gradient
#   following paper: Continuous control with deep reinforcement learning
#                   (https://arxiv.org/pdf/1509.02971.pdf)
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License

import numpy as np
import tensorflow as tf


class DDPG(object):


  def __init__(self, actor, critic, exprep, noise, gamma=0.99, action_bound=1):
    self.actor = actor
    self.critic = critic
    self.exprep = exprep
    self.noise = noise
    self.total_steps = 0
    self.gamma = 0.99
    self.action_bound = action_bound


  def add_step(self, step):
    self.total_steps = self.total_steps + 1
    self.exprep.add_step(step)


  def get_action(self, state, sess):
    state = np.reshape(state,[-1, self.actor.state_size])
    action = self.actor.get_action(state, sess) * self.action_bound
    return action


  def get_action_noise(self, state, sess, rate=1):
    state = np.reshape(state,[-1, self.actor.state_size])
    action = self.actor.get_action(state, sess) * self.action_bound
    action = action + self.noise.noise() * rate
    # print self.ou.noise()
    return action


  def learn_batch(self, sess):
    # sample a random minibatch of N tranistions
    batch = self.exprep.sample()
    if len(batch)==0:
      return

    # compute y_i (target q)
    next_s = [s.next_step for s in batch]
    next_a_target = self.actor.get_action_target(next_s, sess)
    next_q_target = self.critic.get_qvalue_target(next_s, next_a_target, sess)
    y = np.array([s.reward + self.gamma*next_q_target[i]*(1-s.done) for i,s in enumerate(batch)])
    y = y.reshape([len(batch)])

    # update ciritc by minimizing l2 loss
    cur_s = [s.cur_step for s in batch]
    a = [s.action for s in batch]
    l = self.critic.train(cur_s, a, y, sess)

    # update actor policy with sampled gradient
    cur_a_pred = self.actor.get_action(cur_s, sess)
    a_gradients = self.critic.get_gradients(cur_s, cur_a_pred, sess)
    self.actor.train(cur_s, a_gradients[0], sess)

    # update target network:
    self.actor.update_target(sess)
    self.critic.update_target(sess)
    return l


