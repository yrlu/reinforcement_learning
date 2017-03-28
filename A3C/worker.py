import tensorflow as tf
import numpy as np
import actor
import critic
from collections import namedtuple


LOCAL_ACTOR_LEARNING_RATE = 0.0001
LOCAL_CRITIC_LEARNING_RATE = 0.001

STATE_SIZE = 4
ACTION_SIZE = 2


Step = namedtuple('Step','cur_step action next_step reward done')


class Worker(object):


  def __init__(self, env, worker_name, global_actor, global_critic, global_episodes, gamma, lock, sess):
    self.env = env
    self.name = worker_name
    self.worker_name = worker_name
    self.global_actor = global_actor
    self.global_critic = global_critic
    self.gamma = gamma
    self.sess = sess
    self.lock = lock

    # total number of episodes
    self.T = global_episodes

    # get state size/action size from env
    # for simple low dimensional observations
    self.state_size = STATE_SIZE
    self.action_size = ACTION_SIZE

    # local copy of the actor and critic networks
    self.local_actor = actor.ActorNetwork(self.state_size, self.action_size, LOCAL_ACTOR_LEARNING_RATE, self.name+"_actor")
    self.local_critic = critic.CriticNetwork(self.state_size, LOCAL_CRITIC_LEARNING_RATE, self.name+"_critic")

    # actor gradients accumulators (dtheta)
    self.acc_grad_actor = [tf.get_variable("-".join(var.name.split(':'))+"_"+str(i), var.get_shape(),
                                          initializer=tf.constant_initializer(0))
                          for i, var in enumerate(self.local_actor.get_trainable_variables())]

    # critic gradients accumulators (dtheta_v)
    self.acc_grad_critic = [tf.get_variable("-".join(var.name.split(':'))+"_"+str(i), var.get_shape(),
                                          initializer=tf.constant_initializer(0))
                          for i, var in enumerate(self.local_critic.get_trainable_variables())]

    # reset to zero operations
    self.reset_zero_op = [acc_grad_a.assign(tf.zeros(acc_grad_a.get_shape()))
                              for acc_grad_a in self.acc_grad_actor] + \
                         [acc_grad_c.assign(tf.zeros(acc_grad_c.get_shape()))
                              for acc_grad_c in self.acc_grad_critic]

    # make local copy operations
    self.copy_to_local_op = [tf.assign(lcl, glb) for lcl, glb in zip(self.local_actor.get_trainable_variables(), self.global_actor.get_trainable_variables())] + \
                            [tf.assign(lcl, glb) for lcl, glb in zip(self.local_critic.get_trainable_variables(), self.global_critic.get_trainable_variables())]

    # accumulate gradients operations                        
    self.grad_actor = self.local_actor.gradients
    self.grad_critic = self.local_critic.gradients
    self.acc_grad_op = [acc_grad_a.assign_add(grad)
                        for acc_grad_a, grad in zip(self.acc_grad_actor, self.grad_actor)] + \
                       [acc_grad_c.assign_add(grad)
                        for acc_grad_c, grad in zip(self.acc_grad_critic, self.grad_critic)]
    
  def _copy_to_local(self):
    """copy global model's variables to local model"""
    self.sess.run(self.copy_to_local_op)


  def run_episode(self, t_max=10000):
    """accumulate gradients for an episode, following A3C paper's 
    pseudocode for each actor-learner thread
    """
    self.T += 1
    episode = []

    # reset gradients to zeros
    self._reset_accu_grad()
    # sync thread-specific params theta^prime = theta, theta^prime_v = theta_v
    self._copy_to_local()

    # get state s_t
    cur_state = self.env.reset()
    cum_reward = 0

    for t in xrange(t_max):
      # perform a_t according to policy pi(a_t|s_t; theta^prime)
      action = self.local_actor.get_action(cur_state, self.sess)
      # receive reward r_t and new state s_{t+1}
      next_state, reward, done, info = self.env.step(action)
      # T = T + 1
      
      cum_reward += reward
      if done:
        # episode.append(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
        break
      episode.append(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
      cur_state = next_state
    
    # for debug use
    print("Worker: {} Episode {} finished after {} timesteps, cum_reward: {}".format(self.worker_name, self.T, t + 1, cum_reward))
    print action

    if done:
      R = 0
    else:
      R = self.local_critic.get_value(next_state, self.sess)
    for i in reversed(xrange(len(episode))):
      step = episode[i]
      R = step.reward + self.gamma * R
      target_v = self.local_critic.get_value(step.cur_step, self.sess)
      self._acc_gradients(step.cur_step, step.action, R, target_v)
    
    # perform async update to global model (theta/theta_v)
    # not thread safe!
    self._update_global_model()


  def _reset_accu_grad(self):
    """reset accumulated gradients to zeros"""
    self.sess.run(self.reset_zero_op)


  def _acc_gradients(self, state, action, R, target_v):
    """add gradients to the accumulated gradients"""
    state = np.reshape(state,[-1, self.state_size])
    action = np.reshape(action, [-1])

    self.sess.run(self.acc_grad_op, feed_dict={
        self.local_actor.input_s: state,
        self.local_actor.action: action,
        self.local_actor.advantage: R - target_v,
        self.local_critic.input_s: state,
        self.local_critic.target_v: target_v
      })


  def _update_global_model(self):
    """
    perform async update of accumulated gradients to 
    the global actor/critic model
    """
    self.lock.acquire()
    self.global_actor.apply_gradients(self.acc_grad_actor, self.sess)
    self.global_critic.apply_gradients(self.acc_grad_critic, self.sess)
    self.lock.release()