import tensorflow as tf
import numpy as np
import ac_net
from collections import namedtuple
import random
import tf_utils

MAX_STEPS = 10000


Step = namedtuple('Step','cur_step action next_step reward done')


class Worker(object):


  def __init__(self, env, state_size, action_size, worker_name, global_name, lr, gamma, t_max, sess, history, n_h1=400, n_h2=300):
    self.env = env
    self.name = worker_name
    self.gamma = gamma
    self.sess = sess
    self.t_max = t_max
    self.history = history

    self.local_model = ac_net.AC_Net(state_size, action_size, lr, 
              worker_name, n_h1=n_h1, n_h2=n_h2, global_name=global_name)
    self.copy_to_local_op = tf_utils.update_target_graph(global_name, worker_name)

    self.summary_writer = tf.summary.FileWriter("logs/train_{}".format(worker_name))


  def _copy_to_local(self):
    self.sess.run(self.copy_to_local_op)


  def work(self, n_episodes):
    episode_i = 0
    episode_len = 1
    cur_state = self.env.reset()
    count = 1
    cum_reward = 0
    while episode_i < n_episodes:
      # 1) sync from global model to local model
      self._copy_to_local()
      # 2) collect t_max steps (if terminated then i++)
      steps = []
      for _ in xrange(self.t_max):
        action = self.local_model.get_action(cur_state, self.sess)
        # action = self.env.action_space.sample()
        next_state, reward, done, info = self.env.step(action)
        cum_reward += reward
        episode_len = episode_len + 1
        steps.append(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
        if done or episode_len >= MAX_STEPS:
          cur_state = self.env.reset()
          self.history.append(episode_len)
          summary = tf.Summary()
          summary.value.add(tag='Perf/episode_len', simple_value=float(episode_len))
          summary.value.add(tag='Perf/episode_reward', simple_value=float(cum_reward))
          self.summary_writer.add_summary(summary, episode_i)
          print 'worker {}: episode {} finished in {} steps, cumulative reward: {}'.format(self.name, episode_i, episode_len, cum_reward)
          print action
          print self.local_model.predict_policy(cur_state, self.sess)
          cum_reward = 0
          episode_i = episode_i + 1
          episode_len = 0
          break
        cur_state = next_state
      # 3) convert the t_max steps into a batch
      if steps[-1].done:
        R = 0
      else:
        R = self.local_model.predict_value(cur_state, self.sess)
      R_batch = np.zeros(len(steps))
      advantage_batch = np.zeros(len(steps))
      target_v_batch = np.zeros(len(steps))
      for i in reversed(xrange(len(steps))):
        step = steps[i]
        R = step.reward + self.gamma * R
        R_batch[i] = R
      cur_state_batch = [step.cur_step for step in steps]
      pred_v_batch = self.local_model.predict_value(cur_state_batch, self.sess)
      action_batch = [step.action for step in steps]
      advantage_batch = [R_batch[i]-pred_v_batch[i] for i in xrange(len(steps))]
      # 4) compute the gradient and update the global model
      action_batch = np.reshape(action_batch, [-1])
      advantage_batch = np.reshape(advantage_batch, [-1])
      R_batch = np.reshape(R_batch, [-1])
      feed_dict = {
        self.local_model.input_s: cur_state_batch,
        self.local_model.input_a: action_batch,
        self.local_model.advantage: advantage_batch,
        self.local_model.target_v: R_batch,
      }
      v_l, p_l, e_l, loss, _, _, v_n = self.sess.run(
                                              [self.local_model.value_loss, 
                                              self.local_model.policy_loss, 
                                              self.local_model.entropy_loss, 
                                              self.local_model.loss, 
                                              self.local_model.gradients, 
                                              self.local_model.apply_gradients, 
                                              # self.local_model.grad_norms, 
                                              self.local_model.var_norms], 
                                              feed_dict)


      mean_reward = np.mean([step.reward for step in steps])
      mean_value = np.mean(R_batch)

      summary = tf.Summary()
      summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
      summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
      summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
      summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
      summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
      # summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
      summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
      self.summary_writer.add_summary(summary, count)
      count += 1

