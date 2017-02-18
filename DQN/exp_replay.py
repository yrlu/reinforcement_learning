import numpy as np
import random
from collections import namedtuple


Step = namedtuple('Step','cur_step action next_step reward done')


class ExpReplay():
  """Experience replay"""


  def __init__(self, mem_size, state_size=[84, 84], kth=4, drop_rate=0.2):
    self.state_size = state_size
    self.drop_rate = drop_rate
    self.mem_size = mem_size
    self.kth = kth
    self.mem = []


  def add_step(self, step):
    """
    Store episode to memory and check if it reaches the mem_size. 
    If so, drop [self.drop_rate] of the oldest memory

    args
      step      namedtuple Step, where step.cur_step and step.next_step are of size {state_size}
    """
    self.mem.append(step)
    while len(self.mem) > self.mem_size:
      self.mem = self.mem[int(len(self.mem)*self.drop_rate):]


  def get_last_state(self):
    if len(self.mem) > self.kth:
      last_state = np.stack([s.cur_step for s in self.mem[-self.kth:]], axis=len(self.state_size))
      return last_state
    return []

  def sample(self, num):
    """Randomly draw [num] samples"""
    if len(self.mem) < self.mem_size/20:
      return []
    sampled_idx = random.sample(range(self.kth,len(self.mem)), num)
    samples = []
    for idx in sampled_idx:
      # found the last kth steps
      steps = self.mem[idx-self.kth:idx]
      cur_state = np.stack([s.cur_step for s in steps], axis=len(self.state_size))
      next_state = np.stack([s.next_step for s in steps], axis=len(self.state_size))
      reward = steps[-1].reward
      action = steps[-1].action
      done = steps[-1].action
      samples.append(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))
    return samples




