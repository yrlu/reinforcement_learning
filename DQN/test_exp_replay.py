import unittest
import exp_replay
from exp_replay import Step
import numpy as np


class ExpReplayTest(unittest.TestCase):
  """
  Unit test for ExpReplay class
  """


  def test1(self):
    exprep = exp_replay.ExpReplay(mem_size=100, state_size=[1], kth=1)
    for i in xrange(120):
      exprep.add_step(Step(cur_step=i, action=0, next_step=i+1, reward=0, done=False))
    self.assertEqual(len(exprep.mem), 100)
    self.assertEqual(exprep.mem[-1:][0].cur_step, 119)


  def test2(self):
    exprep = exp_replay.ExpReplay(mem_size=100, state_size=[1], kth=4)
    for i in xrange(120):
      exprep.add_step(Step(cur_step=i, action=0, next_step=i+1, reward=0, done=False))
    self.assertEqual(len(exprep.mem), 100)
    self.assertEqual(exprep.mem[-1:][0].cur_step, 119)
    self.assertEqual(exprep.get_last_state(), [116,117,118,119])


  def test3(self):
    exprep = exp_replay.ExpReplay(mem_size=100, state_size=[2,2], kth=4)
    for i in xrange(120):
      exprep.add_step(Step(cur_step=[[i,i],[i,i]], action=0, next_step=[[i+1,i+1],[i+1,i+1]], reward=0, done=False))
    self.assertEqual(len(exprep.mem), 100)
    self.assertEqual(exprep.mem[-1:][0].cur_step, [[119,119],[119,119]])
    last_state = exprep.get_last_state()

    self.assertEqual(np.shape(last_state),(2,2,4))
    self.assertTrue(np.array_equal(last_state[:,:,0], [[116,116],[116,116]]))
    self.assertTrue(np.array_equal(last_state[:,:,1], [[117,117],[117,117]]))
    self.assertTrue(np.array_equal(last_state[:,:,2], [[118,118],[118,118]]))
    self.assertTrue(np.array_equal(last_state[:,:,3], [[119,119],[119,119]]))

    sample = exprep.sample(5)
    self.assertEqual(len(sample), 5)
    self.assertEqual(np.shape(sample[0].cur_step), (2,2,4))
    self.assertEqual(np.shape(sample[0].next_step), (2,2,4))


  def test4(self):
    exprep = exp_replay.ExpReplay(mem_size=100, state_size=[4], kth=1)
    for i in xrange(120):
      exprep.add_step(Step(cur_step=[i,i,i,i], action=0, next_step=[i+1,i+1,i+1,i+1], reward=0, done=False))
    last_state = exprep.get_last_state()
    self.assertEqual(np.shape(last_state),(4,))
    self.assertTrue(np.array_equal(last_state, [119,119,119,119]))

    sample = exprep.sample(5)
    self.assertEqual(len(sample), 5)
    self.assertEqual(np.shape(sample[0].cur_step), (4,))

if __name__ == '__main__':
  unittest.main()
