# Utilities
#
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License


class Counter:
  """
  Counter class 
  """

  def __init__(self):
    self.counter = {}

  def add(self, key):
    if key in self.counter:
      self.counter[key] = self.counter[key] + 1
    else:
      self.counter[key] = 1

  def get(self, key):
    if key in self.counter:
      return self.counter[key]
    else:
      return 0
