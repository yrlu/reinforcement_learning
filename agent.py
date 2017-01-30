# Some common agents
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
#
# MIT License


class Agent(object):

  def get_action(self):
    """
    returns
      an action to take
    """
    abstract

class RLAgent(Agent):

  def learn(self, s, a, s1, r):
    """
    args
      s     current state
      a     action taken
      s1    next state
      r     reward
    """
    abstract


