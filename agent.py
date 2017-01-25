# Some common agents
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
# 
# MIT License

class Agent:

  def __init__(self, index=0):
    """
    args
      index     the id of the agent
    """
    self.index = index

  def get_action(self, state):
    """
      args
        state    current state
      returns
        an action to take given the state
    """
    abstract