# Some common agents
# ---
# @author Yiren Lu
# @email luyiren [at] seas [dot] upenn [dot] edu
# 
# MIT License

class Agent:

  def __init__(self, index=0):
    """
    input:
      index     the id of the agent
    """
    self.index = index

  def get_action(self, state):
    """
      input:
        state    current state
      output:
        an action to take given the state
    """
    abstract


class ValueIterationAgent(Agent):

  def __init__(self, gridworld, gamma):
    self.gw = gridworld
    self.gamma = gamma
    self.current_state = 

  def get_action(self, state):


