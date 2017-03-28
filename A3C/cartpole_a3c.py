import tensorflow as tf
import a3c



LOCAL_ACTOR_LEARNING_RATE = 0.0001
LOCAL_CRITIC_LEARNING_RATE = 0.001
GAMMA = 0.99
STATE_SIZE = 4
ACTION_SIZE = 2
DEVICE = 'cpu'


with tf.Graph().as_default() as g:
  with tf.device('/{}:0'.format(DEVICE)):
    g = tf.Graph()
    with g.as_default():
      agent = a3c.A3C('CartPole-v0', graph=g, gamma=GAMMA, n_workers=8)
      agent.train()
      agent.evaluate()
