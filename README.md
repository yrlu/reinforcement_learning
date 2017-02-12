## Implementations of Reinforcement Learning Algorithms in Python

(Working in progress) Implementations of selected reinforcement learning algorithms with tensorflow and openai gym. Trying to make each algorithm standalone and easy to read and run. 

-Yiren Lu (luyiren [at] seas [dot] upenn [dot] edu)

### Implemented Algorithms

##### Dynamic Programming MDP Solver

- `DP/value_iteration_agent.py`: value iteration
- `DP/policy_iteration_agent.py`: policy iteration - policy evaluation & policy improvement

##### TD Learning

- `qleanring_agent.py`: standard epsilon greedy qlearning agent
- `dqn.py`: Deep Q-network agent - vanilla network

##### Monte Carlo Methods

- `monte_carlo.py`: epsilon greedy monte carlo agent that learns episodes of experiences

##### Policy Gradient Methods

- `policy_gradient/policy_gradient_nn.py`: policy gradient agent with policy network

### OpenAI Gym Examples

- Cartpole-v0
  - `cartpole_qlearning.py`: [solved cartpole-v0](https://gym.openai.com/evaluations/eval_qXAq3TZxS6WBnMci1xJ4XQ#reproducibility)
  - `cartpole_dqn.py`: [solved cartpole-v0](https://gym.openai.com/evaluations/eval_ry9ynv6ZQQm14FJdT7dvQ)

- Breakout-v0 (code to be updated)

<img src="imgs/breakout3.gif" alt="breakout" width="200">

### Environments

- `envs/gridworld.py`: minimium gridworld implementation for testings

### Dependencies

- Numpy
- OpenAI Gym (with Atari)
- Tensorflow
- matplotlib (optional)

### Tests

- Files: `test_*.py`
- Run unit test for [class]:

`python test_[class].py`

- Test coverage (requires `coverage` and `nose`):

`nosetests --with-coverage --cover-package=.`

### MIT License


