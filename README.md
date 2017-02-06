## Implementations of Reinforcement Learning Algorithms in Python

(Working in progress) Implementations of selected reinforcement learning algorithms with tensorflow and openai gym. Trying to make each algorithm standalone and easy to read and run. 

-Yiren Lu (luyiren [at] seas [dot] upenn [dot] edu)

### Dependencies

- Numpy
- OpenAI Gym (with Atari)
- Tensorflow
- matplotlib (optional)

### Implemented Algorithms

##### Dynamic Programming MDP Solver

- `value_iteration_agent.py`: value iteration
- `policy_iteration_agent.py`: policy iteration - policy evaluation & policy improvement

##### TD Learning

- `qleanring_agent.py`: standard epsilon greedy qlearning agent
- `dqn.py`: Deep Q-network agent - vanilla network

##### Monte Carlo Methods

- `monte_carlo.py`: epsilon greedy monte carlo agent that learns episodes of experiences

### OpenAI Gym Examples

- Cartpole-v0
  - `cartpole_qlearning.py`: [solved cartpole-v0](https://gym.openai.com/evaluations/eval_qXAq3TZxS6WBnMci1xJ4XQ#reproducibility)
  - `cartpole_dqn.py`: [solved cartpole-v0](https://gym.openai.com/evaluations/eval_ry9ynv6ZQQm14FJdT7dvQ)

### Environments

- `gridworld.py`: minimium gridworld implementation for testings

### Tests

- Files: `test_*.py`
- Run unit test for [class]:

`python test_[class].py`

- Test coverage (requires `coverage` and `nose`):

`nosetests --with-coverage --cover-package=.`

### MIT License


