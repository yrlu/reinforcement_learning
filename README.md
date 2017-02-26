## Implementations of Reinforcement Learning Algorithms in Python

Implementations of selected reinforcement learning algorithms with tensorflow and openai gym. Working examples associated with implemented algorithms.

### Implemented Algorithms

(Reverse chronological order)

##### Policy Gradient Methods

- [`policy_gradient/policy_gradient_nn.py`](policy_gradient/policy_gradient_nn.py): REINFORCE with policy function approximation

##### Temporal Difference Learning

- [`TD/qlearning.py`](TD/qlearning.py): standard epsilon greedy qlearning
- [`DQN/dqn.py`](DQN/dqn.py): Q-learning with action value function approximation

##### Monte Carlo Methods

- [`monte_carlo/monte_carlo.py`](monte_carlo/monte_carlo.py): epsilon greedy monte carlo agent that learns episodes of experiences

##### Dynamic Programming MDP Solver

- [`DP/value_iteration.py`](DP/value_iteration.py): value iteration
- [`DP/policy_iteration.py`](DP/policy_iteration.py): policy iteration - policy evaluation & policy improvement

### OpenAI Gym Examples

- Cartpole-v0
  - [`TD/cartpole_qlearning.py`](TD/cartpole_qlearning.py): [solved cartpole-v0 after 1598 episodes of training](https://gym.openai.com/evaluations/eval_qXAq3TZxS6WBnMci1xJ4XQ#reproducibility)
  - [`DQN/cartpole_dqn.py`](DQN/): [solved cartpole-v0 after 75 episodes of training](https://gym.openai.com/evaluations/eval_ry9ynv6ZQQm14FJdT7dvQ)
  - [`policy_gradient/cartpole_policy_gradient.py`](policy_gradient/): REINFORCE [solved cartpole-v0 after 632 episodes](https://gym.openai.com/evaluations/eval_0qE4YdUoQMi60hslLEGg)

- Breakout-v0 (refactoring.., code coming soon)

<img src="imgs/breakout10.gif" alt="breakout" width="200">

### Environments

- `envs/gridworld.py`: minimium gridworld implementation for testings

### Dependencies

- Python 2.7
- Numpy
- Tensorflow
- OpenAI Gym (with Atari)
- matplotlib (optional)

### Tests

- Files: `test_*.py`
- Run unit test for [class]:

`python test_[class].py`

- Test coverage (requires `coverage` and `nose`):

`nosetests --with-coverage --cover-package=.`

### MIT License

