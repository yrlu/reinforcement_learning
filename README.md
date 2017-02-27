## Implementations of Reinforcement Learning Algorithms in Python

Implementations of selected reinforcement learning algorithms with tensorflow and openai gym. Working examples associated with implemented algorithms.

<!-- | Implemented Algorthms   |      Working Examples
|-----------------|:--------------|
| *Policy Gradient Methods* |   |
| [REINFORCE with policy function approximation](policy_gradient/) |    [`policy_gradient/cartpole_policy_gradient.py`](policy_gradient/)   |
| [REINFORCE with baseline](policy_gradient/) | [`policy_gradient/cartpole_reinforce_baseline.py`](policy_gradient/) |
| *TD Learning* |   |
| [Standard epsilon greedy Q-learning](TD/qlearning.py) | [`TD/cartpole_qlearning.py`](TD/cartpole_qlearning.py) |
| [Deep Q-learning](DQN/) | [`DQN/cartpole_dqn.py`](DQN/) |
| *Monte Carlo Methods* |   |
| [Monte Carlo (MC) estimation of action values](monte_carlo/monte_carlo.py) | [`monte_carlo/test_monte_carlo.py`](monte_carlo/test_monte_carlo.py) |
| *Dynamic Programming MDP Solver* |   |
| [Value iteration](DP/value_iteration.py) | [`DP/test_value_iteration.py`](DP/test_value_iteration.py) |
| [Policy iteration - policy evaluation & policy improvement](DP/policy_iteration.py) | [`DP/test_value_iteration.py`](DP/test_value_iteration.py) | -->

### Implemented Algorithms

(Reverse chronological order)

##### Policy Gradient Methods

- [REINFORCE with policy function approximation](policy_gradient/)
- [REINFORCE with baseline](policy_gradient/reinforce_w_baseline.py)

##### Temporal Difference Learning

- [Standard epsilon greedy Q-learning](TD/qlearning.py)
- [Deep Q-learning](DQN/)

##### Monte Carlo Methods

- [Monte Carlo (MC) estimation of action values](monte_carlo/monte_carlo.py)

##### Dynamic Programming MDP Solver

- [Value iteration](DP/value_iteration.py)
- [Policy iteration - policy evaluation & policy improvement](DP/policy_iteration.py)

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

