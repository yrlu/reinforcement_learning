## Implementation of Reinforcement Learning Algorithms in Python

Implementation of selected reinforcement learning algorithms with tensorflow.
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

(Click into the links for more details)

##### Advanced 

- [Asynchronized Advantage Actor-Critic (A3C)](A3C/)
- [Deep Deterministic Policy Gradient (DDPG)](ddpg/)

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

### Environments

- `envs/gridworld.py`: minimium gridworld implementation for testings

### Dependencies

- Python 2.7
- Numpy
- Tensorflow 0.12.1
- OpenAI Gym (with Atari) 0.8.0
- matplotlib (optional)

### Tests

- Files: `test_*.py`
- Run unit test for [class]:

`python test_[class].py`
<!-- 
- Test coverage (requires `coverage` and `nose`):

`nosetests --with-coverage --cover-package=.`
 -->
### MIT License

