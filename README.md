## Implementations of Reinforcement Learning Algorithms in Python

-Yiren Lu (luyiren [at] seas [dot] upenn [dot] edu)

(Working in progress)

### Dependencies

- OpenAI Gym (with Atari)
- Numpy
- matplotlib

### Implemented Algorithms

##### Dynamic Programming MDP Solver

- `value_iteration_agent.py`: value iteration
- `policy_iteration_agent.py`: policy iteration - policy evaluation & policy improvement

##### TD Learning

- `qleanring_agent.py`: epsilon greedy qlearning agent

##### Monte Carlo Methods

- `monte_carlo.py`: epsilon greedy monte carlo agent that learns episodes of experiences

### Implemented Environments

- `mdp.py`: Abstract class of markov decision process
- `env.py`: Abstract class of environments
- `gridworld.py`: Gridworld based on mdp.MDP and env.Env

### Unit Tests

- Files: `test_*.py`
- Run unit test for [class]:

`python test_[class].py`

- Test coverage (requires `coverage` and `nose`):

`nosetests --with-coverage --cover-package=.`

### OpenAI Gym Examples

- Cartpole-v0
  - `cartpole_qlearning.py`: [solved cartpole-v0](https://gym.openai.com/evaluations/eval_qXAq3TZxS6WBnMci1xJ4XQ#reproducibility)

### Math

Check out my [blog post: Learn Reinforcement Learning by Coding (in progress)](http://blog.luyiren.me/posts/reinforcement-learning-notes.html)

### MIT License


