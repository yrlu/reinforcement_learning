## Implementations of Reinforcement Learning Algorithms in Python

-Yiren Lu (luyiren [at] seas [dot] upenn [dot] edu)

(Working in progress)

### Dependencies

- OpenAI Gym
- Numpy

### Implemented Environments

- `mdp.py`: Abstract class of the environments
- `gridworld.py`: Gridworld based on mdp

### Implemented Algorithms

##### Dynamic Programming MDP Solver

- `value_iteration_agent.py`: value iteration
- `policy_iteration_agent.py`: policy iteration - policy evaluation & policy improvement

##### Off-Policy TD Learning

- `qleanring_agent.py`: qlearning agent

### Unit Tests

- Files: `test_*.py`
- Run unit test for [class]:

`python test_[class].py`

- Test coverage (requires `coverage` and `nose`):

`nosetests --with-coverage --cover-inclusive --cover-package=.`

### Math

Check out my [blog post: Learn Reinforcement Learning by Coding (in progress)](http://blog.luyiren.me/posts/reinforcement-learning-notes.html)

### MIT License


