## Implementations of Reinforcement Learning algorithms in Python

### Dependencies

- OpenAI Gym
- Numpy

### Implemented Environments

- `mdp.py`: Abstract class of the environments
- `gridworld.py`: Gridworld based on mdp

### Implemented Agents

##### Dynamic Programming MDP Solver

- `value_iteration_agent.py`: value iteration
- `policy_iteration_agent.py`: policy iteration - policy evaluation & policy improvement

### Unit Tests

- Files: `test_*.py`
- Run unit test for [class]:

`python test_[class].py`

- Test coverage (requires `coverage` and `nose`):

`nosetests --with-coverage --cover-inclusive --cover-package=.`

### Math

Check out my [blog post: Learn Reinforcement Learning by Coding (in progress)](http://blog.luyiren.me/posts/reinforcement-learning-notes.html)

### MIT License


