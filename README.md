## Implementations of Reinforcement Learning algorithms in Python

To really understand RL algorithms and get adapted for RL research, I decided to implement popular RL algorithms myself.

#### Dependencies

- conda [(Installation)](http://conda.pydata.org/docs/install/quick.html)

<pre><code>$ conda create --name rl
$ source activate rl
</code></pre>

- Tensorflow

`$ conda install -c conda-forge tensorflow`

- OpenAI Gym

`$ pip install gym`

#### Environments

After some thoughts, I decided to implement my own environments (e.g. Gridworld) to gain a thorough understanding of how the agent interacts with environments and to hone my software engineering skills.

I did refer to some existing environment implementations to see their design trade-offs.

- Openai Gym

Most of the Gym environments requires only one agent.  Openai Gym assumes nothing about the structures of the agents, and hence, users can implement agents whatever as they like.

- UC Berkeley Pacman project

UCB Pacman project is more complicated. Some environment involves multiple agents. It defined a base interfaces for agents and had a class `game` to handle agents' information.







