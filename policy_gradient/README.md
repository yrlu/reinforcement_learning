## Policy Gradient Methods

(Working in progress)

### REINFORCE 

[solved cartpole-v0 after 632 episodes](https://gym.openai.com/evaluations/eval_0qE4YdUoQMi60hslLEGg)

- `policy_gradient_nn.py`: REINFORCE with policy function approximation
- `cartpole_policy_gradient.py`: working example on cartpole-v0

#### Run Code

`$ python cartpole_policy_gradient.py`

#### Cartpole-v0 Result

![cartpole training](imgs/cartpole_reinforce.png "cartpole training")

### REINFORCE with Baseline

Have not been tuning the hyperparameters too much. Sometimes the model quickly converges to a local optimal (degenerate policy), but a few attempts (<5) should be sufficient.

- `reinforce_w_baseline.py`: REINFORCE with baseline
- `cartpole_policy_gradient_reinforce_baseline.py`: working example on cartpole-v0

#### Run Code

`$ python cartpole_policy_gradient_reinforce_baseline.py`

#### Cartpole-v0 Result

![cartpole training](imgs/cartpole_reinforce_w_baseline.png "cartpole training")