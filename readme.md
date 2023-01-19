# RLEnv: yet another RL framework
This framework aims at high level abstractions of RL models, allowing to build algorithms on top of it.

## Designing an environment
To create an environment that is compatible with RLEnv, you should inherit from the `rl.models.RLEnv` class.

## Instanciating an environment
### Simple environments
```python
import rlenv
print(rlenv.__version__)

# From Gym
env = rlenv.make("CartPole-v1")

# From pettingzoo
from pettingzoo.sisl import pursuit_v4
env = rlenv.make(pursuit_v4.parallel_env())
```

### Adding extra information to the observations
```python
import rlenv
# Building the environment with additional information
from pettingzoo.sisl import pursuit_v4
env = rlenv.Builder(pursuit_v4.parallel_env())\
    .with_agent_id()\
    .with_last_action()\
    .build()
# 8 agents  + 5 actions = 13 extras
assert env.extra_feature_shape == (13, )
```