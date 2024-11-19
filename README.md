# `marlenv` - A unified interface for muti-agent reinforcement learning
The objective of `marlenv` is to provide a common (typed) interface for many different reinforcement learning environments.

As such, `marlenv` provides high level abstractions of RL concepts such as `Observation`s or `Transition`s that are commonly represented as mere (confusing) lists or tuples.

## Using `marlenv` with existing libraries
`marlenv` unifies multiple popular libraries under a single interface. Namely, `marlenv` supports `smac`, `gymnasium` and `pettingzoo`.

```python
import marlenv

# You can instanciate gymnasium environments directly via their registry ID
gym_env = marlenv.make("CartPole-v1", seed=25)

# You can seemlessly instanciate a SMAC environment and directly pass your required arguments
from marlenv.adapters import SMAC
smac_env = env2 = SMAC("3m", debug=True, difficulty="9")

# pettingzoo is also supported
from pettingzoo.sisl import pursuit_v4
from marlenv.adapters import PettingZoo
pz_env = PettingZoo(pursuit_v4.parallel_env())
```


## Designing custom environments
You can create your own custom environment by inheriting from the `RLEnv` class. The below example illustrates a gridworld with a discrete action space. Note that other methods such as `step` or `render` must also be implemented.
```python
import numpy as np
from marlenv import RLEnv, DiscreteActionSpace, Observation

N_AGENTS = 3
N_ACTIONS = 5

class CustomEnv(RLEnv[DiscreteActionSpace]):
    def __init__(self, width: int, height: int):
        super().__init__(
            action_space=DiscreteActionSpace(N_AGENTS, N_ACTIONS),
            observation_shape=(height, width),
            state_shape=(1,),
        )
        self.time = 0

    def reset(self) -> Observation:
        self.time = 0
        ...
        return obs

    def get_state(self):
        return np.array([self.time])
```

## Useful wrappers
`marlenv` comes with multiple common environment wrappers, check the documentation for a complete list. The preferred way of using the wrappers is through a `marlenv.Builder`. The below example shows how to add a time limit (in number of steps) and an agent id to the observations of a SMAC environment.

```python
from marlenv import Builder
from marlenv.adapters import SMAC

env = Builder(SMAC("3m")).agent_id().time_limit(20).build()
print(env.extra_shape) # -> (4, ) because there are 3 agents and the time counter
```


# Related projects
- MARL: Collection of multi-agent reinforcement learning algorithms based on `marlenv` [https://github.com/yamoling/marl](https://github.com/yamoling/marl)
- Laser Learning Environment: a multi-agent gridworld that leverages `marlenv`'s capabilities [https://pypi.org/project/laser-learning-environment/](https://pypi.org/project/laser-learning-environment/)