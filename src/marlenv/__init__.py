"""
`marlenv` is a strongly typed library for multi-agent and multi-objective reinforcement learning.

It aims to provide a simple and consistent interface for reinforcement learning environments by providing abstraction models such as `Observation`s or `Episode`s. `marlenv` provides adapters for popular libraries such as `gym` or `pettingzoo` and provides utility wrappers to add functionalities such as video recording or limiting the number of steps.

Almost every class is a dataclassto enable seemless serialiation with the `orjson` library.

# Existing environments
The `MARLEnv` class represents a multi-agent RL environment and is at the center of this library, and `marlenv` provides an adapted implementation of multiple common MARL environments (gym, pettingzoo, smac and overcooked) in `marlenv.adapters`. Note that these adapters will only work if you have the corresponding library installed.

```python
from marlenv.adapters import Gym, PettingZoo, SMAC, Overcooked
import marlenv

env1 = Gym("CartPole-v1")
env2 = marlenv.make("CartPole-v1")
env3 = PettingZoo("prospector_v4")
env4 = SMAC("3m")
env5 = Overcooked.from_layout("cramped_room")
```

# Wrappers & Builder
To facilitate the create of an environment with common wrappers, `marlenv` provides a `Builder` class that can be used to chain the creation of multiple wrappers.

```python
from marlenv import make, Builder

env = <your env>
env = Builder(env).agent_id().time_limit(50).record("videos").build()
```

# Using the library
A typical environment loop would look like this:

```python
from marlenv import DiscreteMockEnv, Builder, Episode

env = Builder(DicreteMockEnv()).agent_id().build()
obs, state = env.reset()
terminated = False
episode = Episode.new(obs, state)
while not episode.is_finished:
    action = env.sample_action() # a valid random action
    step = env.step(action) # Step data `step.obs`, `step.reward`, ...
    episode.add(step, action) # Progressively build the episode
```

# Extras
To cope with complex observation spaces, `marlenv` distinguishes the "main" observation data from the "extra" observation data. A typical example would be the observation of a gridworld environment with a time limit. In that case, the main observation has shape (height, width), i.e. the content of the grid, but the current time is an extra observation data of shape (1, ).

```python
env = GridWorldEnv()
print(env.observation_shape) # (height, width)
print(env.extras_shape) # (0, )

env = Builder(env).time_limit(25).build()
print(env.observation_shape) # (height, width)
print(env.extras_shape) # (1, )
```

# Creating a new environment
If you want to create a new environment, you can simply create a class that inherits from `MARLEnv`. If you want to create a wrapper around an existing `MARLEnv`, you probably want to subclass `RLEnvWrapper` which implements a default behaviour for every method.
"""

__version__ = "3.3.3"

from . import models
from . import wrappers
from . import adapters
from .models import spaces


from .env_builder import make, Builder
from .models import (
    MARLEnv,
    State,
    Step,
    Observation,
    Episode,
    Transition,
    DiscreteSpace,
    ContinuousSpace,
    ActionSpace,
    DiscreteActionSpace,
    ContinuousActionSpace,
)
from .wrappers import RLEnvWrapper
from .mock_env import DiscreteMockEnv, DiscreteMOMockEnv

__all__ = [
    "models",
    "wrappers",
    "adapters",
    "spaces",
    "make",
    "Builder",
    "MARLEnv",
    "Step",
    "State",
    "Observation",
    "Episode",
    "Transition",
    "ActionSpace",
    "DiscreteSpace",
    "ContinuousSpace",
    "DiscreteActionSpace",
    "ContinuousActionSpace",
    "DiscreteMockEnv",
    "DiscreteMOMockEnv",
    "RLEnvWrapper",
]
