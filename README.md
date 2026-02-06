# `marlenv` - A unified framework for muti-agent reinforcement learning
**Documentation: [https://yamoling.github.io/multi-agent-rlenv](https://yamoling.github.io/multi-agent-rlenv)**

`marlenv` is a strongly typed library for multi-agent and multi-objective reinforcement learning.

Install the library with
```sh
$ pip install multi-agent-rlenv      # Basics
$ pip install multi-agent-rlenv[all] # With all optional dependecies
$ pip install multi-agent-rlenv[smac,overcooked] # Only SMAC & Overcooked
```

It aims to provide a simple and consistent interface for reinforcement learning environments by providing abstraction models such as `Observation`s or `Episode`s. `marlenv` provides adapters for popular libraries such as `gym` or `pettingzoo` and provides utility wrappers to add functionalities such as video recording or limiting the number of steps.

Almost every class is a dataclass to enable seemless serialiation with the `orjson` library.

# Fundamentals
## States & Observations
`MARLEnv.reset()` returns a pair of `(Observation, State)` and `MARLEnv.step()` returns a `Step`.

- `Observation` contains:
  - `data`: shape `[n_agents, *observation_shape]`
  - `available_actions`: boolean mask `[n_agents, n_actions]`
  - `extras`: extra features per agent (default shape `(n_agents, 0)`)
- `State` represents the environment state and can also carry `extras`.
- `Step` bundles `obs`, `state`, `reward`, `done`, `truncated`, and `info`.

Rewards are stored as `np.float32` arrays. Multi-objective envs use reward vectors with `reward_space.size > 1`.

## Extras
Extras are auxiliary features appended by wrappers (agent id, last action, time ratio, available actions, ...).
Wrappers that add extras must update both `extras_shape` and `extras_meanings` so downstream users can interpret them.
`State` extras should stay in sync with `Observation` extras when applicable.

# Environment catalog
`marlenv.catalog` exposes curated environments and lazily imports optional dependencies.

```python
from marlenv import catalog

env1 = catalog.overcooked().from_layout("scenario4")
env2 = catalog.lle().level(6)
env3 = catalog.DeepSea(mex_depth=5)
```

Catalog entries require their corresponding extras at install time (e.g., `marlenv[overcooked]`, `marlenv[lle]`).

# Wrappers & builders
Wrappers are composable through `RLEnvWrapper` and can be chained via `Builder` for fluent configuration.

```python
from marlenv import Builder
from marlenv.adapters import SMAC

env = (
    Builder(SMAC("3m"))
    .agent_id()
    .time_limit(20)
    .available_actions()
    .build()
)
```

Common wrappers include time limits, delayed rewards, masking available actions, and video recording.

# Using the library
## Adapters for existing libraries
Adapters normalize external APIs into `MARLEnv`:

```python
import marlenv

gym_env = marlenv.make("CartPole-v1", seed=25)

from marlenv.adapters import SMAC
smac_env = SMAC("3m", debug=True, difficulty="9")

from pettingzoo.sisl import pursuit_v4
from marlenv.adapters import PettingZoo
env = PettingZoo(pursuit_v4.parallel_env())
```

## Designing a custom environment
Create a custom environment by inheriting from `MARLEnv` and implementing `reset`, `step`, `get_observation`, and `get_state`.

```python
import numpy as np
from marlenv import MARLEnv, DiscreteSpace, Observation, State, Step

class CustomEnv(MARLEnv[DiscreteSpace]):
    def __init__(self):
        super().__init__(
            n_agents=3,
            action_space=DiscreteSpace.action(5).repeat(3),
            observation_shape=(4,),
            state_shape=(2,),
        )
        self.t = 0

    def reset(self):
        self.t = 0
        return self.get_observation(), self.get_state()

    def step(self, action):
        self.t += 1
        return Step(self.get_observation(), self.get_state(), reward=0.0, done=False)

    def get_observation(self):
        return Observation(np.zeros((3, 4), dtype=np.float32), self.available_actions())

    def get_state(self):
        return State(np.array([self.t, 0], dtype=np.float32))
```

# Related projects
- MARL: Collection of multi-agent reinforcement learning algorithms based on `marlenv` [https://github.com/yamoling/marl](https://github.com/yamoling/marl)
- Laser Learning Environment: a multi-agent gridworld that leverages `marlenv`'s capabilities [https://pypi.org/project/laser-learning-environment/](https://pypi.org/project/laser-learning-environment/)