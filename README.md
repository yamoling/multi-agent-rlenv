# `marlenv` - A unified framework for multi-agent reinforcement learning

**Documentation: [https://yamoling.github.io/multi-agent-rlenv](https://yamoling.github.io/multi-agent-rlenv)**

`marlenv` is a strongly typed library for multi-agent and multi-objective reinforcement learning.

Install the library with:

```sh
$ pip install multi-agent-rlenv      # Basics
$ pip install multi-agent-rlenv[all] # With all optional dependencies
$ pip install multi-agent-rlenv[smac,overcooked] # Only SMAC & Overcooked
```

It aims to provide a simple and consistent interface for reinforcement learning environments by providing abstraction models such as `Observation`s or `Episode`s. `marlenv` provides adapters for popular libraries such as `gym` or `pettingzoo` and provides utility wrappers to add functionalities such as video recording or limiting the number of steps.

Most classes are dataclasses, which makes serialization straightforward (for example with `orjson`).

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
env3 = catalog.DeepSea(max_depth=5)
env4 = catalog.connect_n()(width=7, height=6, n=4)
```

Catalog entries require their corresponding extras at install time (e.g., `multi-agent-rlenv[overcooked]`, `multi-agent-rlenv[lle]`).

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

# Parallel environments

`ParallelMARLEnv` steps several environments with the same inputs and outputs, and batches their observations so that a policy can select the actions of all environments in a single forward pass.

```python
from marlenv import Builder, ParallelMARLEnv
from marlenv.catalog import DeepSea, EnvPool

env = ParallelMARLEnv.from_factory(lambda: Builder(DeepSea(max_depth=5)).time_limit(20).build(), n_envs=8)
# Or, from an existing pool: env = EnvPool([...]).to_parallel()
obs, state = env.reset(seed=0)  # obs.data has shape [n_envs, n_agents, *observation_shape]
while True:
    step = env.step(policy(obs))  # One action per environment: [n_envs, ...]
    if step.is_terminal:  # All environments are done or truncated
        break
    obs = step.obs
```

Environments are not reset automatically: once an environment is done or truncated, it is no longer stepped and its actions are ignored until the next `reset()`.
Each `ParallelStep` has per-environment `done` and `truncated` flags, a `mask` of the environments that were actually stepped, and the individual `Step` of each environment in `steps` (`None` for masked environments).

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

For deterministic behavior, seed the environment:

```python
env.seed(123)
obs, state = env.reset()
```

## Designing a custom environment

Create a custom environment by inheriting from `MARLEnv` and implementing `reset`, `step`, `get_observation`, and `get_state`.

```python
import numpy as np
from marlenv import MARLEnv, DiscreteSpace, MultiDiscreteSpace, Observation, State, Step

class CustomEnv(MARLEnv[MultiDiscreteSpace]):
    def __init__(self):
        super().__init__(
            n_agents=3,
            action_space=DiscreteSpace.action(5).repeat(3),
            observation_shape=(4,),
            state_shape=(2,),
        )
        self.t = 0

    def reset(self, * seed:int|None=None):
        if seed is not None:
            self.seed(seed)
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
