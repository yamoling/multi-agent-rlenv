"""
`marlenv` is a strongly typed library for multi-agent and multi-objective
reinforcement learning.

Install with:
```sh
$ pip install multi-agent-rlenv
$ pip install multi-agent-rlenv[all]
```

The package provides:
- A consistent `MARLEnv` interface and typed models (`Observation`, `Step`, `Episode`, ...).
- Adapters for external ecosystems such as Gymnasium, PettingZoo, and SMAC.
- Composable wrappers and a `Builder` API for common environment transformations.

For full documentation and examples, see `README.md` and:
https://yamoling.github.io/multi-agent-rlenv
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multi-agent-rlenv")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for CI


from . import adapters, catalog, models, wrappers
from .adapters import make
from .env_builder import Builder
from .models import (
    ContinuousMARLEnv,
    ContinuousSpace,
    DiscreteMARLEnv,
    DiscreteSpace,
    Episode,
    MARLEnv,
    MultiDiscreteSpace,
    Observation,
    Space,
    State,
    Step,
    Transition,
    spaces,
)
from .wrappers import RLEnvWrapper

__all__ = [
    "models",
    "make",
    "catalog",
    "wrappers",
    "adapters",
    "spaces",
    "Builder",
    "MARLEnv",
    "DiscreteMARLEnv",
    "ContinuousMARLEnv",
    "Step",
    "State",
    "Observation",
    "Episode",
    "Transition",
    "DiscreteSpace",
    "ContinuousSpace",
    "RLEnvWrapper",
    "Space",
    "MultiDiscreteSpace",
]
