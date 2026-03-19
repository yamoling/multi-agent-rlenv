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

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("multi-agent-rlenv")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for CI


from . import models
from .models import (
    spaces,
    MARLEnv,
    State,
    Step,
    Observation,
    Episode,
    Transition,
    DiscreteSpace,
    ContinuousSpace,
    Space,
    MultiDiscreteSpace,
)


from . import wrappers
from . import adapters
from .env_builder import Builder
from .wrappers import RLEnvWrapper
from .mock_env import DiscreteMockEnv, DiscreteMOMockEnv
from . import catalog
from .adapters import make

__all__ = [
    "models",
    "make",
    "catalog",
    "wrappers",
    "adapters",
    "spaces",
    "Builder",
    "MARLEnv",
    "Step",
    "State",
    "Observation",
    "Episode",
    "Transition",
    "DiscreteSpace",
    "ContinuousSpace",
    "DiscreteMockEnv",
    "DiscreteMOMockEnv",
    "RLEnvWrapper",
    "Space",
    "MultiDiscreteSpace",
]
