"""
Environment catalog for `marlenv`.

This submodule exposes curated environments and provides lazy imports for optional
dependencies to keep the base install lightweight. Use the catalog to construct
environments without importing their packages directly.

Examples:
```python
from marlenv import catalog

env1 = catalog.DeepSea(mex_depth=5)
env2 = catalog.CoordinatedGrid()
env3 = catalog.connect_n()(width=7, height=6, n_to_connect=4)
env4 = catalog.smac()("3m")
```

Optional entries such as `smac`, `lle`, and `overcooked` require installing their
corresponding extras (e.g., `marlenv[smac]`, `marlenv[lle]`, `marlenv[overcooked]`).
"""

from .deepsea import DeepSea
from .matrix_game import MatrixGame
from .coordinated_grid import CoordinatedGrid


__all__ = ["smac", "DeepSea", "lle", "overcooked", "MatrixGame", "connect_n", "CoordinatedGrid"]


def smac():
    """
    Quick loading of the SMAC adapter, if smacv2 is instaleld.

    **Example:**
    ```python
    env = marlenv.catalog.smac()("3m")
    obs, state = env.reset()
    ```
    """
    from marlenv.adapters import SMAC

    return SMAC


def lle():
    """
    If installed, returns the LLE class that implements the `MARLEnv` class.
    Refer to the LLE's documentation for further information.

    **Example:**
    ```python
    env = marlenv.catalog.lle().level(6).build()
    obs, state = env.reset()
    ```
    """
    from lle import LLE  # pyright: ignore[reportMissingImports]

    return LLE


def overcooked():
    """
    If installed, returns the Overcooked class that implements the `MARLEnv` class.
    Refer to Overcooked's documentation for further information.

    **Example:**
    ```python
    env = marlenv.catalog.overcooked().from_layout("asymmetric_advantages")
    obs, state = env.reset()
    ```
    """
    from overcooked import Overcooked  # pyright: ignore[reportMissingImports]

    return Overcooked


def connect_n():
    """
    Returs the `ConnectN` class if `pygame` is installed.
    """

    from .connectn import ConnectN

    return ConnectN
