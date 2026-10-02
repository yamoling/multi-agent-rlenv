"""
Synchronous parallel environments that batch the observations of several `MARLEnv`s.

The objective is to perform a single (batched) forward pass of a policy for all the environments,
then dispatch the resulting actions to their respective environments.

Environments are *not* automatically reset: once an environment is done or truncated, it is frozen
until the next call to `ParallelMARLEnv.reset`, and the `mask` indicates which environments are still running.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from .env import MARLEnv
from .episode import Episode
from .observation import Observation
from .spaces import Space
from .state import State
from .step import Step

if TYPE_CHECKING:
    from torch import Tensor  # pyright: ignore[reportMissingImports]

A = TypeVar("A")


@dataclass(eq=False)
class ParallelObservation:
    """Batch of observations, one per parallel environment."""

    data: npt.NDArray[np.float32]
    """Shape `[n_envs, n_agents, *obs_shape]`"""
    extras: npt.NDArray[np.float32]
    """Shape `[n_envs, n_agents, *extras_shape]`"""
    available_actions: npt.NDArray[np.bool]
    """Shape `[n_envs, n_agents, n_actions]`"""

    @staticmethod
    def stack(observations: Sequence[Observation]) -> "ParallelObservation":
        return ParallelObservation(
            data=np.stack([o.data for o in observations]),
            extras=np.stack([o.extras for o in observations]),
            available_actions=np.stack([o.available_actions for o in observations]),
        )

    @property
    def n_envs(self) -> int:
        return self.data.shape[0]

    @property
    def n_agents(self) -> int:
        return self.data.shape[1]

    def __len__(self):
        return self.n_envs

    def __getitem__(self, env_index: int) -> Observation:
        """The observation of the environment at the given index."""
        return Observation(self.data[env_index], self.available_actions[env_index], self.extras[env_index])

    def as_tensors(self, device=None, *, actions: bool = False) -> "tuple[Tensor, ...]":
        """
        Convert the batch to tensors of shape `[n_envs, n_agents, ...]`: `(data, extras)`, or
        `(data, extras, available_actions)` if `actions` is True.
        """
        import torch  # pyright: ignore[reportMissingImports]

        arrays: list[npt.NDArray] = [self.data, self.extras]
        if actions:
            arrays.append(self.available_actions)
        return tuple(torch.from_numpy(a).to(device, non_blocking=True) for a in arrays)


@dataclass(eq=False)
class ParallelState:
    """Batch of states, one per parallel environment. Only supports numpy states."""

    data: npt.NDArray[np.float32]
    """Shape `[n_envs, *state_shape]`"""
    extras: npt.NDArray[np.float32]
    """Shape `[n_envs, *state_extras_shape]`"""

    @staticmethod
    def stack(states: Sequence[State]) -> "ParallelState":
        return ParallelState(
            data=np.stack([s.data for s in states]),
            extras=np.stack([s.extras for s in states]),
        )

    @property
    def n_envs(self) -> int:
        return self.data.shape[0]

    def __len__(self):
        return self.n_envs

    def __getitem__(self, env_index: int) -> State:
        """The state of the environment at the given index."""
        return State(self.data[env_index], self.extras[env_index])

    def as_tensors(self, device=None) -> "tuple[Tensor, Tensor]":
        """Convert the batch to tensors of shape `[n_envs, *state_shape]` and `[n_envs, *state_extras_shape]`."""
        import torch  # pyright: ignore[reportMissingImports]

        data = torch.from_numpy(self.data).to(device, non_blocking=True)
        extras = torch.from_numpy(self.extras).to(device, non_blocking=True)
        return data, extras


@dataclass(eq=False)
class ParallelStep:
    """
    Result of a step in a `ParallelMARLEnv`.

    Each environment has its own `done` and `truncated` flags. These flags are sticky: once an environment
    has finished, it is no longer stepped and its flags remain set until the next reset. The parallel step
    is terminal when all the environments have finished.
    """

    action: npt.NDArray
    """The actions given to `ParallelMARLEnv.step`, with shape `[n_envs, ...]`."""
    obs: ParallelObservation
    """The new observations. Finished environments keep their last observation."""
    state: ParallelState
    """The new states. Finished environments keep their last state."""
    reward: npt.NDArray[np.float32]
    """Shape `[n_envs, *reward_shape]`. The reward is zero for the environments that have not been stepped (`~mask`)."""
    done: npt.NDArray[np.bool]
    """Shape `[n_envs]`. Whether each environment is done."""
    truncated: npt.NDArray[np.bool]
    """Shape `[n_envs]`. Whether each environment has been truncated."""
    mask: npt.NDArray[np.bool]
    """Shape `[n_envs]`. Whether each environment was running and has actually been stepped in this step."""
    steps: list[Step | None]
    """The `Step` of each environment, or `None` if the environment has not been stepped (`~mask`)."""
    infos: list[dict[str, Any]]
    """The info of each environment (empty for the environments that have not been stepped)."""

    @property
    def n_envs(self) -> int:
        return len(self.steps)

    @property
    def is_terminal(self) -> bool:
        """Whether all the environments are done or truncated."""
        return bool(np.all(self.done | self.truncated))


class ParallelMARLEnv(Generic[A]):
    """
    Synchronously runs several environments with the same inputs and outputs and batches their results.

    ```python
    env = ParallelMARLEnv.from_factory(lambda: marlenv.make("CartPole-v1"), n_envs=8)
    obs, state = env.reset()
    step = env.step(policy(obs))  # obs.data has shape [8, n_agents, *obs_shape]
    while not step.is_terminal:
        step = env.step(policy(step.obs))  # Actions of finished environments (`~step.mask`) are ignored
    ```
    """

    envs: list[MARLEnv[A]]

    def __init__(self, envs: Sequence[MARLEnv[A]]):
        if len(envs) == 0:
            raise ValueError("ParallelMARLEnv requires at least one environment")
        if len({id(env) for env in envs}) != len(envs):
            raise ValueError("The same environment instance has been given multiple times. Use `ParallelMARLEnv.from_factory` instead.")
        for i, env in enumerate(envs[1:], start=1):
            if not env.has_same_inouts(envs[0]):
                raise ValueError(f"Environment {i} does not have the same inputs and outputs as environment 0")
        self.envs = list(envs)
        n = len(self.envs)
        self._running = np.zeros(n, dtype=np.bool)
        self._done = np.zeros(n, dtype=np.bool)
        self._truncated = np.zeros(n, dtype=np.bool)
        self._obs: list[Observation] = []
        self._states: list[State] = []

    @staticmethod
    def from_factory(factory: Callable[[], MARLEnv[A]], n_envs: int) -> "ParallelMARLEnv[A]":
        """Create `n_envs` environments with the given factory."""
        return ParallelMARLEnv([factory() for _ in range(n_envs)])

    @property
    def n_envs(self) -> int:
        return len(self.envs)

    @property
    def name(self) -> str:
        return f"Parallel-{self.n_envs}x{self.envs[0].name}"

    @property
    def n_agents(self) -> int:
        return self.envs[0].n_agents

    @property
    def n_actions(self) -> int:
        return self.envs[0].n_actions

    @property
    def action_space(self) -> Space[A]:
        return self.envs[0].action_space

    @property
    def reward_space(self):
        return self.envs[0].reward_space

    @property
    def n_objectives(self) -> int:
        return self.envs[0].n_objectives

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self.envs[0].observation_shape

    @property
    def extras_shape(self) -> tuple[int, ...]:
        return self.envs[0].extras_shape

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.envs[0].state_shape

    @property
    def state_extras_shape(self) -> tuple[int, ...]:
        return self.envs[0].state_extras_shape

    @property
    def mask(self) -> npt.NDArray[np.bool]:
        """Shape `[n_envs]`. Whether each environment is still running, i.e. will be stepped by the next call to `step`."""
        return self._running.copy()

    def seed(self, seed_value: int):
        """Seed the `i`-th environment with `seed_value + i`."""
        for i, env in enumerate(self.envs):
            env.seed(seed_value + i)

    def reset(self, *, seed: int | None = None) -> tuple[ParallelObservation, ParallelState]:
        """Reset all the environments. If a seed is given, the `i`-th environment is reset with `seed + i`."""
        self._obs, self._states = [], []
        for i, env in enumerate(self.envs):
            obs, state = env.reset(seed=None if seed is None else seed + i)
            self._obs.append(obs)
            self._states.append(state)
        self._running[:] = True
        self._done[:] = False
        self._truncated[:] = False
        return self.get_observation(), self.get_state()

    def step(self, actions: Sequence[A] | npt.ArrayLike) -> ParallelStep:
        """
        Dispatch `actions[i]` to the `i`-th environment if it is still running. The actions of the
        environments that have already finished are ignored.
        """
        if not self._running.any():
            raise RuntimeError("All the environments have finished (or have never been reset). Call `reset()` first.")
        actions = np.asarray(actions)
        if len(actions) != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} actions (one per environment) but got {len(actions)}")
        mask = self._running.copy()
        rewards = np.zeros((self.n_envs, *self.reward_space.shape), dtype=np.float32)
        steps: list[Step | None] = [None] * self.n_envs
        infos: list[dict[str, Any]] = [{} for _ in range(self.n_envs)]
        for i in np.nonzero(mask)[0]:
            step = self.envs[i].step(actions[i])
            steps[i] = step
            infos[i] = step.info
            rewards[i] = step.reward
            self._obs[i] = step.obs
            self._states[i] = step.state
            self._done[i] = step.done
            self._truncated[i] = step.truncated
            self._running[i] = not step.is_terminal
        return ParallelStep(
            action=actions,
            obs=self.get_observation(),
            state=self.get_state(),
            reward=rewards,
            done=self._done.copy(),
            truncated=self._truncated.copy(),
            mask=mask,
            steps=steps,
            infos=infos,
        )

    def get_observation(self) -> ParallelObservation:
        """The current (or last, for finished environments) observation of each environment."""
        return ParallelObservation.stack(self._obs)

    def get_state(self) -> ParallelState:
        """The current (or last, for finished environments) state of each environment."""
        return ParallelState.stack(self._states)

    def available_actions(self) -> npt.NDArray[np.bool]:
        """Shape `[n_envs, n_agents, n_actions]`."""
        return np.stack([env.available_actions() for env in self.envs])

    def sample_action(self) -> npt.NDArray:
        """Sample an available action for every environment, with shape `[n_envs, ...]`."""
        return np.stack([np.asarray(env.sample_action()) for env in self.envs])

    def rollout(self, agent: Callable[[ParallelObservation], Sequence[A] | npt.ArrayLike], *, seed: int | None = None) -> list[Episode]:
        """Play one episode in every environment with a batched agent and return the episodes."""
        obs, state = self.reset(seed=seed)
        episodes = [Episode.new(obs[i], state[i]) for i in range(self.n_envs)]
        step = self.step(agent(obs))
        while not step.is_terminal:
            for episode, env_step in zip(episodes, step.steps):
                if env_step is not None:
                    episode.add(env_step)
            obs = step.obs
            step = self.step(agent(obs))
        for episode, env_step in zip(episodes, step.steps):
            if env_step is not None:
                episode.add(env_step)
        return episodes
