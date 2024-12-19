from dataclasses import dataclass
from typing import Iterable, Optional, TypeVar, Generic
import numpy as np
import numpy.typing as npt
from functools import cached_property

from .transition import Transition
from .observation import Observation
from .state import State

A = TypeVar("A")


@dataclass
class Episode(Generic[A]):
    """Episode model made of observations, actions, rewards, ..."""

    _observations: list[npt.NDArray[np.float32]]
    _extras: list[npt.NDArray[np.float32]]
    actions: list[npt.NDArray]
    rewards: list[npt.NDArray[np.float32]]
    _available_actions: list[npt.NDArray[np.bool_]]
    _states: list[npt.NDArray[np.float32]]
    _states_extras: list[npt.NDArray[np.float32]]
    actions_probs: list[npt.NDArray[np.float32]]
    metrics: dict[str, float]
    episode_len: int
    is_done: bool = False
    is_truncated: bool = False
    """Whether the episode did reach a terminal state (different from truncated)"""

    @staticmethod
    def new(obs: Observation, state: State, metrics: Optional[dict[str, float]] = None) -> "Episode":
        if metrics is None:
            metrics = {}
        return Episode(
            _observations=[obs.data],
            _extras=[obs.extras],
            _states=[state.data],
            _states_extras=[state.extras],
            _available_actions=[obs.available_actions],
            actions=[],
            rewards=[],
            actions_probs=[],
            metrics=metrics,
            episode_len=0,
            is_done=False,
            is_truncated=False,
        )

    def padded(self, target_len: int) -> "Episode":
        """Copy of the episode, padded with mock items to the target length"""
        if target_len == self.episode_len:
            return self
        if target_len < self.episode_len:
            raise ValueError(f"Cannot pad episode to a smaller size: {target_len} < {self.episode_len}")
        padding_size = target_len - self.episode_len
        obs = self._observations + [self._observations[0]] * padding_size
        extras = self._extras + [self._extras[0]] * padding_size
        actions = self.actions + [self.actions[0]] * padding_size
        rewards = self.rewards + [self.rewards[0]] * padding_size
        availables = self._available_actions + [self._available_actions[0]] * padding_size
        states = self._states + [self._states[0]] * padding_size
        states_extras = self._states_extras + [self._states_extras[0]] * padding_size
        return Episode(
            _observations=obs,
            _extras=extras,
            _states=states,
            _states_extras=states_extras,
            _available_actions=availables,
            actions=actions,
            rewards=rewards,
            metrics=self.metrics,
            episode_len=self.episode_len,
            actions_probs=[],
            is_done=self.is_done,
        )

    @cached_property
    def states(self):
        """The states"""
        return self._states[:-1]

    @cached_property
    def states_extras(self):
        """The extra features of the states"""
        return self._states_extras[:-1]

    @cached_property
    def next_states(self):
        """The next states"""
        return self._states[1:]

    @cached_property
    def next_states_extras(self):
        """The next extra features of the states"""
        return self._states_extras[1:]

    @cached_property
    def mask(self):
        """Get the mask for the current episide (when padded)"""
        mask = np.ones_like(self.rewards, dtype=np.float32)
        mask[self.episode_len :] = 0
        return mask

    @cached_property
    def obs(self):
        """The observations"""
        return self._observations[:-1]

    @cached_property
    def next_obs(self):
        """The next observations"""
        return self._observations[1:]

    @cached_property
    def extras(self):
        """Get the extra features"""
        return self._extras[:-1]

    @cached_property
    def next_extras(self):
        """Get the next extra features"""
        return self._extras[1:]

    @cached_property
    def n_agents(self):
        """The number of agents in the episode"""
        return self._extras[0].shape[1]

    @cached_property
    def n_actions(self):
        """The number of actions"""
        return len(self._available_actions[0][0])

    @cached_property
    def available_actions(self):
        """The available actions"""
        return self._available_actions[:-1]

    @cached_property
    def next_available_actions(self):
        """The next available actions"""
        return self._available_actions[1:]

    @cached_property
    def dones(self):
        """The done flags for each transition"""
        dones = np.zeros_like(self.rewards, dtype=np.float32)
        if self.is_done:
            dones[self.episode_len - 1 :] = 1.0
        return dones

    @property
    def is_finished(self) -> bool:
        """Whether the episode is done or truncated"""
        return self.is_done or self.is_truncated

    def transitions(self) -> Iterable[Transition]:
        """The transitions that compose the episode"""
        for i in range(self.episode_len):
            yield Transition(
                obs=Observation(
                    data=self._observations[i],
                    available_actions=self._available_actions[i],
                    extras=self._extras[i],
                ),
                state=State(data=self._states[i], extras=self._states_extras[i]),
                action=self.actions[i],
                reward=self.rewards[i],
                done=bool(self.dones[i]),
                info={},
                next_obs=Observation(
                    data=self._observations[i + 1],
                    available_actions=self._available_actions[i + 1],
                    extras=self._extras[i + 1],
                ),
                next_state=State(data=self._states[i + 1], extras=self._states_extras[i + 1]),
                truncated=not self.is_done and i == self.episode_len - 1,
            )

    def __iter__(self) -> Iterable[Transition]:
        return self.transitions()

    def __len__(self):
        return self.episode_len

    @cached_property
    def score(self) -> float:
        """The episode score (sum of all rewards)"""
        return self.metrics["score"]

    def compute_returns(self, discount: float = 1.0):
        """Compute the returns (discounted sum of future rewards) of the episode at each time step"""
        returns = np.zeros_like(self.rewards)
        returns[-1] = self.rewards[-1]
        for t in range(len(self.rewards) - 2, -1, -1):
            returns[t] = self.rewards[t] + discount * returns[t + 1]
        return returns

    def add(self, transition: Transition[A]):
        """Add a transition to the episode"""
        self.episode_len += 1
        self._observations.append(transition.next_obs.data)
        self._extras.append(transition.next_obs.extras)
        self._available_actions.append(transition.next_obs.available_actions)
        self._states.append(transition.next_state.data)
        self._states_extras.append(transition.next_state.extras)
        match transition.action:
            case np.ndarray() as action:
                self.actions.append(action)
            case other:
                action = np.array(other)
                self.actions.append(action)
        self.rewards.append(transition.reward)
        if transition.action_probs is not None:
            self.actions_probs.append(transition.action_probs)
        if transition.is_terminal:
            # Only set the truncated flag if the episode is not done (both could happen with a time limit)
            self.is_truncated = transition.truncated
            self.is_done = transition.done
            # Add metrics that can be plotted
            for key, value in transition.info.items():
                if isinstance(value, bool):
                    value = int(value)
                self.metrics[key] = value
            self.metrics["episode_len"] = self.episode_len

            rewards = np.array(self.rewards)
            scores = np.sum(rewards, axis=0)
            for i, s in enumerate(scores):
                self.metrics[f"score-{i}"] = float(s)

    def add_metrics(self, metrics: dict[str, float]):
        """Add metrics to the episode"""
        self.metrics.update(metrics)
