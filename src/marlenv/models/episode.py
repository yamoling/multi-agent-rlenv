from dataclasses import dataclass
from typing import Optional, Iterable
import numpy as np
import numpy.typing as npt
from functools import cached_property

from .transition import Transition, RewardType
from .observation import Observation
from .state import State


@dataclass
class Episode:
    """Episode model made of observations, actions, rewards, ..."""

    _observations: npt.NDArray[np.float32]
    _extras: npt.NDArray[np.float32]
    actions: np.ndarray
    rewards: npt.NDArray[np.float32]
    _available_actions: npt.NDArray[np.bool_]
    _states: npt.NDArray[np.float32]
    _states_extras: npt.NDArray[np.float32]
    actions_probs: npt.NDArray[np.float32] | None
    metrics: dict[str, float]
    episode_len: int
    is_done: bool
    """Whether the episode did reach a terminal state (different from truncated)"""

    def padded(self, target_len: int) -> "Episode":
        """Copy of the episode, padded with zeros to the target length"""
        if target_len == self.episode_len:
            return self
        if target_len < self.episode_len:
            raise ValueError(f"Cannot pad episode to a smaller size: {target_len} < {self.episode_len}")
        padding_size = target_len - self.episode_len
        padding = np.zeros((padding_size, *self._observations.shape[1:]), dtype=np.float32)
        obs = np.concatenate([self._observations, padding])
        extras_padding_shape = (padding_size, *self._extras.shape[1:])
        extras = np.concatenate([self._extras, np.zeros(extras_padding_shape, dtype=np.float32)])
        actions = np.concatenate([self.actions, np.zeros((padding_size, self.n_agents), dtype=self.actions.dtype)])
        rewards_padding_shape = (padding_size, *self.rewards.shape[1:])
        rewards = np.concatenate([self.rewards, np.zeros(rewards_padding_shape, dtype=np.float32)])
        availables = np.concatenate([self._available_actions, np.full((padding_size, self.n_agents, self.n_actions), True)])
        states = np.concatenate([self._states, np.zeros((padding_size, *self._states.shape[1:]), dtype=np.float32)])
        states_extras = np.concatenate([self._states_extras, np.zeros((padding_size, *self._states_extras.shape[1:]), dtype=np.float32)])
        return Episode(
            _observations=obs,
            actions=actions,
            rewards=rewards,
            _states=states,
            _states_extras=states_extras,
            metrics=self.metrics,
            episode_len=self.episode_len,
            _available_actions=availables,
            _extras=extras,
            actions_probs=None,
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
        return self._observations.shape[1]

    @cached_property
    def n_actions(self):
        """The number of actions"""
        return self._available_actions.shape[2]

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


class EpisodeBuilder:
    """EpisodeBuilder gives away the complexity of building an Episode to another class"""

    def __init__(self):
        self.observations = list[np.ndarray]()
        self.extras = list[np.ndarray]()
        self.actions = list[np.ndarray]()
        self.rewards = list()
        self.available_actions = list[np.ndarray]()
        self.states = list[np.ndarray]()
        self.state_extras = list[np.ndarray]()
        self.action_probs = list[np.ndarray]()
        self.episode_len = 0
        self.metrics = {}
        self._done = False
        self._truncated = False

    @property
    def is_finished(self) -> bool:
        return self._done or self._truncated

    @property
    def t(self) -> int:
        """The current time step (i.e. the current episode length)"""
        return self.episode_len

    def add(self, transition: Transition[RewardType, np.ndarray, np.ndarray]):
        """Add a transition to the episode"""
        self.episode_len += 1
        self.observations.append(transition.obs.data)
        self.extras.append(transition.obs.extras)
        self.actions.append(transition.action)
        self.rewards.append(transition.reward)
        self.available_actions.append(transition.obs.available_actions)
        self.states.append(transition.state.data)
        self.state_extras.append(transition.state.extras)
        if transition.action_probs is not None:
            self.action_probs.append(transition.action_probs)
        if transition.is_terminal:
            # Only set the truncated flag if the episode is not done (both could happen with a time limit)
            self._truncated = transition.truncated
            self._done = transition.done
            # Add metrics that can be plotted
            for key, value in transition.info.items():
                if isinstance(value, bool):
                    value = int(value)
                self.metrics[key] = value
            self.observations.append(transition.next_obs.data)
            self.extras.append(transition.next_obs.extras)
            self.available_actions.append(transition.next_obs.available_actions)
            self.states.append(transition.next_state.data)
            self.state_extras.append(transition.next_state.extras)

    def build(self, extra_metrics: Optional[dict[str, float]] = None) -> Episode:
        """Build the Episode"""
        assert (
            self.is_finished
        ), "Cannot build an episode that is not finished. Set truncated=True when adding the last transition of the episode."
        self.metrics["score"] = float(np.sum(self.rewards))
        self.metrics["episode_length"] = self.episode_len
        if extra_metrics is not None:
            self.metrics.update(extra_metrics)
        action_probs = None
        if len(self.action_probs) > 0:
            action_probs = np.array(self.action_probs, dtype=np.float32)
        return Episode(
            _observations=np.array(self.observations, dtype=np.float32),
            _extras=np.array(self.extras, dtype=np.float32),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards, np.float32),
            _states=np.array(self.states),
            _states_extras=np.array(self.state_extras),
            metrics=self.metrics,
            episode_len=self.episode_len,
            _available_actions=np.array(self.available_actions),
            actions_probs=action_probs,
            is_done=self._done,
        )

    def __len__(self) -> int:
        return self.episode_len
