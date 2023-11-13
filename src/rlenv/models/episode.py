from dataclasses import dataclass
from serde import serde
from typing import Iterable
import numpy as np

from .metrics import Metrics
from .transition import Transition
from .observation import Observation


@serde
@dataclass
class Episode:
    """Episode model made of observations, actions, rewards, ..."""

    _observations: np.ndarray[np.float32]
    _extras: np.ndarray[np.float32]
    actions: np.ndarray[np.int64]
    rewards: np.ndarray[np.float32]
    _available_actions: np.ndarray[np.int64]
    states: np.ndarray[np.float32]
    actions_probs: np.ndarray[np.float32] | None
    metrics: Metrics
    episode_len: int
    is_finished: bool

    def padded(self, target_len: int) -> "Episode":
        """Copy of the episode, padded with zeros to the target length"""
        if target_len == self.episode_len:
            return self
        if target_len < self.episode_len:
            raise ValueError(
                f"Cannot pad episode to a smaller size: {target_len} < {self.episode_len}"
            )
        padding_size = target_len - self.episode_len
        obs = np.concatenate(
            [
                self._observations,
                np.zeros(
                    (padding_size, self.n_agents, self.obs_size), dtype=np.float32
                ),
            ]
        )
        extras_shape = list(self._extras.shape)
        extras_shape[0] = padding_size
        extras = np.concatenate(
            [self._extras, np.zeros(extras_shape, dtype=np.float32)]
        )
        actions = np.concatenate(
            [self.actions, np.zeros((padding_size, self.n_agents), dtype=np.int64)]
        )
        rewards_padding_shape = list(self.rewards.shape)
        rewards_padding_shape[0] = padding_size
        rewards = np.concatenate(
            [self.rewards, np.zeros(rewards_padding_shape, dtype=np.float32)]
        )
        availables = np.concatenate(
            [
                self._available_actions,
                np.ones(
                    (padding_size, self.n_agents, self.n_actions), dtype=np.float32
                ),
            ]
        )
        states = np.concatenate(
            [
                self.states,
                np.zeros((padding_size, *self.states.shape[1:]), dtype=np.float32),
            ]
        )
        return Episode(
            _observations=obs,
            actions=actions,
            rewards=rewards,
            states=states,
            metrics=self.metrics,
            episode_len=self.episode_len,
            _available_actions=availables,
            _extras=extras,
            actions_probs=None,
            is_finished=self.is_finished,
        )

    @property
    def mask(self) -> np.ndarray:
        """Get the mask for the current episide (when padded)"""
        mask = np.ones_like(self.rewards, dtype=np.float32)
        mask[self.episode_len :] = 0
        return mask

    @property
    def obs(self) -> np.ndarray:
        """The observations"""
        return self._observations[:-1]

    @property
    def obs_(self) -> np.ndarray:
        """The next observations"""
        return self._observations[1:]

    @property
    def extras(self) -> np.ndarray[np.float32]:
        """Get the extra features"""
        return self._extras[:-1]

    @property
    def extras_(self) -> np.ndarray[np.float32]:
        """Get the next extra features"""
        return self._extras[1:]

    @property
    def n_agents(self) -> int:
        """The number of agents in the episode"""
        return self._observations.shape[1]

    @property
    def obs_size(self) -> int:
        """The observation size"""
        return self._observations.shape[2]

    @property
    def n_actions(self) -> int:
        """The number of actions"""
        return self._available_actions.shape[2]

    @property
    def available_actions(self) -> np.ndarray[np.int64]:
        """The available actions"""
        return self._available_actions[:-1]

    @property
    def available_actions_(self) -> np.ndarray[np.int64]:
        """The next available actions"""
        return self._available_actions[1:]

    @property
    def dones(self) -> np.ndarray:
        """The done flags for each transition"""
        dones = np.zeros_like(self.rewards, dtype=np.float32)
        if self.is_finished:
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
                    state=self.states[i],
                ),
                action=self.actions[i],
                reward=self.rewards[i],
                done=self.dones[i],
                info={},
                obs_=Observation(
                    data=self._observations[i + 1],
                    available_actions=self._available_actions[i + 1],
                    extras=self._extras[i + 1],
                    state=self.states[i + 1],
                ),
                truncated=not self.is_finished and i == self.episode_len - 1,
            )

    def __iter__(self) -> Iterable[Transition]:
        return self.transitions()

    @staticmethod
    def agregate_metrics(episodes: list["Episode"]) -> Metrics:
        """Agregate metrics of a list of episodes (min, max, avg)"""
        metrics = [e.metrics for e in episodes]
        return Metrics.agregate(metrics)

    def __len__(self):
        return self.episode_len

    @property
    def score(self) -> float:
        """The episode score (sum of all rewards)"""
        return self.metrics["score"].value

    def to_json(self) -> dict:
        """Creates a json serialisable dictionary"""
        return {
            "obs": self._observations.tolist(),
            "extras": self._extras.tolist(),
            "actions": self.actions.tolist(),
            "rewards": self.rewards.tolist(),
            "available_actions": self._available_actions.tolist(),
            "states": self.states.tolist(),
            "metrics": self.metrics.to_json(),
        }

    def compute_returns(self, discount: float = 1.0) -> np.ndarray[np.float32]:
        """Compute the returns (discounted sum of future rewards) of the episode at each time step"""
        returns = np.zeros_like(self.rewards)
        returns[-1] = self.rewards[-1]
        for t in range(len(self.rewards) - 2, -1, -1):
            returns[t] = self.rewards[t] + discount * returns[t + 1]
        return returns


class EpisodeBuilder:
    """EpisodeBuilder gives away the complexity of building an Episode to another class"""

    def __init__(self) -> None:
        self.observations = []
        self.extras = []
        self.actions = []
        self.rewards = []
        self.available_actions = []
        self.states = []
        self.action_probs = []
        self.episode_len = 0
        self.metrics = Metrics()
        self._done = False
        self._truncated = False

    @property
    def is_finished(self) -> bool:
        return self._done or self._truncated

    @property
    def t(self) -> int:
        """The current time step (i.e. the current episode length)"""
        return self.episode_len

    def add(self, transition: Transition):
        """Add a transition to the episode"""
        self.episode_len += 1
        self.observations.append(transition.obs.data)
        self.extras.append(transition.obs.extras)
        self.actions.append(transition.action)
        self.rewards.append(transition.reward)
        self.available_actions.append(transition.obs.available_actions)
        self.states.append(transition.obs.state)
        if transition.is_terminal:
            # Only set the truncated flag if the episode is not done (both could happen with a time limit)
            self._truncated = transition.truncated
            self._done = transition.done
            # Add metrics that can be plotted
            for key, value in transition.info.items():
                if isinstance(value, bool):
                    value = int(value)
                self.metrics[key] = value
            self.observations.append(transition.obs_.data)
            self.extras.append(transition.obs_.extras)
            self.available_actions.append(np.ones_like(self.available_actions[-1]))
            self.states.append(np.zeros_like(self.states[-1]))

    def build(self, extra_metrics: dict[str, float] = None) -> Episode:
        """Build the Episode"""
        assert (
            self.is_finished
        ), "Cannot build an episode that is not finished. Set truncated=True when adding the last transition of the episode."
        self.metrics["score"] = float(np.sum(self.rewards))
        self.metrics["episode_length"] = self.episode_len
        if extra_metrics is not None:
            self.metrics.update(extra_metrics)
        return Episode(
            _observations=np.array(self.observations),
            _extras=np.array(self.extras),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            states=np.array(self.states),
            metrics=self.metrics,
            episode_len=self.episode_len,
            _available_actions=np.array(self.available_actions),
            actions_probs=np.array(self.action_probs),
            is_finished=self._done,
        )

    def __len__(self) -> int:
        return self.episode_len
