from pettingzoo import ParallelEnv
from gymnasium import spaces  # pettingzoo uses gymnasium spaces
from marlenv.models import MARLEnv, Observation, ActionSpace, DiscreteActionSpace, ContinuousActionSpace
import numpy as np
import numpy.typing as npt


class PettingZoo(MARLEnv[ActionSpace, npt.NDArray[np.float32], npt.NDArray[np.float32], float]):
    def __init__(self, env: ParallelEnv):
        env.reset()
        aspace = env.action_space(env.possible_agents[0])

        match aspace:
            case spaces.Discrete() as s:
                space = DiscreteActionSpace(env.num_agents, int(s.n))

            case spaces.Box() as s:
                low = s.low.astype(np.float32)
                high = s.high.astype(np.float32)
                if not isinstance(low, np.ndarray):
                    low = np.full(s.shape, s.low, dtype=np.float32)
                if not isinstance(high, np.ndarray):
                    high = np.full(s.shape, s.high, dtype=np.float32)
                space = ContinuousActionSpace(env.num_agents, low, high=high)
            case other:
                raise NotImplementedError(f"Action space {other} not supported")

        obs_space = env.observation_space(env.possible_agents[0])
        if obs_space.shape is None:
            raise NotImplementedError("Only discrete observation spaces are supported")
        self._env = env
        super().__init__(space, obs_space.shape, self.get_state().shape)
        self.agents = env.possible_agents

    def get_state(self):
        try:
            return self._env.state()
        except NotImplementedError:
            return np.array([0])

    def step(self, actions: npt.NDArray[np.int64]):
        action_dict = dict(zip(self.agents, actions))
        obs, reward, term, trunc, info = self._env.step(action_dict)
        obs_data = np.array([v for v in obs.values()])
        reward = np.sum([r for r in reward.values()], keepdims=True)
        observation = Observation(obs_data, self.available_actions(), self.get_state())
        return observation, reward, any(term.values()), any(trunc.values()), info

    def reset(self) -> Observation:
        obs = self._env.reset()[0]
        obs_data = np.array([v for v in obs.values()])
        return Observation(obs_data, self.available_actions(), self.get_state())

    def seed(self, seed_value: int):
        self._env.reset(seed=seed_value)

    def render(self, *_):
        return self._env.render()
