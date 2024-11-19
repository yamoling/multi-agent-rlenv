import numpy as np
from marlenv import Builder, Transition, Observation
from marlenv.adapters import SMAC
from marlenv.mock_env import DiscreteMockEnv


env = DiscreteMockEnv()
env.step(np.array([0.1, 20.5]))
