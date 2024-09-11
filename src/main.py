import numpy as np
from marlenv import Builder, Transition, Observation
from marlenv.adapters import SMAC

o = Observation("a", [True, True], "state")

t = Transition(obs=o, action=np.array([1]), info={}, done=False, reward=[0.5], obs_=o, truncated=False)
