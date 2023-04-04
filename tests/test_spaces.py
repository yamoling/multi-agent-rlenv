import numpy as np
from rlenv.models import DiscreteActionSpace, ContinuousActionSpace


def test_discrete_action_space():
    s = DiscreteActionSpace(2, 3)
    available_actions = np.array([[1, 1, 0], [1, 0, 1]])
    for _ in range(100):
        actions = s.sample(available_actions)
        assert actions.shape == (2, )
        assert actions[0] in [0, 1]
        assert actions[1] in [0, 2]

        actions = s.sample()
        assert actions.shape == (2, )
        assert actions[0] in [0, 1, 2]
        assert actions[1] in [0, 1, 2]

        
def test_continuous_action_space():
    s = ContinuousActionSpace(2, 3, low=[0., -1., 0.], high=[1., 1., 2.])
    for _ in range(100):
        actions = s.sample()
        assert actions.shape == (2, 3)
        assert np.all(actions[:, 0] >= 0) and np.all(actions[:, 0] < 1)
        assert np.all(actions[:, 1] >= -1) and np.all(actions[:, 1] < 1)
        assert np.all(actions[:, 2] >= 0) and np.all(actions[:, 2] < 2)
        
        