from marlenv import Builder
from marlenv.adapters import SMAC

env = Builder(SMAC("3m")).agent_id().time_limit(20).build()
print(env.extra_feature_shape)
