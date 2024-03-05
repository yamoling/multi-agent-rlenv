from serde.json import to_json
from dataclasses import asdict
import rlenv
from pprint import pprint
from pettingzoo.sisl import pursuit_v4


env = rlenv.Builder(rlenv.make(pursuit_v4.parallel_env())).time_limit(20).agent_id().build()
#env = rlenv.Builder(rlenv.make("CartPole-v1")).time_limit(20).agent_id().build()

env.reset()
pprint(asdict(env))