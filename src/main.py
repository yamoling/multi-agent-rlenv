from serde.json import to_json
from dataclasses import asdict
import rlenv
from pprint import pprint


env = rlenv.Builder(rlenv.make("CartPole-v1")).time_limit(20).agent_id().build()
pprint(asdict(env))
