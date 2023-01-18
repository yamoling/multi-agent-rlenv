from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment configuration"""
    env: str
    """The name/id of the environment"""
    horizon: int
    """The time limit"""
    with_last_action: bool
    """Whether to include the last action in the observations"""
    with_agent_id: bool
    """Whether to include the agent id in the observations"""
