class UnknownEnvironmentException(Exception):
    """The error returns a message telling that the enivornment name is not found in the registry"""

    def __init__(self, env_name: str):
        super().__init__(f"Environment name {env_name} is not known in the registry. Try adding it with rlenv.register(<RLEnvClass>)")


class EnvironmentMismatchException(Exception):
    def __init__(self, env, episode):
        message = "Cannot replay the episode on the given environment:\n"
        message += f"\tEnvironment(n_agents={env.n_agents}, n_actions={env.n_actions}, obs_shape={env.observation_shape}, extra_shape={env.extra_shape})\n"
        message += f"\tEpisode(n_agents={episode.n_agents}, n_actions={episode.n_actions}, obs_shape={episode.observation_shape}, extra_shape={episode.extra_shape})"
        super().__init__(message)


class ReplayMismatchException(Exception):
    def __init__(self, what: str, env_value, episode_value, time_step: int):
        message = f"Replaying the actions did not yield the same {what} at step={time_step}:\n"
        message += f"\t(env {what}) {env_value} != {episode_value} (episode {what})"
        super().__init__(message)
