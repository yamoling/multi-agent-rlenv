class UnknownEnvironmentException(Exception):
    """Raised when a requested environment name cannot be resolved."""

    def __init__(self, env_name: str):
        super().__init__(
            f"Environment name {env_name} is not known. "
            "Install the matching optional dependency or build the environment with the proper adapter."
        )


class EnvironmentMismatchException(Exception):
    def __init__(self, env, episode):
        message = "Cannot replay the episode on the given environment:\n"
        message += f"\tEnvironment(n_agents={env.n_agents}, n_actions={env.n_actions}, obs_shape={env.observation_shape}, extras_shape={env.extras_shape})\n"
        message += f"\tEpisode(n_agents={episode.n_agents}, n_actions={episode.n_actions}, obs_shape={episode.observation_shape}, extras_shape={episode.extras_shape})"
        super().__init__(message)


class ReplayMismatchException(Exception):
    def __init__(self, what: str, env_value, episode_value, time_step: int):
        message = f"Replaying the actions did not yield the same {what} at step={time_step}:\n"
        message += f"\t(env {what}) {env_value} != {episode_value} (episode {what})"
        super().__init__(message)
