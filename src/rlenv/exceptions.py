

class UnknownEnvironmentException(Exception):
    """The error returns a message telling that the enivornment name is not found in the registry"""
    def __init__(self, env_name: str):
        super().__init__(f"Environment name {env_name} is not known in the registry. Try adding it with rlenv.register(<RLEnvClass>)")