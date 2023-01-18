#! /usr/bin/env python3
import argparse
import rlenv


def default_subparser(parser: argparse.ArgumentParser):
    parser._subparsers

def parse_arguments():
    """
    Parse command line arguments.
    These arguments match the Config attributes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, help="The command to execute", default="train")
    parser.add_argument("--env", required=True, type=str, help="The environment. Can be a gym environment or SMAC.")
    parser.add_argument("--algo", type=str.lower,
                             choices=["qlearning", "dqn", "vdn", "random", "qmix"], required=True, help="The algorithm to use")
    parser.add_argument("--config", default="config/default_arguments.cfg",
                             type=str, help="The config file to override the default one")
    parser.add_argument("--logdir", type=str, help="The log directory. Is created if it does not exist.")
    parser.add_argument("--record", action="store_true", default=False, help="Record videos of the tests")

    parser.add_argument("--double_qlearning", default=None, action="store_true", help="Whether to use double Q-learning or not")
    parser.add_argument("--curiosity", default=None, help="Whether to add intrinsic curiosity rewards or not")
    parser.add_argument("--lr", type=float, help="The learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adam", "rmsprop"], help="The optimiser to use")
    parser.add_argument("--gamma", type=float, help="The discount factor gamma")
    parser.add_argument("--policy", type=str.lower, help="The name of the policy to use")
    parser.add_argument("--eps", type=float, help="Value of epsilon. Use only for epsilon greedy policies.")
    parser.add_argument("--eps_min", type=float, help="Minimal value for epsilon. Use only with --annealed")
    parser.add_argument("--test_epsilon", type=float, help="Epsilon value when testing the model")
    parser.add_argument("--anneal", type=int, help="How much steps to anneal epsilon on. 0 by default = no annealing")
    parser.add_argument("--blind_probability", type=float,
                              help="If > 0, probabilistically blinds the observations by replacing it by zeros")
    parser.add_argument("--load", type=str, help="Model to load")
    parser.add_argument("--with_agent_id", default=None, type=bool,
                              help="Whether one-hot vectors must be added to the agent observation")
    parser.add_argument("--with_last_action", action="store_true", default=None,
                              help="Whether to include the last action to the agent observation")
    parser.add_argument("--render", action="store_true", default=None, help="Whether to render the tests or not")
    parser.add_argument("--batch_size", type=int, help="The batch size to use for training")
    parser.add_argument("--memory_size", type=int, help="The size of the replay memory")
    parser.add_argument("--per", action="store_true", default=None, help="Whether to use Prioritized Experience Replay or not")
    parser.add_argument("--seed", type=int, help="Seed to use in torch, numpy, random and env for reproducibility")
    parser.add_argument("--device", type=str)
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument("--horizon", type=int, help="The time limit for the environment")

    parser.add_argument("--training_episodes", type=int, default=None, help="Amount of training episodes to train on")
    parser.add_argument("--training_steps", type=int, default=None, help="Amount of training steps to train on")
    parser.add_argument("--test_interval", type=int, default=None, help="Episode interval at which tests will be performed")
    parser.add_argument("--n_tests", type=int, default=None, help="How many tests are performed")
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args()
    return args


def train(args):
    """Train an agent with the given configuration"""
    env = args.env
    algo = args.algo
    config_path = args.config

    config = rlenv.Config(config_path, env, algo)
    config.override_with_args(args)
    runner = rlenv.build_runner(config)
    runner.train(test_interval=config.test_interval, n_tests=config.n_tests, n_episodes=config.training_episodes, n_steps=config.training_steps)
    return runner.logger.logdir


def compare_per(arguments):
    """Compare performance with and without PER"""
    from time import time
    from analysis.aggretage import aggregate
    from multiprocessing import Pool
    from copy import deepcopy
    N_REPEATS = 5
    per_jobs, uniform_jobs = [], []
    with Pool(5) as pool:
        for seed in range(N_REPEATS):
            arguments.seed = seed
            arguments.per = True
            args = deepcopy(arguments)
            args.logdir = f"per-{seed}-{args.env}-{args.algo}"
            per_jobs.append(pool.apply_async(train, (args, )))
            arguments.per = False
            args = deepcopy(arguments)
            args.logdir = f"no_per-{seed}-{args.env}-{args.algo}"
            uniform_jobs.append(pool.apply_async(train, (args, )))
        per_logs = [result.get() for result in per_jobs]
        uniform_logs = [result.get() for result in uniform_jobs]
    result_folder = f"results/{time()}-per-comparison-{arguments.env}"
    aggregate(per_logs, uniform_logs, ["Test/avg_score", "Test/max_score"], output_dir=result_folder)

    

if __name__ == "__main__":
    from analysis.aggretage import aggregate
    uniform = "logs/no_per-0 logs/no_per-1 logs/no_per-2 logs/no_per-3 logs/no_per-4".split()
    per = "logs/per-0 logs/per-1 logs/per-2 logs/per-3 logs/per-4".split()
    aggregate(per, uniform, ["Test/avg_score", "Test/max_score"], "aggregated")
    arguments = parse_arguments()
    exit()
    command = arguments.command
    delattr(arguments, "command")
    if command == "train":
        train(arguments)
    elif command == "compare":
        compare_per(arguments)
    else:
        raise ValueError(f"Unknown command {command}")
