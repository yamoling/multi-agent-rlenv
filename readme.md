# RLEnv: yet another RL framework
This framework aims at high level abstractions of RL concepts, allowing building blocks to fit into place.
Every building block is defined by an interface (python abstract class).

## Designing an environment
To create an environment that is compatible with RLEnv, you should inherit from the `rl.models.RLEnv` class.

## Start a training
### From code
```python
env = rl.make_env("CartPole-v1")
model = rl.nn.model_bank.MLP(env.observation_shape, env.extra_feature_shape, (env.n_actions, ))
optimizer = torch.optim.Adam(model.parameters(), 5e-4)
qlearning = rl.qlearning.DQNBuilder(model, optimizer, 0.99, 50000, 32, 200, False, loss_function=rl.nn.loss_functions.mse).build()
policy = rl.policies.DecreasingEpsilonGreedy(env.n_actions, env.n_agents, qlearning, decrease_amount=0.0002)
agent = rl.Agent(policy)
runner = rl.Runner(env, agent)
runner.train(n_steps=10_000)
```

### Through the command line
```bash
python3 src/main.py train --algo=dqn --env=CartPole-v0 --model=MLP
```
All the arguments can be specified on the command lines and default values reside in `config/default_arguments.cfg`.
