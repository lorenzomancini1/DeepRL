# DeepRL
Implementations of some of the most well known Deep Reinforcement Learning algorithms, namely:
* Advantage Actor Critic (A2C), which make us of the Generalized Advantage Estimation procedure;
* Proximal Policy Optimization (PPO);
* Deep Q-Network;
* Coming soon

To run a specific algorithm with default options, run:
```
python3 src/<algorithm name>.py 
```

Some command-line options supported:
* `--episodes`, sets the number of episodes for the training;
* `--max_steps` or `-s`, sets the maximum length of a trajectory;
* `--gamma` or `-g`, sets the discount factor.

For example, to execute A2C with 500 episodes and 150-steps trajectories:
```
python3 src/a2c.py --episodes 500 --max_steps 150 
```

## Advantage Actor Critic (A2C)
To test it with the OpenAI `gym CartPole-v1` environment run:
```
python3 src/a2c.py -vv
```
## Proximal Policy Optimization (PPO)
Specific options:
* `--epsilon`, to set the clip parameter;
* `--epochs`, number of updates of the Actor and Critic networks per episode;
* `--mini_batch_size`, number of samples to be used for the updates;
