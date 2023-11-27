import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import argparse

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, new_state, done):
        self.buffer.append( (state, action, reward, new_state, done) )
    
    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class dqn(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(dqn, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        
    def forward(self, x):
        return self.linear(x)
    
def get_env_shape(env):
    try:    act_dim = env.action_space.n
    except: act_dim = env.action_space.shape[0]

    try:    obs_dim = env.observation_space.n
    except: obs_dim = env.observation_space.shape[0]

    return obs_dim, act_dim

def get_action(dqn, state, T):

    with torch.no_grad():
        q = dqn(torch.tensor(state, dtype = torch.float32).unsqueeze(0))

    T = max(T, 1e-4) # for numerical stability
    softmax = nn.functional.softmax(q / T, dim = 1).numpy()
    
    all_possible_actions = np.arange(0, softmax.shape[-1])
    action = np.random.choice(all_possible_actions, p=np.squeeze(softmax))
    
    return action, q.numpy()

class DeepQN:
    def __init__(self, env_name, hidden_dim = 128, batch_size = 128, gamma = 0.99, lr = 3e-4, capacity = int(1e5)):

        self.gamma = gamma
        self.batch_size = batch_size
        self.env = gym.make(env_name)
        self.obs_dim , self.act_dim = get_env_shape(self.env)

        # initialize buffer
        self.buffer = ReplayBuffer(capacity)

        # initialize dqn network, optimizer and loss function
        self.dqn        = dqn(self.obs_dim, self.act_dim, hidden_dim)
        self.target_dqn = dqn(self.obs_dim, self.act_dim, hidden_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict()) # copy the weights of dqn into target_dqn

        self.optimizer  = optim.Adam(self.dqn.parameters(), lr = lr)
        self.loss_fn    = nn.SmoothL1Loss()
    
    def close(self):
        self.env.close()

    def get_profile(self, episodes, Ti = 5):
        exp_decay = np.exp(-np.log(Ti) / episodes * 6)
        profile   = [Ti * (exp_decay ** i) for i in range(episodes)]
        return profile

    def rollout(self, T):
        '''
        Collect rollout data by running a trajectory until the end
        ''' 
        state = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            action, Qvalues = get_action(self.dqn, state, T)
            new_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            self.buffer.push(state, action, reward, new_state, done)

            if len(self.buffer) > self.batch_size: self.update()
            state = new_state
        return total_reward

    def sync(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
    
    def update(self):#, dqn, target_dqn):
        
        batch = self.buffer.sample(self.batch_size)
        states     = torch.tensor([j[0] for j in batch], dtype = torch.float32)
        actions    = torch.tensor([j[1] for j in batch], dtype = torch.int64)
        rewards    = torch.tensor([j[2] for j in batch], dtype = torch.float32)
        new_states = torch.tensor([j[3] for j in batch], dtype = torch.float32)

        # a mask to keep track of terminal states (1 if not terminal and 0 if terminal)
        masks = torch.tensor([1 - j[4] for j in batch], dtype = torch.int64)

        # set training mode and compute q values
        self.dqn.train()
        qvalues = self.dqn(states)
        qvalues = qvalues.gather(1, actions.unsqueeze(1))

        # compute the value of "new_states"
        with torch.no_grad():
            self.target_dqn.eval()
            target_qvalues = self.target_dqn(new_states)
        target_qvalues = target_qvalues.max(dim = 1)[0]
        target_qvalues = target_qvalues.unsqueeze(1)

        expected_qvalues = rewards.unsqueeze(1) + self.gamma * masks.unsqueeze(1) * target_qvalues
        loss = self.loss_fn(qvalues, expected_qvalues)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), 2)
        self.optimizer.step()

# training loop
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="CartPole-v1", help="set the gym environment")
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes to run")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="set the discount factor")
    parser.add_argument("-v", "--verbose", action="count", help="show log of rewards", default=0)

    args = parser.parse_args()

    env_name = args.env
    print("#################################")
    print("Running:", env_name)
    print("#################################")

    model = DeepQN(env_name=env_name, gamma=args.gamma)
    episodes = args.episodes

    profile = model.get_profile(episodes = episodes)

    rewards_history = np.zeros(episodes)
    for episode, T in enumerate(profile):
        episode_reward = model.rollout(T)
        rewards_history[episode] = episode_reward

        if episode % 10 == 0: model.sync()

        if args.verbose >= 1:
            if episode % 50 == 0:
                print("episode {} --> tot_reward = {}".format(episode, episode_reward))
    
    model.close()

    if args.verbose >= 2:
        fig = plt.figure(figsize = (5,5))
        plt.plot(range(episodes), rewards_history)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.show()

