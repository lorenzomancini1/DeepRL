import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import argparse

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(Actor, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim), # because we want the actor to return a distribution for the actions
            nn.Softmax(dim = 1) 
        )
        
    def forward(self, state):
        policy = self.actor(state)
        return policy
    
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super(Critic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        value = self.critic(state)
        return value

def get_env_shape(env):
    try:    act_dim = env.action_space.n
    except: act_dim = env.action_space.shape[0]

    try:    obs_dim = env.observation_space.n
    except: obs_dim = env.observation_space.shape[0]

    return obs_dim, act_dim

def get_returns(rewards, next_value, mask, γ):
    returns = np.zeros_like(rewards)
    R = next_value
    for t in reversed(range(len(rewards))):
        R = rewards[t] + γ * R * mask[t]
        #returns.insert(0, R)
        returns[t] = R
    return returns

class A2C:
    def __init__(self, env_name, hidden_dim = 128, lr_a = 3e-4, lr_c = 3e-4, gamma = 0.99):

        self.gamma = gamma
        #self.max_steps = max_steps
        self.env  = gym.make(env_name)
        self.obs_dim , self.act_dim = get_env_shape(self.env)

        #initialize networks
        self.actor  = Actor(self.obs_dim, self.act_dim, hidden_dim)
        self.critic = Critic(self.obs_dim, hidden_dim)

        #initialize optimizers
        self.optimizer_a = optim.Adam(self.actor.parameters(),  lr = lr_a)
        self.optimizer_c = optim.Adam(self.critic.parameters(),  lr = lr_c)

    def rollout(self, max_steps = 100):
        '''
        Collect rollout data by performing a max_steps trajectory
        ''' 
        # initialize empty lists for log_probs, values, rewards and mask
        log_probs = []
        values    = []
        rewards   = []
        mask      = []

        state, _ = self.env.reset()
        #print(state)

        for step in range(1, max_steps + 1):
            policy = self.actor(torch.tensor(state).unsqueeze(0))
            value  = self.critic(torch.tensor(state).unsqueeze(0))

            dist   = policy.detach().numpy()
            # select an action according to the policy
            action   = np.random.choice(self.act_dim, p = np.squeeze(dist))
            # compute the log_prob of that action
            log_prob = torch.log(policy.squeeze(0)[action])

            # do a step
            new_state, reward, done, truncated, info = self.env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            mask.append(1 - done)

            state = new_state

            if done: break
        return rewards, values, log_probs, mask, new_state


    def update(self, rewards, values, log_probs, mask, new_state):
        new_value = self.critic(torch.tensor(new_state).unsqueeze(0))
        returns   = get_returns(rewards, new_value, mask, self.gamma)

        returns   = torch.tensor(returns)
        values    = torch.stack(values)
        log_probs = torch.stack(log_probs)

        #compute advantage
        advantage = returns - values

        #compute actor and critic losses
        actor_loss  = - (log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()
        
        self.optimizer_a.step()
        self.optimizer_c.step()

#training loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="CartPole-v1")
    args = parser.parse_args()

    model = A2C(args.env)
    episodes = 1000
    max_steps = 100
    for episode in range(episodes):
        rewards, values, log_probs, mask, new_state = model.rollout(max_steps)

        model.update(rewards, values, log_probs, mask, new_state)

        if episode % 50 == 0:
            print("episode {} --> tot_reward = {}".format(episode, np.sum(rewards)))


    

                           

