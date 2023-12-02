
import numpy as np
import matplotlib.pyplot as plt
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

def get_returns(rewards, values, mask, gamma, lambda_):
    returns = np.zeros_like(rewards)
    GAE = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * mask[t] - values[t]
        GAE = delta + gamma * lambda_ * GAE * mask[t]
        returns[t] = GAE + values[t]
    return returns

class PPO:
    def __init__(self, env_name, hidden_dim = 256, lr_a = 3e-3, lr_c = 3e-3, gamma = 0.99, lambda_ = 0.95, mini_batch_size = 7, epsilon = 0.2, device = "cpu"):

        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.mini_batch_size = mini_batch_size
        self.device = device
        #self.max_steps = max_steps
        self.env  = gym.make(env_name)
        self.obs_dim , self.act_dim = get_env_shape(self.env)

        #initialize networks
        self.actor  = Actor(self.obs_dim, self.act_dim, hidden_dim)
        self.critic = Critic(self.obs_dim, hidden_dim)

        #initialize optimizers
        self.optimizer_a = optim.Adam(self.actor.parameters(),  lr = lr_a)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr = lr_c)

    def get_actor(self):
        return self.actor
    
    def close(self):
        self.env.close()

    def rollout(self, max_steps = 100):

        env = self.env

        states    = np.empty(max_steps, dtype = object)
        actions   = np.zeros(max_steps, dtype = int)
        rewards   = np.zeros(max_steps, dtype = float)
        values    = np.zeros(max_steps + 1, dtype = float)
        log_probs = np.zeros(max_steps, dtype = float)
        mask      = np.zeros(max_steps, dtype = int)

        state = env.reset()

        for step in range(max_steps):
            with torch.no_grad():
                policy = self.actor(torch.tensor(state).unsqueeze(0))
                value  = self.critic(torch.tensor(state).unsqueeze(0))

            dist = policy.numpy()
            # select an action according to the policy
            action   = np.random.choice(self.act_dim, p = np.squeeze(dist))
            # compute the log_prob of that action
            log_prob = torch.log(policy.squeeze(0)[action])

            # do a step
            new_state, reward, done, info = self.env.step(action)

            states[step]    = state
            actions[step]   = action
            rewards[step]   = reward
            values[step]    = value#.numpy()#[0, 0]
            log_probs[step] = log_prob
            mask[step]      = 1 - done
        
            state = new_state
            if done: break
        
        stop = step + 1
        
        with torch.no_grad():
            values[stop] = self.critic(torch.tensor(new_state).unsqueeze(0))
            

        states    = states[:stop]
        actions   = actions[:stop]
        rewards   = rewards[:stop]
        values    = values[:stop + 1]
        #print(values[-1] != 0)
        log_probs = log_probs[:stop]
        #print(log_probs)
        mask      = mask[:stop]

        returns = get_returns(rewards, values, mask, self.gamma, self.lambda_)

        advantages = returns - values[:-1]

        return states, actions, rewards, returns, advantages, log_probs
    
    def update(self, states, actions, log_probs, returns, advantages):

        mini_batch_size = self.mini_batch_size
    
        actor_loss  = torch.empty(mini_batch_size)#, dtype = object)
        critic_loss = torch.empty(mini_batch_size)

        for i in range(mini_batch_size):
            state  = states[i]
            action = actions[i]

            policy = self.actor(torch.tensor(state).unsqueeze(0).to(self.device))
            value  = self.critic(torch.tensor(state).unsqueeze(0).to(self.device))

            adv     = advantages[i]
            return_ = returns[i]

            curr_log_prob = torch.log(policy.squeeze(0)[action])
            old_log_prob  = log_probs[i]

            ratio = (curr_log_prob - old_log_prob).exp()
            s1    = ratio * adv
            s2    = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv

            actor_loss[i]  = torch.min(s1, s2) 
            critic_loss[i] = return_ - value

        epoch_actor_loss  = - actor_loss.mean()
        epoch_critic_loss = critic_loss.pow(2).mean()

        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()

        epoch_actor_loss.backward()
        epoch_critic_loss.backward()

        self.optimizer_a.step()
        self.optimizer_c.step()
    
    def sample_batch(self, states, actions, log_probs, returns, advantages):
        '''
        Sample a minibatch from the trajectory
        '''

        assert len(states) == len(log_probs) == len(returns) == len(advantages)

        high = len(states)
        n = self.mini_batch_size

        random_idxs = np.random.randint(0, high, n)

        mb_states     = states[random_idxs]
        mb_actions    = actions[random_idxs]
        #mb_values     = values[random_idxs]
        mb_log_probs  = log_probs[random_idxs]
        mb_returns    = returns[random_idxs]
        mb_advantages = advantages[random_idxs]

        return mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="CartPole-v1", help="set the gym environment")
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes to run")
    parser.add_argument("-s", "--steps", type=int, default=100, help="number of steps per episode")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="set the discount factor")
    parser.add_argument("-l", "--lambda_", type=float, default=1, help="set the lambda value for the GAE")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs per episode")
    parser.add_argument("-v", "--verbose", action="count", help="show log of rewards", default=0)

    args = parser.parse_args()

    env_name = args.env
    print("#################################")
    print("Running:", env_name)
    print("#################################")

    model = PPO(env_name=env_name, gamma=args.gamma, lambda_ = args.lambda_)
    episodes = args.episodes
    max_steps = args.steps

    ###########################
    ###### TRAINING LOOP ######
    ###########################
    rewards_history = np.zeros(episodes)
    for episode in range(episodes):

        states, actions, rewards, returns, advantages, log_probs = model.rollout(max_steps = 150)

        rewards_history[episode] = np.sum(rewards)
        if args.verbose >= 1:
            if episode % 50 == 0:
                print("episode {} --> tot_reward = {}".format(episode, np.sum(rewards)))

        for epoch in range(args.epochs):
             mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages = model.sample_batch(states, actions, log_probs, returns, advantages)

             model.update(mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages)

    model.close()

    if args.verbose >= 2:
        fig = plt.figure(figsize = (5,5))
        plt.plot(range(episodes), rewards_history)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.show()

