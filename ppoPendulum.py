import numpy as np
import torch
import gym
import pygame
import collections
import random
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from rl_utils import plot_smooth_reward


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state):
        hidden_state = F.relu(self.fc1(state))
        acion_value = self.fc2(hidden_state)
        return acion_value


class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):

        hidden_state = F.relu(self.fc1(state))
        mu = 2.0 * F.tanh(self.fc_mu(hidden_state))
        sigma = F.softplus(self.fc_std(hidden_state))
        return mu, sigma


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, lmbda, eps, epochs, device) -> None:
        self.actor = ActorNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = CriticNet(state_dim, hidden_dim, action_dim).to(device)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.device = device

    def save_model(self, save_path='./', filename='model'):
        torch.save(self.actor.state_dict(), f"{save_path}\\{filename}.pth")

    def load_model(self, load_path):
        self.actor.load_state_dict(torch.load(load_path))

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        # print(action)
        return action.cpu().numpy().flatten()

    def update(self, transition_dict):
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        next_actions = torch.tensor(transition_dict['next_actions']).view(
            -1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).to(self.device).view(-1, 1)

        v_now = self.critic(states).view(-1, 1)
        v_next = self.critic(next_states).view(-1, 1)
        y_now = (self.gamma * v_next * (1 - dones) + rewards).view(-1, 1)
        td_delta = y_now - v_now
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        mu, sigma = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), sigma.detach())
        old_log_probs = action_dists.log_prob(actions)
        for _ in range(self.epochs):

            mu, sigma = self.actor(states)
            action_dists = torch.distributions.Normal(mu, sigma)
            log_probs = action_dists.log_prob(actions)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), y_now.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == "__main__":

    gamma = 0.9
    algorithm_name = "PPOdemo"
    num_episodes = 2000
    actor_lr = 1e-4
    critic_lr = 5e-3
    lmbda = 0.9
    eps = 0.2
    epochs = 10
    device = torch.device('cuda')
    env_name = 'Pendulum-v1'  #'CartPole-v0'
    max_step = 500
    # 注册环境
    gym.register(id='Snake-v0', entry_point='snake_env:SnakeEnv')

    env = gym.make(env_name)

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,
                lmbda, eps, epochs, device)

    return_list = []
    inf = float('inf')
    max_reward = -inf
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episodes in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False

                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'next_actions': [],
                    'rewards': [],
                    'dones': []
                }

                for _ in range(max_step):
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)

                    next_action = agent.take_action(next_state)
                    # print(next_action, next_state, action, state, reward)

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['next_actions'].append(next_action)
                    transition_dict['rewards'].append((reward + 8.0) / 8.0)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward
                    if i_episodes == int(num_episodes / 10) - 1:
                        env.render()
                        time.sleep(0.1)
                    if done:
                        break
                agent.update(transition_dict)

                return_list.append(episode_return)
                plot_smooth_reward(return_list, 100, env_name, algorithm_name)
                if episode_return > max_reward:
                    max_reward = episode_return
                    agent.save_model(env_name, algorithm_name)
                if (i_episodes + 1 % 10 == 0):
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episodes + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
