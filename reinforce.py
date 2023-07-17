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


class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):

        hidden_state = F.relu(self.fc1(state))
        # print(self.fc2(hidden_state))
        probs = F.softmax(self.fc2(hidden_state), dim=1)
        return probs


class Reinforce:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device) -> None:
        self.actor = ActorNet(state_dim, hidden_dim, action_dim).to(device)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.actor.parameters(),
                                          lr=learning_rate)
        self.device = device

    def save_model(self, save_path='./', filename='model'):
        torch.save(self.actor.state_dict(), f"{save_path}\\{filename}.pth")

    def load_model(self, load_path):
        self.actor.load_state_dict(torch.load(load_path))

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.actor(state).gather(1, action))
            G = self.gamma * G + reward
            loss = torch.mean(-log_prob * G)
            loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    algorithm_name = "REINFORCE"
    gamma = 0.98

    num_episodes = 5000
    learning_rate = 2e-3
    device = torch.device('cuda')

    env_name = 'Snake-v0'  #'CartPole-v0'
    # 注册环境
    gym.register(id='Snake-v0', entry_point='snake_env:SnakeEnv')

    env = gym.make(env_name)

    env.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    update_target = 100
    agent = Reinforce(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                      device)

    return_list = []
    max_reward = 0
    for i in range(20):
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
                    'rewards': [],
                    'dones': []
                }
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    env.render()
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    if i_episodes == int(num_episodes / 10) - 1:
                        time.sleep(0.1)
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
