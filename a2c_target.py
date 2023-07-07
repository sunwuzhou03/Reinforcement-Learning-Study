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
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):

        hidden_state = F.relu(self.fc1(state))
        # print(self.fc2(hidden_state))
        probs = F.softmax(self.fc2(hidden_state), dim=1)
        return probs


class AC:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, tau, device) -> None:
        self.actor = ActorNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = CriticNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target = CriticNet(state_dim, hidden_dim,
                                       action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.gamma = gamma
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.device = device
        self.tau = tau

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

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        reward = transition_dict['rewards']
        state = transition_dict['states']
        next_state = transition_dict['next_states']
        action = transition_dict['actions']
        next_action = transition_dict['next_actions']
        done = transition_dict['dones']
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        reward = torch.tensor([reward],
                              dtype=torch.float).view(-1, 1).to(self.device)
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = torch.tensor([action]).view(-1, 1).to(self.device)
        next_state = torch.tensor([next_state],
                                  dtype=torch.float).to(self.device)
        next_action = torch.tensor([next_action]).view(-1, 1).to(self.device)
        done = torch.tensor(done,
                            dtype=torch.float).to(self.device).view(-1, 1)
        v_now = self.critic(state).view(-1, 1)
        v_next = self.critic_target(next_state).view(-1, 1)
        # y_now = self.gamma * v_next + reward
        y_now = (self.gamma * v_next + reward).view(-1, 1)
        # print(y_now)
        # print(q_next)
        td_delta = y_now - v_now
        log_prob = torch.log(self.actor(state).gather(1, action))
        # print(log_prob)
        # print(action)
        actor_loss = torch.mean(-log_prob * td_delta.detach())
        actor_loss.backward()

        critic_loss = torch.mean(F.mse_loss(y_now, v_now))
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.soft_update(self.critic, self.critic_target)


def plot_smooth_reward(rewards, window_size=100):
    # 计算滑动窗口平均值
    smoothed_rewards = np.convolve(rewards,
                                   np.ones(window_size) / window_size,
                                   mode='valid')

    # 绘制原始奖励和平滑奖励曲线
    plt.plot(rewards, label='Raw Reward')
    plt.plot(smoothed_rewards, label='Smoothed Reward')

    # 设置图例、标题和轴标签
    plt.legend()
    plt.title('Smoothed Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # 显示图像
    plt.show()


if __name__ == "__main__":

    gamma = 0.99
    algorithm_name = "demo"
    num_episodes = 1000
    actor_lr = 1e-3
    critic_lr = 2e-3
    device = torch.device('cuda')
    env_name = 'Snake-v0'  #'CartPole-v0'

    # 注册环境
    gym.register(id='Snake-v0', entry_point='snake_env:SnakeEnv')

    env = gym.make(env_name)

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    tau = 5e-4
    agent = AC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,
               tau, device)

    return_list = []
    max_reward = 0
    for i in range(30):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episodes in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    next_action = agent.take_action(next_state)
                    env.render()

                    transition_dict = {
                        'states': state,
                        'actions': action,
                        'next_states': next_state,
                        'next_actions': next_action,
                        'rewards': reward,
                        'dones': done
                    }

                    agent.update(transition_dict)

                    state = next_state
                    episode_return += reward
                    if i_episodes == int(num_episodes / 10) - 1:
                        time.sleep(0.1)

                return_list.append(episode_return)
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

    plot_smooth_reward(return_list)