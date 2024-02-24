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
import yaml
import os


def plot_smooth_reward(rewards,
                       directory="./",
                       filename="smooth_reward_plot",
                       y_label='Reward',
                       x_label='Episode',
                       window_size=100):

    # 创建目标目录（如果不存在）
    os.makedirs(directory, exist_ok=True)

    img_title = filename
    # 拼接文件名和扩展名
    filename = f"{filename}.png"

    # 构建完整的文件路径
    filepath = os.path.join(directory, filename)

    # 计算滑动窗口平均值
    smoothed_rewards = np.convolve(rewards,
                                   np.ones(window_size) / window_size,
                                   mode='valid')

    # 绘制原始奖励和平滑奖励曲线
    plt.plot(rewards, label='Raw Data Curve')
    plt.plot(smoothed_rewards, label='Smoothed Data Curve')

    # 设置图例、标题和轴标签
    plt.legend()
    plt.title(img_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 保存图像
    plt.savefig(filepath)

    # 关闭图像
    plt.close()


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


class A2C:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device) -> None:
        self.actor = ActorNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = CriticNet(state_dim, hidden_dim, action_dim).to(device)
        self.gamma = gamma
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.device = device

    def save_model(self, save_path='./', filename='model'):
        model = {'actor': self.actor, 'critic': self.critic}
        torch.save(model, f"{save_path}\\{filename}.pth")

    def load_model(self, load_path):
        model = torch.load(load_path)
        self.actor = model['actor']
        self.critic = model['critic']

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def evaluate(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action = torch.argmax(probs, dim=1)
        return action.item()

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
        log_prob = torch.log(self.actor(states).gather(1, actions))

        actor_loss = torch.mean(-log_prob * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(y_now.detach(), v_now))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()


if __name__ == "__main__":

    gamma = 0.99
    algorithm_name = "A2C"
    num_episodes = 1000
    actor_lr = 1e-3
    critic_lr = 1e-2
    env_name = 'CartPole-v0'
    hidden_dim = 128

    #选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 注册环境
    gym.register(id='Snake-v0', entry_point='snake_env:SnakeEnv')

    env = gym.make(env_name)

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.n
    agent = A2C(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,
                device)

    return_list = []
    max_reward = 0
    start_time = time.time()
    flag = 0
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

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    next_action = agent.take_action(next_state)

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['next_actions'].append(next_action)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward
                agent.update(transition_dict)

                return_list.append(episode_return)
                plot_smooth_reward(return_list, env_name, "reward_curve(A2C)")
                if episode_return >= max_reward:
                    max_reward = episode_return
                    agent.save_model(env_name, algorithm_name)

                if episode_return >= 200 and flag == 0:
                    flag = 1
                    end_time = time.time()
                    run_time = end_time - start_time

                    # 打印程序运行时间
                    print(f"A2C到达200需要：{run_time}秒")

                if (i_episodes + 1 % 10 == 0):
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episodes + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
