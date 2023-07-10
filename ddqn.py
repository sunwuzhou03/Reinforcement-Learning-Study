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


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        hidden_state = F.relu(self.fc1(state))
        acion_value = self.fc2(hidden_state)
        return acion_value


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, update_target, device) -> None:
        self.Q = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.Q_target = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.Adam(self.Q.parameters(),
                                          lr=learning_rate)
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.count = 1
        self.update_target = update_target

    def save_model(self, save_path='./', filename='model'):
        torch.save(self.Q.state_dict(), f"{save_path}\\{filename}.pth")

    def load_model(self, load_path):
        self.Q.load_state_dict(torch.load(load_path))
        self.Q_target.load_state_dict(torch.load(load_path))

    def take_action(self, state):

        Q_value = self.Q(state)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = torch.argmax(Q_value).item()
        return action

    def update(self, states, actions, rewards, next_states, dones):
        states_cuda = torch.tensor(states, dtype=torch.float).to(self.device)
        actions_cuda = torch.tensor(actions,
                                    dtype=torch.float).view(-1,
                                                            1).to(self.device)
        rewards_cuda = torch.tensor(rewards,
                                    dtype=torch.float).to(self.device).view(
                                        -1, 1)
        next_states_cuda = torch.tensor(next_states,
                                        dtype=torch.float).to(self.device)
        dones_cuda = torch.tensor(dones,
                                  dtype=torch.float).to(self.device).view(
                                      -1, 1)
        q_now = self.Q(states_cuda).gather(1, actions_cuda.long()).view(-1, 1)

        actions_star = self.Q(next_states_cuda).max(1)[1].view(-1, 1)

        q_next = self.Q_target(next_states_cuda).gather(1, actions_star).view(
            -1, 1)

        y_now = (rewards_cuda + self.gamma * q_next * (1 - dones_cuda)).view(
            -1, 1)
        td_error = q_now - y_now

        loss = torch.mean(F.mse_loss(q_now, y_now))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.update_target == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        self.count += 1


class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


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
    algorithm_name = "DDQN"
    gamma = 0.99

    num_episodes = 5000
    buffersize = 10000
    minmal_size = 500
    batch_size = 64
    epsilon = 0.01
    learning_rate = 2e-3
    device = torch.device('cuda')

    env_name = 'Snake-v0'
    # 注册环境
    gym.register(id='Snake-v0', entry_point='snake_env:SnakeEnv')

    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffersize)

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    update_target = 100
    agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                epsilon, update_target, device)

    return_list = []
    max_reward = 0
    for i in range(20):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episodes in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(
                        torch.tensor(state, dtype=torch.float).to(device))
                    next_state, reward, done, _ = env.step(action)
                    env.render()
                    if i_episodes == int(num_episodes / 10) - 1:
                        time.sleep(0.1)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minmal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)

                        agent.update(b_s, b_a, b_r, b_ns, b_d)
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
