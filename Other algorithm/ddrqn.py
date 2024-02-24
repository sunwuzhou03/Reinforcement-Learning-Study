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
from rl_utils import plot_smooth_reward


class Qnet(torch.nn.Module):
    def __init__(self,
                 obs_dim,
                 hidden_dim,
                 action_dim,
                 device,
                 num_layers=1,
                 batch_first=True) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.lstm = torch.nn.LSTM(hidden_dim,
                                  hidden_dim,
                                  self.num_layers,
                                  batch_first=True)

    def forward(self, state, max_seq=1, batch_size=1):
        h0 = torch.zeros(self.num_layers, state.size(0),
                         self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, state.size(0),
                         self.hidden_dim).to(device)

        state = F.relu(self.fc1(state)).reshape(-1, max_seq, self.hidden_dim)
        state, _ = self.lstm(state, (h0, c0))
        action_value = self.fc2(state).view(-1, self.action_dim)
        return action_value


class DDRQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, update_target, device) -> None:
        self.Q = Qnet(state_dim, hidden_dim, action_dim, device).to(device)
        self.Q_target = Qnet(state_dim, hidden_dim, action_dim,
                             device).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.Adam(self.Q.parameters(),
                                          lr=learning_rate)
        self.action_dim = action_dim
        print(action_dim)
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
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        Q_value = self.Q(state)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = torch.argmax(Q_value).item()
        return action

    def update(self,
               states,
               actions,
               rewards,
               next_states,
               dones,
               max_seq=1,
               batch_size=1):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions,
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards,
                               dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(next_states,
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(dones,
                             dtype=torch.float).to(self.device).view(-1, 1)
        q_now = self.Q.forward(states, max_seq,
                               batch_size).gather(1,
                                                  actions.long()).view(-1, 1)

        actions_star = self.Q.forward(next_states, max_seq,
                                      batch_size).max(1)[1].view(-1, 1)

        q_next = self.Q_target.forward(next_states,
                                       max_seq, batch_size).gather(
                                           1, actions_star).view(-1, 1)

        y_now = (rewards + self.gamma * q_next * (1 - dones)).view(-1, 1)
        td_error = q_now - y_now

        loss = torch.mean(F.mse_loss(q_now, y_now))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.update_target == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        self.count += 1


class Recurrent_Memory_ReplayBuffer:
    def __init__(self, capacity, max_seq) -> None:
        self.max_seq = max_seq
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    # def sample(self, batch_size):
    #     transitions = random.sample(self.buffer, batch_size)
    #     states, actions, rewards, next_states, dones = zip(*transitions)
    #     return states, actions, rewards, next_states, dones

    def sample(self, batch_size):
        # sample episodic memory
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _ in range(batch_size):
            finish = random.randint(self.max_seq, self.size() - 1)
            begin = finish - self.max_seq
            data = []
            for idx in range(begin, finish):
                data.append(self.buffer[idx])
            state, action, reward, next_state, done = zip(*data)
            states.append(np.vstack(state))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.vstack(next_state))
            dones.append(done)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


if __name__ == "__main__":
    algorithm_name = "DDRQN1"
    gamma = 0.99

    num_episodes = 5000
    buffersize = 10000
    minmal_size = 500
    batch_size = 64
    epsilon = 0.01
    learning_rate = 2e-3
    max_seq = 1
    device = torch.device('cuda')

    env_name = 'Snake-v0'
    # 注册环境
    gym.register(id='Snake-v0', entry_point='snake_env:SnakeEnv')

    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = Recurrent_Memory_ReplayBuffer(buffersize, max_seq)

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    update_target = 50
    agent = DDRQN(state_dim, hidden_dim, action_dim, learning_rate, gamma,
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
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)

                    if i_episodes == int(num_episodes / 10) - 1:
                        env.render()
                        time.sleep(0.1)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minmal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        agent.update(b_s, b_a, b_r, b_ns, b_d, max_seq,
                                     batch_size)
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
