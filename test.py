from ddqn import DQN, plot_smooth_reward, ReplayBuffer
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
import pygame
import datetime

gamma = 0.99
num_episodes = 5000
hidden_dim = 128
buffersize = 10000
minmal_size = 500
batch_size = 32
epsilon = 0.01
target_update = 50
learning_rate = 2e-3
max_step = 500
device = torch.device('cuda')

env_name = 'Snake-v0'
# 注册环境
gym.register(id='Snake-v0', entry_point='snake_env:SnakeEnv')

env = gym.make(env_name, width=10, height=10)
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffersize)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon,
            target_update, device)

load_path = 'Snake-v0/DDQN.pth'  #5best

# 加载模型的状态字典
state_dict = torch.load(load_path)

# 加载状态字典到模型中

agent.Q.load_state_dict(state_dict)
state = env.reset()
done = False
while not done:
    action = agent.take_action(
        torch.tensor(state, dtype=torch.float).to(device))  # 随机选择一个行动
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.1)
