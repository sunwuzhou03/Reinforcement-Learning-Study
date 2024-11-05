from ddqn import DDQN, plot_smooth_reward, ReplayBuffer
from a2c import A2C
from ppo import PPO
from reinforce_baseline import REINFORCE_BASELINE
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
from rl_utils import save_video

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

load_path = 'Snake-v0/DDQN.pth'
agent = DDQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon,
             target_update, device)

load_path = 'Snake-v0/REINFORCE_baseline.pth'
agent = REINFORCE_BASELINE(state_dim, hidden_dim, action_dim, learning_rate,
                           gamma, device)

load_path = 'Snake-v0/PPO.pth'  #5best
agent = PPO(state_dim, hidden_dim, action_dim, 1e-3, 1e-3, gamma, 0.95, 0.2, 5,
            device)

load_path = 'Snake-v0/A2C.pth'  #5best
agent = A2C(state_dim, hidden_dim, action_dim, 1e-3, 1e-3, gamma, device)

# 加载模型的状态字典
state_dict = torch.load(load_path)

# 加载状态字典到模型中

agent.load_model(load_path)

state = env.reset()
done = False
frame_list = []
while not done:
    action = agent.take_action(state)  # 随机选择一个行动
    state, reward, done, info = env.step(action)
    rendered_image = env.render('rgb_array')
    frame_list.append(rendered_image)
save_video(frame_list)
