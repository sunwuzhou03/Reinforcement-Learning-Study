import gym
from sacwalker import SACContinuous
import collections
import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils

algorithm_name = 'SAC'
env_name = 'BipedalWalkerHardcore-v3'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

actor_lr = 4e-4
critic_lr = 4e-4
alpha_lr = 4e-4
num_episodes = 10000
hidden_dim = 256
gamma = 0.98
tau = 0.01  # 软更新参数
buffer_size = 100000
minimal_size = 1000
batch_size = 64
max_step = 500
target_entropy = -env.action_space.shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)

agent.load_model('BipedalWalker-v3\SACtrick.pth')
reward_list = []
observation = env.reset()
done = False
step = 0
rs = 0
frame_list = []
while not done:
    action = agent.take_action(observation)
    observation, reward, done, info = env.step(action)
    frame_list.append(env.render('rgb_array'))
    rs += reward
    if done:
        if rs > 300:
            rl_utils.save_video(frame_list, env_name)
        break
    step += 1
print(rs)
reward_list.append(rs)
rl_utils.plot_smooth_reward(reward_list, 5)
env.close()
