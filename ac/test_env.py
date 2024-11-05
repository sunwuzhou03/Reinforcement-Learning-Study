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
import multiprocessing
import os
import argparse
from a3c import get_args, A3C, ActorNet, CriticNet
from a2c import A2C
from ac import AC
from rl_utils import save_video

opt = get_args()
env = gym.make(opt.env_name)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
env.seed(0)

state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n

AC_agent = AC(state_dim, hidden_dim, action_dim, opt.actor_lr, opt.critic_lr,
              opt.gamma, opt.device)
AC_agent.load_model("CartPole-v0\AC.pth")
AC_agent.actor.cpu()
AC_agent.actor.cpu()

# use gpu to train A2C algorithm so that we need to cpu()
A2C_agent = A2C(state_dim, hidden_dim, action_dim, opt.actor_lr, opt.critic_lr,
                opt.gamma, opt.device)
A2C_agent.load_model("CartPole-v0\A2C.pth")
A2C_agent.actor.cpu()
A2C_agent.actor.cpu()

A3C_agent = A3C(state_dim, hidden_dim, action_dim, opt.actor_lr, opt.critic_lr,
                opt.gamma, opt.device)
A3C_agent.load_model("CartPole-v0\A3C.pth")

print("test AC")
print("-----------------------------------------")
state = env.reset()
done = False
frame_list = []
total_reward = 0
step = 0
max_step = 200
while step < max_step:
    action = AC_agent.evaluate(state)  # 随机选择一个行动
    state, reward, done, info = env.step(action)
    rendered_image = env.render('rgb_array')
    frame_list.append(rendered_image)
    total_reward += reward
    step += 1

save_video(frame_list, "CartPole-V0", "AC", "gif")
print(total_reward)

print("test A2C")
print("-----------------------------------------")
state = env.reset()
done = False
frame_list = []
total_reward = 0
step = 0
max_step = 200
while step < max_step:
    action = A2C_agent.evaluate(state)  # 随机选择一个行动
    state, reward, done, info = env.step(action)
    rendered_image = env.render('rgb_array')
    frame_list.append(rendered_image)
    total_reward += reward
    step += 1

save_video(frame_list, "CartPole-V0", "A2C", "gif")
print(total_reward)

print("test A3C")
print("-----------------------------------------")
state = env.reset()
done = False
frame_list = []
total_reward = 0
step = 0
max_step = 200
while step < max_step:
    action = A2C_agent.evaluate(state)  # 随机选择一个行动
    state, reward, done, info = env.step(action)
    rendered_image = env.render('rgb_array')
    frame_list.append(rendered_image)
    total_reward += reward
    step += 1

save_video(frame_list, "CartPole-v0", "A3C", "gif")
print(total_reward)
