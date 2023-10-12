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

opt = get_args()
env = gym.make(opt.env_name)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
env.seed(0)

state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n

A3C_agent = A3C(state_dim, hidden_dim, action_dim, opt.actor_lr, opt.critic_lr,
                opt.gamma, opt.device)
A3C_agent.load_model("CartPole-v0\A3C.pth")

# use gpu to train A2C algorithm so that we need to cpu()
A2C_agent = A2C(state_dim, hidden_dim, action_dim, opt.actor_lr, opt.critic_lr,
                opt.gamma, opt.device)
A2C_agent.load_model("CartPole-v0\A2C.pth")
A2C_agent.actor.cpu()
A2C_agent.actor.cpu()

print("test A3C")
print("-----------------------------------------")
state = env.reset()
done = False
while not done:
    action = A3C_agent.evaluate(state)
    next_state, reward, done, _ = env.step(action)
    env.render()
    state = next_state

print("test A2C")
print("-----------------------------------------")
state = env.reset()
done = False
while not done:
    action = A2C_agent.evaluate(state)
    next_state, reward, done, _ = env.step(action)
    env.render()
    state = next_state