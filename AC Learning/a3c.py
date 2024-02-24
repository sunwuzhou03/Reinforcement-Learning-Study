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


class A3C:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device) -> None:
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
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

    def calculate_grad(self, transition_dict):

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

        return actor_loss, critic_loss


def worker(conn, world_index, opt):

    env = gym.make(opt.env_name)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env.seed(world_index)

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n

    local_agent = A3C(state_dim, hidden_dim, action_dim, opt.actor_lr,
                      opt.critic_lr, opt.gamma, opt.device)
    max_step = opt.max_step
    while True:
        global_agent = conn.recv()
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
        episode_reward = 0

        local_agent.actor.load_state_dict(global_agent.actor.state_dict())
        local_agent.critic.load_state_dict(global_agent.critic.state_dict())

        for _ in range(max_step):
            action = local_agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            next_action = local_agent.take_action(next_state)

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['next_actions'].append(next_action)
            transition_dict['rewards'].append((reward + 8.0) / 8.0)
            transition_dict['dones'].append(done)
            episode_reward += reward
            state = next_state

            if done:
                break

        actor_loss, critic_loss = local_agent.calculate_grad(transition_dict)

        local_agent.actor_optimizer.zero_grad()
        local_agent.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        for local_actor_param, global_actor_param in zip(
                local_agent.actor.parameters(),
                global_agent.actor.parameters()):
            if global_actor_param.grad is not None:
                break
            global_actor_param._grad = local_actor_param.grad
            if opt.device == 'cpu':
                global_actor_param._grad = local_actor_param.grad
            else:
                global_actor_param._grad = local_actor_param.grad.cuda()

        for local_critic_param, global_critic_param in zip(
                local_agent.critic.parameters(),
                global_agent.critic.parameters()):
            if global_critic_param.grad is not None:
                break
            if opt.device == 'cpu':
                global_critic_param._grad = local_critic_param.grad
            else:
                global_critic_param._grad = local_critic_param.grad.cuda()

        global_agent.actor_optimizer.step()
        global_agent.critic_optimizer.step()

        # 管道传送
        conn.send((episode_reward, actor_loss.detach(), critic_loss.detach()))
        # conn.close()


def on_policy_train(opt):

    env = gym.make(opt.env_name)

    process_num = opt.process_num
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num)
                     for pipe1, pipe2 in (multiprocessing.Pipe(), ))
    child_process_list = []
    for i in range(process_num):
        pro = multiprocessing.Process(target=worker,
                                      args=(pipe_dict[i][1], i, opt))
        child_process_list.append(pro)
    [p.start() for p in child_process_list]

    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    num_episodes = opt.num_episodes
    global_agent = A3C(state_dim, hidden_dim, action_dim, opt.actor_lr,
                       opt.critic_lr, opt.gamma, opt.device)

    return_list = []
    critic_list = []
    actor_list = []
    inf = float('inf')
    max_reward = -inf
    re_reward = 0
    start_time = time.time()
    flag = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episodes in range(int(num_episodes / 10)):
                [
                    pipe_dict[i][0].send(global_agent)
                    for i in range(process_num)
                ]
                total_reward = 0
                actor_total = 0
                critic_total = 0
                for i in range(process_num):
                    # 这句带同步子进程的功能，收不到子进程的数据就都不会走到for之后的语句
                    reward, actor_loss, critic_loss = pipe_dict[i][0].recv()
                    re_reward = max(re_reward, reward)
                    # print(reward)
                    total_reward += reward
                    actor_total += actor_loss.cpu().item()
                    critic_total += critic_loss.cpu().item()

                # print(f"total_reward={total_reward}")
                # print(f"max_reward={total_reward/process_num}")

                return_list.append(total_reward / process_num)

                actor_list.append(actor_total / process_num)

                critic_list.append(critic_total / process_num)

                plot_smooth_reward(return_list, "./", "reward_curve", "Reward")
                plot_smooth_reward(actor_list, "./", "actor_loss_curve",
                                   "Actor_Loss")
                plot_smooth_reward(critic_list, "./", "critic_loss_curve",
                                   "Critic_Loss")

                pbar.update(1)

                if total_reward / process_num >= max_reward:
                    max_reward = total_reward / process_num
                    global_agent.save_model(opt.env_name, "A3C")

                if re_reward >= 200 and flag == 0:
                    flag = 1
                    end_time = time.time()
                    run_time = end_time - start_time

                    # 打印程序运行时间
                    print(f"A3C到达200需要：{run_time}秒")

    [p.terminate() for p in child_process_list]


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model: Asynchronous Methods for Deep Reinforcement Learning for CartPole-v0"""
    )

    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-2)
    parser.add_argument('--gamma',
                        type=float,
                        default=0.9,
                        help='discount factor for rewards')

    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--env_name", type=str, default="CartPole-v0")

    parser.add_argument("--process_num", type=int, default=4)

    parser.add_argument("--max_step", type=int, default=1000)

    parser.add_argument("--num_episodes", type=int, default=1000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = get_args()
    on_policy_train(opt)
