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
import os
import imageio


def save_video(frames, directory="./", filename="video", mode="gif", fps=30):
    height, width, _ = frames[0].shape

    # 创建目标目录（如果不存在）
    os.makedirs(directory, exist_ok=True)

    # 拼接文件名和扩展名
    filename = f"{filename}.{mode}"
    # 或者使用下面的语句
    # filename = "{}.{}".format(filename, mode)

    # 构建完整的文件路径
    filepath = os.path.join(directory, filename)

    # 创建视频写入器
    writer = imageio.get_writer(filepath, fps=fps)

    # 将所有帧写入视频
    for frame in frames:
        writer.append_data(frame)

    # 关闭视频写入器
    writer.close()


def plot_smooth_reward(rewards,
                       window_size=100,
                       directory="./",
                       filename="smooth_reward_plot"):

    # 创建目标目录（如果不存在）
    os.makedirs(directory, exist_ok=True)

    # 拼接文件名和扩展名
    filename = f"{filename}.png"

    # 构建完整的文件路径
    filepath = os.path.join(directory, filename)

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

    # 保存图像
    plt.savefig(filepath)

    # 关闭图像
    plt.close()
