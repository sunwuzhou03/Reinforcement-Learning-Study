import gym
import numpy as np


if __name__ == "__main__":
    env = gym.make("BipedalWalkerHardcore-v3")
    state = env.reset()  # 初始化状态
    total_reward = 0
    step_count = 0
    done = False
    from ppo_multienv import PPOAgent
    from ppo_multienv import parse_args
    import torch
    import time

    args = parse_args()
    agent = PPOAgent(env.observation_space, env.action_space, args).to(args.device)
    agent = torch.load("BipedalWalker-v3/2024_02_01_21_54_BipedalWalkerHardcore-v3.pth")
    next_obs = env.reset()  # 初始化状态
    while not done:
        next_obs = torch.tensor(next_obs).unsqueeze(0).to(args.device)
        with torch.no_grad():
            action, logprob, entropy, value = agent.get_action_and_value(next_obs)
        next_obs, reward, done, info = env.step(action.flatten().cpu().numpy())
        if done:
            next_obs = env.reset()
        env.render()
        # action = 1#env.action_space.sample()  # 随机采样一个动作，此处用于演示
        # state, reward, done, info = env.step(action)  # 执行动作
        total_reward += reward
        step_count += 1
        print('At step {}, reward = {}, done = {}'.format(step_count, reward, done))
    print('Total reward: {}'.format(total_reward))