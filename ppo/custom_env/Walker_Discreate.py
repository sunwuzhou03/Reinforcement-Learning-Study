import gym
import numpy as np


class Walker(gym.Env):
    def __init__(self, bins=20):
        self.env = gym.make("BipedalWalker-v3")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.observation_space=self.env.observation_space
        self.action_space = gym.spaces.MultiDiscrete([bins]*self.env.action_space.shape[0])
        self.discrete_action = np.linspace(-1., 1., bins)

    def step(self, action):
        continuous_action = self.discrete_action[action]
        next_state, reward, done, info = self.env.step(continuous_action)
        return next_state, reward, done, info

    def reset(self):
        next_state = self.env.reset()
        return next_state

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def seed(self, seed=None):
        self.env.seed(seed)


if __name__ == "__main__":
    env = Walker()
    state = env.reset()  # 初始化状态
    total_reward = 0
    step_count = 0
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()  # 随机采样一个动作，此处用于演示
        state, reward, done, info = env.step(action)  # 执行动作
        total_reward += reward
        step_count += 1
        print('At step {}, reward = {}, done = {}'.format(step_count, reward, done))
    print('Total reward: {}'.format(total_reward))
