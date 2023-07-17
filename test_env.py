import gym

env = gym.make('BipedalWalker-v3')
observation = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # 替换为你的策略
    observation, reward, done, info = env.step(action)
    env.render()
env.close()