import gym

env = gym.make('Pendulum-v1')

observation = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # 随机选择一个动作
    print(action)
    observation, reward, done, info = env.step(action)  # 执行动作并获取观察、奖励等信息
    print('Observation:', observation)
    print('Reward:', reward)
