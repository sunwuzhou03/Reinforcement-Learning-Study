import gym
import numpy as np
import random
import pygame
import sys
import time


class SnakeEnv(gym.Env):
    def __init__(self, width=20, height=10):
        super(SnakeEnv, self).__init__()

        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.snake = None
        self.food = None
        self.action_mask = [True] * 4
        self.episode_length = 0
        self.episode_reward = 0
        self.episode = 0
        self.max_episode_length = 200
        self.cell_size = 20
        self.window_size = (self.width * self.cell_size,
                            self.height * self.cell_size)
        # 定义颜色
        self.color_bg = (255, 255, 255)
        self.color_head = (0, 120, 120)
        self.color_body = (0, 255, 0)
        self.color_food = (255, 0, 0)

        # 定义贪吃蛇环境的观测空间和行动空间
        inf = self.width + self.height
        low = np.array([-10, -1.0, 0, 0, 0, 0])  # 连续状态空间的最小值
        high = np.array([10, 1.0, 1.0, 1.0, 1.0, 1.0])  # 连续状态空间的最大值
        self.observation_space = gym.spaces.Box(low, high, shape=(6,), dtype=float)

        # 0 左 1上 2右 3下
        self.action_space = gym.spaces.Discrete(4)

    def generate_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                return (x, y)

    def reset(self):
        self.grid = np.zeros((self.height, self.width))
        self.snake = [(0, 0)]
        self.food = self.generate_food()
        self.time = 1
        self.action_mask = self.get_mask()
        self.episode_length = 0
        self.episode_reward = 0
        self.update_grid()
        return self.get_state()

    # 状态获取函数
    def get_state(self):
        state = []
        x, y = self.snake[0]
        fx, fy = self.food
        state.append(fx - x)
        state.append(fy - y)
        for gx, gy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dx, dy = x + gx, y + gy
            if dx < 0 or dy < 0 or dx >= self.width or dy >= self.height or (
                    dx, dy) in self.snake:
                state.append(0)
                continue
            else:
                state.append(1)  # 四个方向可以走
        return np.array(state, dtype=np.float32)

    # 更新当前动画状态
    def update_grid(self):
        self.grid = np.zeros((self.height, self.width))
        x, y = self.snake[0]
        self.grid[y, x] = 1
        for x, y in self.snake[1:]:
            self.grid[y, x] = 2
        fx, fy = self.food
        self.grid[fy, fx] = 3

    # 获取动作mask
    def get_mask(self):
        action_mask = [True] * 4
        x, y = self.snake[0]
        for i, (gx, gy) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
            dx, dy = x + gx, y + gy
            if dx < 0 or dy < 0 or dx >= self.width or dy >= self.height or (
                    dx, dy) in self.snake:
                action_mask[i] = False
            else:
                action_mask[i] = True  # True则表示动作可以执行
        return action_mask

    def step(self, action):
        x, y = self.snake[0]
        direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x = x + direction[action][0]
        y = y + direction[action][1]

        self.episode_length += 1
        if x < 0 or x >= self.width or y < 0 or y >= self.height or (
                x, y) in self.snake:
            reward = -1
            done = True
        elif (x, y) == self.food:
            reward = 1
            self.snake.insert(0, (x, y))
            self.food = self.generate_food()
            self.update_grid()
            done = False
        else:
            fx, fy = self.food
            d = (abs(x - fx) + abs(y - fy))
            reward = 0
            self.snake.insert(0, (x, y))
            self.snake.pop()
            self.update_grid()
            done = False

        info = {}
        self.episode_reward += reward
        # 更新action_mask
        self.action_mask = self.get_mask()
        if done:
            self.episode += 1
            details = {}
            details['r'] = self.episode_reward
            details['l'] = self.episode_length
            details['e'] = self.episode
            info['episode'] = details
        return self.get_state(), reward, done, info

    def render(self, mode='human'):
        # 初始化pygame窗口
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(None, 30)
        self.window = pygame.display.set_mode(self.window_size)

        pygame.display.set_caption("Snake Game")

        if mode == 'rgb_array':
            surface = pygame.Surface(
                (self.width * self.cell_size, self.height * self.cell_size))
            self.window = surface

        self.window.fill(self.color_bg)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for y in range(self.height):
            for x in range(self.width):
                cell_value = self.grid[y, x]
                cell_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                        self.cell_size, self.cell_size)

        for y in range(self.height):
            for x in range(self.width):
                cell_value = self.grid[y, x]
                cell_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                        self.cell_size, self.cell_size)
                if cell_value == 0:  # 白色的空白格子
                    pygame.draw.rect(self.window, (255, 255, 0), cell_rect, 1)
                elif cell_value == 1:  # 贪吃蛇身体
                    pygame.draw.rect(self.window, self.color_head, cell_rect)
                elif cell_value == 2:  # 贪吃蛇身体
                    pygame.draw.rect(self.window, self.color_body, cell_rect)
                elif cell_value == 3:  # 食物
                    # pygame.draw.rect(self.window, self.color_food, cell_rect)
                    pygame.draw.circle(self.window, self.color_food,
                                       (cell_rect.x + self.cell_size // 2, cell_rect.y + self.cell_size // 2),
                                       self.cell_size // 2)

        snake_length_text = self.font.render("Length: " + str(len(self.snake)),
                                             True, (0, 25, 25))
        self.window.blit(snake_length_text, (0, 0))

        pygame.display.flip()

        if mode == 'rgb_array':
            image_array = pygame.surfarray.array3d(self.window)
            return image_array

    def close(self):
        pygame.quit()  # 释放 Pygame 占用的系统资源
        sys.exit()  # 关闭程序


if __name__ == "__main__":

    gym.register(id='Snake-v0', entry_point='snake_env:SnakeEnv')
    env = gym.make('Snake-v0')

    next_obs = env.reset()  # 初始化状态
    total_reward = 0
    step_count = 0
    done = False
    from ppo_snake import PPOAgent
    from ppo_snake import parse_args
    import torch

    args = parse_args()
    agent = PPOAgent(env.observation_space, env.action_space, args).to(args.device)
    agent = torch.load("snake-v0/2024_01_30_22_49_Snake-v0.pth")
    while not done:
        next_obs = torch.tensor(next_obs).unsqueeze(0).to(args.device)
        with torch.no_grad():
            action, logprob, entropy, value = agent.get_action_and_value(next_obs)
        next_obs, reward, done, info = env.step(action.flatten().cpu().numpy().item())
        if done:
            next_obs = env.reset()
        env.render()
        time.sleep(0.1)
        # action = 1#env.action_space.sample()  # 随机采样一个动作，此处用于演示
        # state, reward, done, info = env.step(action)  # 执行动作
        total_reward += reward
        step_count += 1
        print('At step {}, reward = {}, done = {}'.format(step_count, reward, done))
    print('Total reward: {}'.format(total_reward))
