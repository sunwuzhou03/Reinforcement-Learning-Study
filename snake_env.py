import gym
import numpy as np
import random
import pygame
import sys


class SnakeEnv(gym.Env):
    def __init__(self, width=10, height=10):
        super(SnakeEnv, self).__init__()

        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.snake = [(0, 0)]
        self.food = self.generate_food()
        self.direction = "right"
        self.time = 1
        self.cell_size = 40
        self.window_size = (self.width * self.cell_size,
                            self.height * self.cell_size)
        # 定义颜色
        self.color_bg = (255, 255, 255)
        self.color_head = (0, 120, 120)
        self.color_snake = (0, 255, 0)
        self.color_food = (255, 0, 0)

        # 定义贪吃蛇环境的观测空间和行动空间
        inf = self.width + self.height
        low = np.array([-inf, -1.0, 0, 0, 0, 0])  # 连续状态空间的最小值
        high = np.array([inf, 1.0, 1.0, 1.0, 1.0, 1.0])  # 连续状态空间的最大值
        continuous_space = gym.spaces.Box(low, high, shape=(6, ), dtype=float)
        self.observation_space = continuous_space

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
        self.direction = "right"
        self.update_grid()

        return self.get_state()

    def get_state(self):
        state = []
        x, y = self.snake[0]
        fx, fy = self.food
        xbase = self.width
        ybase = self.height
        x_norm, y_norm = (fx - x) / xbase, (fy - y) / ybase
        state.append(fx - x)
        state.append(fy - y)
        for gx, gy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dx, dy = x + gx, y + gy
            if dx < 0 or dy < 0 or dx >= self.width or dy >= self.height or (
                    dx, dy) in self.snake:
                state.append(0)
                continue
            else:
                state.append(1)  #四个方向可以走
        return np.array(state)

    def update_grid(self):
        self.grid = np.zeros((self.height, self.width))
        x, y = self.snake[0]
        self.grid[y, x] = 1
        for x, y in self.snake[1:]:
            self.grid[y, x] = 2
        fx, fy = self.food
        self.grid[fy, fx] = 3

    def step(self, action):
        x, y = self.snake[0]
        if action == 0:  # 左
            if self.direction != "right":
                self.direction = "left"
        elif action == 1:  # 上
            if self.direction != "down":
                self.direction = "up"
        elif action == 2:  # 右
            if self.direction != "left":
                self.direction = "right"
        elif action == 3:  # 下
            if self.direction != "up":
                self.direction = "down"

        if self.direction == "left":
            x -= 1
        elif self.direction == "up":
            y -= 1
        elif self.direction == "right":
            x += 1
        elif self.direction == "down":
            y += 1

        if x < 0 or x >= self.width or y < 0 or y >= self.height or (
                x, y) in self.snake:
            reward = -0.5 - 1 / (len(self.snake))
            self.time = 1
            done = True
        elif (x, y) == self.food:
            reward = 4 + len(self.snake) * 0.1
            self.time = 1
            self.snake.insert(0, (x, y))
            self.food = self.generate_food()
            self.update_grid()
            done = False
        else:
            fx, fy = self.food
            d = (abs(x - fx) + abs(y - fy))
            reward = 0.1 * (2 - d) / (self.time)
            self.snake.insert(0, (x, y))
            self.snake.pop()
            self.update_grid()
            done = False
            self.time += 1
        return self.get_state(), reward, done, {}

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
        snake_length_text = self.font.render("Length: " + str(len(self.snake)),
                                             True, (0, 25, 25))
        self.window.blit(snake_length_text, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for y in range(self.height):
            for x in range(self.width):
                cell_value = self.grid[y, x]
                cell_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                        self.cell_size, self.cell_size)
                if cell_value == 1:  # 贪吃蛇身体
                    pygame.draw.rect(self.window, self.color_head, cell_rect)
                elif cell_value == 2:  # 贪吃蛇身体
                    pygame.draw.rect(self.window, self.color_snake, cell_rect)
                elif cell_value == 3:  # 食物
                    pygame.draw.rect(self.window, self.color_food, cell_rect)

        pygame.display.flip()

        if mode == 'rgb_array':
            image_array = pygame.surfarray.array3d(self.window)
            return image_array
