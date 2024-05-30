import gym
from gym import spaces
import numpy as np
import socket
import pickle
from multiprocessing import Process
import time
import subprocess


class TcpSimulationEnv(gym.Env):
    def __init__(self, host='localhost', port=65432):
        super(TcpSimulationEnv, self).__init__()
        self.host = host
        self.port = port
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(4,), dtype=np.float32)
        self.connection = None

    def _send_command(self, command):
        if self.connection is None:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.host, self.port))
        self.connection.sendall(pickle.dumps(command))
        data = self.connection.recv(4096)
        return pickle.loads(data)

    def reset(self):
        command = {'action': 'reset'}
        observation = self._send_command(command)
        return observation

    def step(self, action):
        command = {'action': 'step', 'value': action}
        observation, reward, done, info = self._send_command(command)
        return observation, reward, done, info

    def close(self):
        if self.connection:
            command = {'action': 'close'}
            self._send_command(command)
            self.connection.close()
            self.connection = None


def start_server(port):
    subprocess.Popen(['python', 'server.py', '--port', str(port)])


if __name__ == '__main__':
    ports = [65432, 65433, 65434]  # 示例端口列表，假设我们启动了3个环境实例

    # 启动服务器进程
    for port in ports:
        start_server(port)
        time.sleep(1)  # 等待服务器启动

    envs = [TcpSimulationEnv(port=port) for port in ports]

    num_steps = 10  # 定义要测试的步数

    for env in envs:
        observation = env.reset()
        print(f'Initial observation from env on port {env.port}: {observation}')

    for step in range(num_steps):
        for env in envs:
            action = env.action_space.sample()  # 随机选择一个动作
            observation, reward, done, info = env.step(action)
            print(f'Step {step + 1} from env on port {env.port}:')
            print(f'  Action: {action}')
            print(f'  Observation: {observation}')
            print(f'  Reward: {reward}')
            print(f'  Done: {done}')
            print(f'  Info: {info}')
            if done:
                print(f'Environment on port {env.port} done, resetting...')
                observation = env.reset()
                print(f'New initial observation from env on port {env.port}: {observation}')

    for env in envs:
        env.close()
