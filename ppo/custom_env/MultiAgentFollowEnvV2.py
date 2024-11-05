import datetime
import io
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
import matplotlib.animation as animation

import imageio

class MultiAgentFollowEnv(gym.Env):
    def __init__(self, target_info):
        np.random.seed(0)
        
        self.num_agents = target_info["num_agents"]

        self.episode_length = 0
        self.episode_rewards = [0]*self.num_agents
        self.episode = 0


        # 团队奖励和个人奖励的加权
        self.tau=0.9

        # 目标状态
        self.target_distances=[]
        self.target_angles=[]

        for i in range(self.num_agents):
            self.target_distances.append(target_info[str(i)][0])
            self.target_angles.append(target_info[str(i)][1])
        
        self.target_distances=np.array(self.target_distances)
        self.target_angles=np.array(self.target_angles)

        # 定义状态空间：每个智能体包括目标位置、智能体位置、速度和方向
        pi = np.pi
        inf=np.inf
        self.id=[0]*self.num_agents
        self.id_=[1]*self.num_agents
        self.low = np.array(self.id+[-100.0, -100.0, -100.0,-100.0, 0.0, 0])
        self.high = np.array(self.id_+[100.0,  100.0,100.0,100.0,1.0, 2*pi])
        self.observation_space = spaces.Box(self.low, self.high,(self.num_agents+2+2+2,), dtype=float)

        # 定义动作空间：每个智能体包括速度和方向两个维度，多维离散
        self.action_space=gym.spaces.MultiDiscrete([3,3])
        # self.action_space = spaces.Tuple([
        #     spaces.Discrete(3),  # 速度：加速、减速、保持
        #     spaces.Discrete(3)   # 方向：向左、向右、保持
        # ])


        # 定义其他环境参数
        self.max_steps = 150
        self.current_step = 0

        # 到达次数记录以及时间记录
        self.arrive_count=0
        self.arrive_time=0

        self.fig, self.ax = plt.subplots()

        # 帧列表
        self.frames=[]

        # 设置矩形边界限制
        self.x_min, self.x_max = 0,3
        self.y_min, self.y_max = 0,3
        self.main_speed = 0.05
        self.move_directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
        self.current_direction_index = 0

        # 重置环境状态，随机设置目标位置和每个智能体初始位置、速度、方向
        self.main_position = np.random.uniform(self.x_min, self.x_max, size=(2,))

        # 随机设置每个智能体初始位置，确保距离目标一定距离
        self.agent_positions = np.random.uniform(self.x_min, self.x_max, size=(self.num_agents,2))
        
        # 定义每个实体的速度和动作
        self.speeds = np.zeros(self.num_agents)
        self.directions = np.zeros(self.num_agents)


    def update_main_position(self):
        # 沿着当前方向移动
        displacement = self.main_speed * self.move_directions[self.current_direction_index]
        new_main_position = self.main_position + displacement

        # 运行轨迹矩形边框
        x_min, x_max = 0.5,2.5
        y_min, y_max = 0.5,2.5

        # 判断是否需要改变方向
        if not (x_min <= new_main_position[0] <= x_max) or \
           not (y_min <= new_main_position[1] <= y_max):
            # 如果超出边界，改变方向
            self.current_direction_index = (self.current_direction_index + 1) % 4

            # 重新计算移动
            displacement = self.main_speed * self.move_directions[self.current_direction_index]
           
            new_main_position = self.main_position + displacement

        # 更新主目标位置
        self.main_position = new_main_position
        
        # 重置环境状态，随机设置目标位置和每个智能体初始位置、速度、方向
        self.main_position = np.clip(self.main_position,x_min, x_max)


    def calculate_target_coordinates(self):

        # Calculate x and y coordinates of the targets
        target_x = self.main_position[0] + self.target_distances * np.cos(self.target_angles)
        target_y = self.main_position[1] + self.target_distances * np.sin(self.target_angles)

        # Now, target_x and target_y contain the coordinates of each target
        target_coordinates = np.column_stack((target_x, target_y))

        return target_coordinates

    def reset(self,seed=0):
        # np.random.seed(seed)
        
        pi = np.pi
        inf=np.inf
        
        self.frames=[]

        # 重置环境记录
        self.episode_length=0
        self.episode_rewards=[0]*self.num_agents

        # 重置环境状态，随机设置参考点位置和每个智能体初始位置、速度、方向
        self.main_position = np.random.uniform(self.x_min, self.x_max, size=(2,))

        # 随机设置每个智能体初始位置，确保距离目标一定距离
        self.agent_positions = np.random.uniform(self.x_min, self.x_max, size=(self.num_agents,2))
        
        self.speeds = np.zeros(self.num_agents)
        self.directions = np.zeros(self.num_agents)

        self.current_step = 0

        # 使用广播操作计算每个代理与其他所有代理位置的差异
        differences = self.agent_positions[:, np.newaxis, :] - self.agent_positions[np.newaxis, :, :]
        # 使用 np.linalg.norm 计算每一行的2范数，即每个代理与其他所有代理之间的欧式距离
        distances = np.linalg.norm(differences, axis=2)
        # 找到距离当前智能体最小的其他智能体位置
        try:
            second_min_indices = np.argsort(distances, axis=1)[:, 1]
            second_min_values = distances[np.arange(len(distances)), second_min_indices]
            done=done or np.any(second_min_values<1)
            print(second_min_values)
            if np.any(second_min_values<1):
                print(second_min_values)
                print("碰撞")
        except:
            second_min_indices=[0]*self.num_agents


        # 返回新的状态，包括目标位置、智能体位置、速度和方向
        states=[]
        for i in range(self.num_agents):
            self.id[i]=1
            states.append(np.concatenate([self.id,self.agent_positions[i]-self.main_position,self.agent_positions[i]-self.agent_positions[second_min_indices[i]],self.speeds[i:i+1],self.directions[i:i+1]]))
            self.id[i]=0

        return states

    def step(self, action):
        # 解析动作为速度和方向
        for i in range(self.num_agents):
            speed_action, direction_action = action[i]

            # 根据速度动作更新速度
            if speed_action == 0:  # 减速
                self.speeds[i] -= 0.1
                self.speeds[i]=max(self.speeds[i],self.low[-2])
            elif speed_action == 1:  # 加速
                self.speeds[i] += 0.1
                self.speeds[i]=min(self.speeds[i],self.high[-2])

            # 根据方向动作更新方向
            if direction_action == 0:  # 向左
                self.directions[i] -= 0.1
            elif direction_action == 1:  # 向右
                self.directions[i] += 0.1
            self.directions[i]=np.mod(self.directions,2*np.pi)
        
        
        # 计算每个智能体的位移
        displacements = self.speeds[:, np.newaxis] * np.column_stack([np.cos(self.directions), np.sin(self.directions)])

        # 更新每个智能体位置
        self.agent_positions += displacements
        
        # 超出范围则进行取余数操作
        self.agent_positions = np.clip(self.agent_positions, self.x_min,self.x_max)

        self.update_main_position()

        # 计算每个代理相对于参考点的位置差异
        target_positions=self.calculate_target_coordinates()
        differences = self.agent_positions - target_positions

        # 计算奖励
        
        distances_to_target = np.linalg.norm(differences, axis=1)
        
        # main位置取余
        self.main_position = np.clip(self.main_position,0, 5)

        # 计算个人奖励
        distance_reward=-0.1*distances_to_target

        rewards = distance_reward # 距离倒数作为奖励，目标是最小化距离

        # 计算团队奖励
        team_reward=np.mean(rewards)

        # 判断是否达到终止条件
        done =  self.current_step >= self.max_steps
        if all(distances < 0.1 for distances in distances_to_target):
            team_reward+=1
        
        # 最终奖励
        rewards=rewards*self.tau+team_reward*(1-self.tau)

        self.current_step += 1

        # 使用广播操作计算每个代理与其他所有代理位置的差异
        differences = self.agent_positions[:, np.newaxis, :] - self.agent_positions[np.newaxis, :, :]
        # 使用 np.linalg.norm 计算每一行的2范数，即每个代理与其他所有代理之间的欧式距离
        distances = np.linalg.norm(differences, axis=2)
        # 找到距离当前智能体最小的其他智能体位置
        try:
            second_min_indices = np.argsort(distances, axis=1)[:, 1]
            second_min_values = distances[np.arange(len(distances)), second_min_indices]
            done=done or np.any(second_min_values<1)
            if np.any(second_min_values<1):
                print(second_min_values)
                print("碰撞")
        except:
            second_min_indices=[0]*self.num_agents      
        
        # 返回新的状态，包括目标位置、智能体位置、速度和方向
        states=[]
        
        for i in range(self.num_agents):
            self.id[i]=1
            states.append(np.concatenate([self.id,self.agent_positions[i]-target_positions[i],self.agent_positions[i]-self.agent_positions[second_min_indices[i]],self.speeds[i:i+1],self.directions[i:i+1]]))
            self.id[i]=0
            self.episode_rewards[i]=rewards[i]

        # 组装info
        self.episode_length+=1

        info={}
        if done:
            self.episode += 1
            details = {}
            details['r'] = self.episode_rewards
            details['l'] = self.episode_length-1
            details['e'] = self.episode
            info['episode'] = details

            if len(self.frames)!=0:
                timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
                save_path = f"Follow/{timestamp}.gif"
                imageio.mimsave(save_path, self.frames, duration=0.1)  # Adjust duration as needed
                print("GIF saved successfully.")

            # print("*"*20)
            # print("距离reward",distance_reward)
            # print("方位reward",angle_reward,0.1*angle_reward)

            # print("*"*20)

        return states,rewards,done,info

    def add_arrow(self, position, direction, color='black'):
        # 添加箭头
        arrow_length = 0.1
        arrow_head_width = 0.05

        arrow_dx = arrow_length * np.cos(direction)
        arrow_dy = arrow_length * np.sin(direction)

        self.ax.arrow(position[0], position[1], arrow_dx, arrow_dy, color=color, width=arrow_head_width)

    def render(self, mode="human"):
        # 清空子图内容
        self.ax.clear()

        # 绘制目标
        self.ax.plot(self.main_position[0], self.main_position[1], 'rs', label='main')

        # 绘制每个智能体
        for i in range(self.num_agents):
            self.ax.plot(self.agent_positions[i, 0], self.agent_positions[i, 1], 'bo', label=f'agent {i + 1}')

            # 添加运动方向箭头
            self.add_arrow(self.agent_positions[i], self.directions[i])

        # 设置图形范围
        self.ax.set_xlim(self.x_min,self.x_max)  # 根据需要调整范围
        self.ax.set_ylim(self.y_min,self.y_max)  # 根据需要调整范围

        # 添加图例
        self.ax.legend()

        if mode == "human":
            # 显示图形
            plt.pause(0.01)  # 添加短暂的时间间隔，单位为秒
        else:
            # 将当前帧的图形添加到列表中
            self.frames.append(self.fig_to_array())
            
    def fig_to_array(self):
        # Convert the current figure to an array of pixels
        buf = io.BytesIO()
        self.ax.figure.savefig(buf, format='png')
        buf.seek(0)
        img = imageio.imread(buf)
        return img


if __name__ == "__main__":


    pi=np.pi
    target_info={"num_agents":1,"0":(0,0),"1":(10,pi/3),"2":(20,pi/6),"3":(20,pi/3)}

    # 循环测试代码
    env = MultiAgentFollowEnv(target_info)

    state = env.reset()  # 重置环境
    done = False

    while not done:
        # 随机生成每个智能体的离散动作，其中第一维控制速度，有加速、减速、保持，
        # 第二维控制方向，有向左、向右、保持
        actions = np.random.randint(3, size=(env.num_agents, 2))
        
        actions=np.array([[1,1]])

        states, rewards, done, _ = env.step(actions)  # 执行动作

        print(rewards)
        
        env.render(mode="human")  # 渲染环境状态

    print("Testing complete.")
