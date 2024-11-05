import numpy as np
import matplotlib.pyplot as plt

def plot_agent_positions(target_info):
    num_agents = target_info["num_agents"]

    # 主位置
    main_position = np.array([0, 0])

    # 初始化代理位置数组
    agent_positions = np.zeros((num_agents, 2))

    # 逐个处理每个代理
    for i in range(num_agents):
        distance, angle = target_info[str(i)]
        # 计算代理相对于主位置的位置
        agent_positions[i, 0] = distance * np.cos(angle)
        agent_positions[i, 1] = distance * np.sin(angle)

    # 绘制代理位置和主位置
    plt.scatter(agent_positions[:, 0], agent_positions[:, 1], label="Agents", marker='o', s=100)
    plt.scatter(main_position[0], main_position[1], color='red', label="Main Position", marker='s', s=200)
    
    # 添加注释
    for i, values in target_info.items():
        if i != "num_agents":
            distance, angle = values
            plt.annotate(f"Agent {i}\nDistance: {distance}\nAngle: {angle:.2f}", 
                         xy=(agent_positions[int(i), 0], agent_positions[int(i), 1]), 
                         xytext=(agent_positions[int(i), 0]+1, agent_positions[int(i), 1]+1),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize=8, ha='center', va='bottom')

    # 设置图形属性
    plt.title("Agent Positions Relative to Main")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 使坐标轴比例相等
    plt.show()

# 示例数据
pi=np.pi
target_info={"num_agents":4,"0":(5,pi/6),"1":(5,pi/3),"2":(10,pi/6),"3":(10,pi/3)}


# 调用函数进行绘图
plot_agent_positions(target_info)
