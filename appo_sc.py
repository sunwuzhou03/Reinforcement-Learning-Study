import gym
from gym.envs.registration import register

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy
import time
import datetime

import ray
import gc
from client import start_server,TcpSimulationEnv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class ActorCritic(nn.Module):
    def __init__(self, state_dim,action_dim):
        super().__init__()
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
              ).float()
        self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Softmax(-1)
              ).float()



class Memory(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype=np.float32), np.array(self.actions[idx], dtype=np.float32), np.array(
            [self.rewards[idx]], dtype=np.float32), \
            np.array([self.dones[idx]], dtype=np.float32), np.array(self.next_states[idx], dtype=np.float32)

    def get_all(self):
        return self.states, self.actions, self.rewards, self.dones, self.next_states

    def save_all(self, states, actions, rewards, dones, next_states):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.next_states = next_states

    def save_eps(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]


class Distributions():
    def __init__(self, myDevice=None):
        self.device = myDevice if myDevice != None else device

    def sample(self, datas):
        distribution = Categorical(datas)
        return distribution.sample().float().to(self.device)

    def entropy(self, datas):
        distribution = Categorical(datas)
        return distribution.entropy().float().to(self.device)

    def logprob(self, datas, value_data):
        distribution = Categorical(datas)
        return distribution.log_prob(value_data).unsqueeze(1).float().to(self.device)

    def kl_divergence(self, datas1, datas2):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)

        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(self.device)


class PolicyFunction():
    def __init__(self, gamma=0.99, lam=0.95, policy_kl_range=0.03, policy_params=2):
        self.gamma = gamma
        self.lam = lam
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns = []

        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)

        return torch.stack(returns)

    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value
        return q_values

    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae = 0
        adv = []

        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        for step in reversed(range(len(rewards))):
            gae = delta[step] + (1.0 - dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)

        return torch.stack(adv)


class Learner():
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip,
                 entropy_coef, vf_loss_coef,clip_coef,
                 minibatch, PPO_epochs, gamma, lam, learning_rate):
        self.clip_coef=clip_coef
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.minibatch = minibatch
        self.PPO_epochs = PPO_epochs
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim

        self.learner_model = ActorCritic(state_dim, action_dim).to(device)
        self.actor_model = ActorCritic(state_dim, action_dim).to(device)

        self.optimizer = Adam(self.learner_model.parameters(), lr=learning_rate)

        self.memory = Memory()
        self.policy_function = PolicyFunction(gamma, lam)
        self.distributions = Distributions()

        if is_training_mode:
            self.learner_model.train()
        else:
            self.learner_model.train()

    def save_all(self, states, actions, rewards, dones, next_states):
        self.memory.save_all(states, actions, rewards, dones, next_states)

    # Loss for PPO
    def get_loss(self, action_probs, values, old_action_probs, old_values, next_values, actions, rewards, dones):
        # Don't use old value in backpropagation
        Old_values = old_values.detach()

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs = self.distributions.logprob(action_probs, actions)
        Old_logprobs = self.distributions.logprob(old_action_probs, actions).detach()

        # Getting general advantages estimator
        Advantages = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
        Returns = (Advantages + values).detach()
        Advantages = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach()

        ratios = (logprobs - Old_logprobs).exp()
        Kl = self.distributions.kl_divergence(old_action_probs, action_probs)

        # Combining TR-PPO with Rollback (Truly PPO)
        # pg_loss         = torch.where(
        #     (Kl >= self.policy_kl_range) & (ratios > 1),
        #     ratios * Advantages - self.policy_params * Kl,
        #     ratios * Advantages
        # )
        # pg_loss         = -pg_loss.mean()

        pg_loss1 = -Advantages * ratios
        pg_loss2 = -Advantages * torch.clamp(ratios, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Getting entropy from the action probability
        dist_entropy = self.distributions.entropy(action_probs).mean()

        # Getting critic loss by using Clipped critic value
        vpredclipped = Old_values + torch.clamp(values - Old_values, -self.value_clip,
                                                self.value_clip)  # Minimize the difference between old value and new value
        vf_losses1 = (Returns - values).pow(2) * 0.5  # Mean Squared Error
        vf_losses2 = (Returns - vpredclipped).pow(2) * 0.5  # Mean Squared Error
        critic_loss = torch.max(vf_losses1, vf_losses2).mean()

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = pg_loss + (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef)
        return loss

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states):
        action_probs, values = self.learner_model.actor(states), self.learner_model.critic(states)
        old_action_probs, old_values = self.actor_model.actor(states), self.actor_model.critic(states)
        next_values = self.learner_model.critic(next_states)

        loss = self.get_loss(action_probs, values, old_action_probs, old_values, next_values, actions, rewards, dones)

        # === Do backpropagation ===

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # === backpropagation has been finished ===

    # Update the model
    def update_ppo(self):
        batch_size = int(len(self.memory) / self.minibatch)
        dataloader = DataLoader(self.memory, batch_size, shuffle=False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for states, actions, rewards, dones, next_states in dataloader:
                self.training_ppo(states.float().to(device), actions.float().to(device), rewards.float().to(device), \
                                  dones.float().to(device), next_states.float().to(device))

        # Clear the memory
        self.memory.clear_memory()

        # Copy new weights into old policy:
        self.actor_model.load_state_dict(self.learner_model.state_dict())

    def get_weights(self):
        return self.learner_model.state_dict()

    def save_weights(self):
        torch.save(self.learner_model.state_dict(), 'agent.pth')


class Agent:
    def __init__(self, state_dim, action_dim, is_training_mode):
        self.is_training_mode = is_training_mode
        self.device = torch.device('cpu')

        self.memory = Memory()
        self.distributions = Distributions(self.device)
        self.actor_model = ActorCritic(state_dim, action_dim).to(self.device)

        if is_training_mode:
            self.actor_model.train()
        else:
            self.actor_model.eval()

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    def get_all(self):
        return self.memory.get_all()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        action_probs = self.actor_model.actor(state)

        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.distributions.sample(action_probs)
        else:
            action = torch.argmax(action_probs, 1)

        return action.cpu().item()

    def set_weights(self, weights):
        self.actor_model.load_state_dict(weights)

    def load_weights(self):
        self.actor_model.load_state_dict(torch.load('agent.pth', map_location=self.device))


@ray.remote
class Runner():
    def __init__(self, port, training_mode, render, n_update, tag):
        self.env = TcpSimulationEnv(port=port)
        self.states = self.env.reset()

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.agent = Agent(self.state_dim, self.action_dim, training_mode)

        self.render = render
        self.tag = tag
        self.training_mode = training_mode
        self.n_update = n_update

        self.last_model_load_time = time.time()

        self.load_model_start_time= self.last_model_load_time

        self.load_model_cnt = 0


    def run_episode(self, i_episode, total_reward, eps_time):
        # self.agent.load_weights()
        # #
        # print(f"process {self.tag} load new model!")


        self.agent.load_weights()  # 载入新模型
        self.load_model_cnt += 1
        print(
            f"Process {self.tag} loaded new model, load cnt is {self.load_model_cnt},load time is {time.time() - self.load_model_start_time} !")
        self.last_model_load_time = time.time()  # 更新载入时间

        for _ in range(self.n_update):

            # if time.time() - self.last_model_load_time >= 10:
            #     self.agent.load_weights()  # 载入新模型
            #     self.load_model_cnt += 1
            #     print(f"Process {self.tag} loaded new model, load cnt is {self.load_model_cnt},load time is {time.time()-self.load_model_start_time} !")
            #     self.last_model_load_time = time.time()  # 更新载入时间

            action = int(self.agent.act(self.states))
            next_state, reward, done, _ = self.env.step(action)

            eps_time += 1
            total_reward += reward

            if self.training_mode:
                self.agent.save_eps(self.states.tolist(), action, reward, float(done), next_state.tolist())

            self.states = next_state

            if self.render:
                self.env.render()

            if done:
                self.states = self.env.reset()
                i_episode += 1
                print('Episode {} \t t_reward: {} \t time: {} \t process no: {} \t'.format(i_episode, total_reward,eps_time, self.tag))

                total_reward = 0
                eps_time = 0

        return self.agent.get_all(), i_episode, total_reward, eps_time, self.tag


def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def main():
    ############## Hyperparameters ##############
    training_mode = True  # If you want to train the agent, set this to True. But set this otherwise if you only want to test it

    render = False  # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_update = 128  # How many episode before you update the Policy. Recommended set to 1024 for Continous
    n_episode = 100000  # How many episode you want to run
    n_agent = 3  # How many agent you want to run asynchronously

    clip_coef=0.2
    policy_kl_range = 0.0008  # Recommended set to 0.03 for Continous
    policy_params = 20  # Recommended set to 5 for Continous
    value_clip = 1.0  # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef = 0.05  # How much randomness of action you will get. Because we use Standard Deviation for Continous, no need to use Entropy for randomness
    vf_loss_coef = 1.0  # Just set to 1
    minibatch = 4  # How many batch per update. size of batch = n_update / minibatch. Recommended set to 32 for Continous
    PPO_epochs = 1  # How many epoch per update. Recommended set to 10 for Continous

    gamma = 0.99  # Just set to 0.99
    lam = 0.95  # Just set to 0.95
    learning_rate = 2.5e-4  # Just set to 0.95
    #############################################
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    learner = Learner(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef,
                      vf_loss_coef,clip_coef,
                      minibatch, PPO_epochs, gamma, lam, learning_rate)
    #############################################
    start = time.time()

    ray.init()
    try:
        ports = [65432, 65433, 65434]
        runners = [Runner.remote(ports[i], training_mode, render, n_update, i) for i in range(n_agent)]
        learner.save_weights()

        episode_ids = []
        for i, runner in enumerate(runners):
            episode_ids.append(runner.run_episode.remote(i, 0, 0))
            time.sleep(0.1)

        for _ in range(1, n_episode + 1):
            ready, not_ready = ray.wait(episode_ids)

            trajectory, i_episode, total_reward, eps_time, tag = ray.get(ready)[0]

            states, actions, rewards, dones, next_states = trajectory
            learner.save_all(states, actions, rewards, dones, next_states)

            learner.update_ppo()
            # print(f"use process {tag} to update learner")
            learner.save_weights()

            episode_ids = not_ready
            episode_ids.append(runners[tag].run_episode.remote(i_episode, total_reward, eps_time))

            gc.collect()
    except KeyboardInterrupt:
        print('\nTraining has been Shutdown \n')
    finally:
        ray.shutdown()

        finish = time.time()
        timedelta = finish - start
        print('Timelength: {}'.format(str(datetime.timedelta(seconds=timedelta))))


if __name__ == '__main__':
    ports = [65432, 65433, 65434]  # 示例端口列表，假设我们启动了3个环境实例

    # 启动服务器进程
    for port in ports:
        start_server(port)
        time.sleep(1)  # 等待服务器启动

    main()