import os
import numpy as np
import torch
import gym
import random
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from  torch.distributions.categorical import Categorical
from distutils.util import strtobool
import argparse
import datetime

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="Snake-v0",
                        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


class RolloutBuffer:
    def __init__(self, obs_space, action_space, args):
        num_envs = args.num_envs
        max_steps = args.num_steps
        device = args.device
        self.observations = torch.zeros((max_steps, num_envs) + obs_space.shape).to(device)
        self.actions = torch.zeros((max_steps, num_envs) + action_space.shape).to(device)
        self.rewards = torch.zeros((max_steps, num_envs)).to(device)
        self.returns = torch.zeros((max_steps, num_envs)).to(device)
        self.advantages = torch.zeros((max_steps, num_envs)).to(device)
        self.dones = torch.zeros((max_steps, num_envs)).to(device)
        self.values = torch.zeros((max_steps, num_envs)).to(device)
        self.logprobs = torch.zeros((max_steps, num_envs)).to(device)
        try:
            # multi discrete
            self.action_masks = torch.zeros((args.num_steps, args.num_envs) + (action_space.nvec.sum(),)).to(device)
        except:
            # discrete
            self.action_masks = torch.zeros((args.num_steps, args.num_envs) + (action_space.n,)).to(device)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def compute_advantage(agent,buffer, args, next_obs, next_done):
    # 计算优势
    with torch.no_grad():
        next_value = agent.get_value(next_obs)
        if args.gae:
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps-1:
                    nextnonterminal = 1 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1 - buffer.dones[t].float()
                    nextvalues = buffer.values[t + 1]
                delta = buffer.rewards[t] + args.gamma * nextvalues * nextnonterminal - buffer.values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                buffer.advantages[t] = lastgaelam
            buffer.returns = buffer.advantages + buffer.values
        else:
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - buffer.dones[t + 1].float()
                    next_return = buffer.returns[t + 1]
                buffer.returns[t] = buffer.rewards[t] + args.gamma * nextnonterminal * next_return
            buffer.advantages = buffer.returns - buffer.values

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[],device="cuda"):
        self.masks = masks
        self.device=device
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)

class PPOAgent(nn.Module):
    def __init__(self, obs_space, action_space,args) -> None:
        super(PPOAgent, self).__init__()
        self.action_space = action_space
        self.obs_space = obs_space
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, self.action_space.n), std=1.0),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': args.learning_rate},
            {'params': self.critic.parameters(), 'lr': args.learning_rate}
        ],eps=1e-5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action_mask=None,action=None):
        logits = self.actor(x)
        if action_mask is not None:
            probs = CategoricalMasked(logits=logits, masks=action_mask)
        else:
            probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()

        return action, logprob, entropy, self.critic(x)


class Learner(nn.Module):
    def __init__(self, agent: PPOAgent, args):
        super(Learner, self).__init__()
        self.action_space = agent.action_space
        self.obs_space = agent.obs_space

    def forward(self, agent,buffer,args):
        # flatten the batch
        b_obs = buffer.observations.reshape((-1,) + self.obs_space.shape)
        b_logprobs = buffer.logprobs.reshape(-1)
        b_actions = buffer.actions.reshape((-1,) + self.action_space.shape)
        b_advantages = buffer.advantages.reshape(-1)
        b_returns = buffer.returns.reshape(-1)
        b_values = buffer.values.reshape(-1)
        b_action_masks = buffer.action_masks.reshape((-1, buffer.action_masks.shape[-1]))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],b_action_masks[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                agent.optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return v_loss.item(),pg_loss.item(),entropy_loss.item(),old_approx_kl.item(),approx_kl.item(),np.mean(clipfracs),explained_var

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    args = parse_args()

    global_step = 0

    gym.register(id='Snake-v0', entry_point='snake_env:SnakeEnv')
    env = gym.make('Snake-v0')

    buffer = RolloutBuffer(env.observation_space, env.action_space, args)
    agent = PPOAgent(env.observation_space, env.action_space,args).to(args.device)
    learner = Learner(agent, args)

    # next_obs,info= env.reset()
    next_obs=env.reset()
    next_obs = torch.tensor(next_obs).unsqueeze(0).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)

    num_updates = args.total_timesteps // args.batch_size

    start_time = time.time()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    mx_rs=-float('inf')
    # 与环境交互
    for update in range(1,num_updates+1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            agent.optimizer.param_groups[0]["lr"] = lrnow

        done = False

        for step in range(0,args.num_steps):
            global_step += 1 * args.num_envs
            buffer.observations[step] = next_obs
            buffer.dones[step] = next_done

            buffer.action_masks[step] = torch.tensor(np.array(env.action_mask))
            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(next_obs,buffer.action_masks[step])
                buffer.values[step] = value.flatten()
            buffer.actions[step] = action
            buffer.logprobs[step] = logprob

            next_obs, reward, done, info = env.step(action.flatten().cpu().numpy().item())
            if done:
                next_obs=env.reset()

            if "episode" in info.keys():
                if mx_rs<info['episode']['r'] and info['episode']['r']>30:
                    mx_rs=info['episode']['r']
                    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
                    save_path = f"snake-v0mask/{timestamp}_{args.gym_id}.pth"
                    torch.save(agent, save_path)  # 保存模型参数
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            buffer.rewards[step] = torch.tensor(np.clip(reward,-10,10)).to(args.device).view(-1)
            next_obs, next_done = torch.tensor(next_obs).unsqueeze(0).to(args.device), torch.tensor(done).to(args.device)

        compute_advantage(agent,buffer, args, next_obs, next_done)
        v_loss,pg_loss,entropy_loss,old_approx_kl,approx_kl,clipfracs_mean,explained_var=learner(agent,buffer,args)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        writer.add_scalar("losses/entropy", entropy_loss, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", clipfracs_mean, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    env.close()
    writer.close()