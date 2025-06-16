import random
from collections import deque
import os
from model import PolicyNet, ValueNet, QNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区

        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        从缓冲区采样一批经验

        :param batch_size: 批大小
        :return: 一批经验 (state, action, reward, next_state, done)
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        获取缓冲区当前大小

        :return: 缓冲区大小
        """
        return len(self.buffer)


from abc import ABC, abstractmethod


class AgentBase(ABC):
    def __init__(self, state_size, action_size, firm_id, gamma):
        """
        初始化Agent

        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        :param firm_id: 企业ID，用于标识训练哪个企业
        :param gamma: 折扣因子
        """
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.gamma = gamma

    def get_name(self):
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """
        执行一步训练

        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        pass

    @abstractmethod
    def act(self, state):
        """
        根据当前状态选择动作

        :param state: 当前状态
        :param epsilon: 探索参数
        :return: 选择的动作
        """
        pass

    @abstractmethod
    def learn(self, experiences):
        pass

    @abstractmethod
    def save(self, filename):

        pass

    @abstractmethod
    def load(self, filename):
        """
        加载模型参数

        :param filename: 文件名
        """
        pass


# 定义DQN智能体
class AgentDQN(AgentBase):
    def __init__(
        self,
        state_size,
        action_size,
        firm_id,
        max_order=20,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        tau=1e-3,
        update_every=4,
    ):
        """
        初始化DQN智能体

        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        :param firm_id: 企业ID，用于标识训练哪个企业
        :param max_order: 最大订单量，用于离散化动作空间
        :param buffer_size: 回放缓冲区大小
        :param batch_size: 批大小
        :param gamma: 折扣因子
        :param learning_rate: 学习率
        :param tau: 软更新参数
        :param update_every: 更新目标网络的频率
        """
        super(AgentDQN, self).__init__(state_size, action_size, firm_id, gamma)

        self.max_order = max_order
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.learning_step = 0
        self.epsilon = 0.2
        # 创建Q网络和目标网络
        self.q_network = QNet(state_size, 64, action_size)
        self.target_network = QNet(state_size, 64, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 设置优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)

        # 跟踪训练进度
        self.t_step = 0

    def get_name(self):
        return "DQN"

    def step(self, state, action, reward, next_state, done):
        # 添加经验到回放缓冲区
        self.memory.add(state, action, reward, next_state, done)

        # 每隔一定步数学习
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state):
        """
        根据当前状态选择动作

        :param state: 当前状态
        :return: 选择的动作
        """
        # 从3维numpy数组转换为1维向量
        state = torch.from_numpy(state.flatten()).float().unsqueeze(0)

        # 切换到评估模式
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        # 切换回训练模式
        self.q_network.train()

        # epsilon-贪婪策略
        if random.random() > self.epsilon:
            return (
                np.argmax(action_values.cpu().data.numpy()) + 1
            )  # +1 因为我们的动作从1开始
        else:
            return random.randint(1, self.max_order)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack([s.flatten() for s in states])).float()
        actions = torch.from_numpy(
            np.vstack([a - 1 for a in actions])
        ).long()  # -1 因为我们的动作从1开始，但索引从0开始
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(
            np.vstack([ns.flatten() for ns in next_states])
        ).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()
        # 从目标网络获取下一个状态的最大预测Q值
        Q_targets_next = (
            self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        )

        # 计算目标Q值
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # 获取当前Q值估计
        Q_expected = self.q_network(states).gather(1, actions)

        # 计算损失
        loss = nn.MSELoss()(Q_expected, Q_targets)

        # 最小化损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.learning_step += 1
        if self.learning_step % self.update_every == 0:
            self.soft_update()

        return loss.item()

    def soft_update(self):
        """
        软更新目标网络参数：θ_target = τ*θ_local + (1-τ)*θ_target
        """
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )
        print(f"模型已保存到 {filename}")

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"从 {filename} 加载了模型")
            return True
        return False


class AgentPPO(AgentBase):
    def __init__(
        self,
        state_size,
        action_size,
        firm_id,
        gamma=0.99,
        actor_lr=2e-4,
        critic_lr=5e-3,
        epoch=10,
        clip_eps=0.2,
        lmbda=0.95,
    ):
        """
        初始化A2C智能体
        """
        super(AgentPPO, self).__init__(state_size, action_size, firm_id, gamma)
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.gamma = gamma
        self.learning_step = 0
        self.epoch = epoch
        self.clip_eps = clip_eps
        self.lmbda = lmbda

        self.trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }

        # 创建策略网络和价值网络
        self.actor = PolicyNet(state_size, 64, action_size)
        self.critic = ValueNet(state_size, 64)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_name(self):
        return "PPO"

    def step(self, state, action, reward, next_state, done):
        self.trajectory["states"].append(state)
        self.trajectory["actions"].append(action)
        self.trajectory["rewards"].append(reward)
        self.trajectory["next_states"].append(next_state)
        self.trajectory["dones"].append(done)

        if done:
            self.learn(self.trajectory)
            self.trajectory = {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "dones": [],
            }

    def act(self, state):
        state = torch.from_numpy(state.flatten()).float().unsqueeze(0)
        self.actor.eval()

        with torch.no_grad():
            action_probs = self.actor(state)

        self.actor.train()
        action = torch.distributions.Categorical(action_probs).sample().item()

        return action + 1

    def _compute_advantage(self, td_delta):
        td_delta = td_delta.numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float32)

    def learn(self, trajectory):
        states = trajectory["states"]
        states = torch.tensor(np.array(states)).reshape(-1, 3).float()
        actions = trajectory["actions"]
        actions = torch.tensor(np.array(actions)).reshape(-1, 1).long() - 1
        rewards = trajectory["rewards"]
        rewards = torch.tensor(np.array(rewards)).reshape(-1, 1).float()
        next_states = trajectory["next_states"]
        next_states = torch.tensor(np.array(next_states)).reshape(-1, 3).float()
        dones = trajectory["dones"]
        dones = torch.tensor(np.array(dones)).reshape(-1, 1).float()
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_target = td_target.detach()
        td_delta = td_target - self.critic(states)
        td_delta = td_delta.detach()
        advantage = self._compute_advantage(td_delta)

        old_log_probs = torch.log(self.actor(states).gather(1, actions) + 1e-6).detach()

        for _ in range(self.epoch):
            log_probs = torch.log(self.actor(states).gather(1, actions) + 1e-6)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            filename,
        )
        print(f"模型已保存到 {filename}")

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
            print(f"从 {filename} 加载了模型")
            return True
        return False


class AgentSAC(AgentBase):
    def __init__(
        self,
        state_size,
        action_size,
        firm_id,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        actor_lr=5e-4,
        critic_lr=2e-3,
        tau=1e-3,
        target_entropy=-1,
    ):
        super(AgentSAC, self).__init__(state_size, action_size, firm_id, gamma)
        self.actor = PolicyNet(state_size, 64, action_size)
        self.critic1 = QNet(state_size, 64, action_size)
        self.critic2 = QNet(state_size, 64, action_size)
        self.target_critic1 = QNet(state_size, 64, action_size)
        self.target_critic2 = QNet(state_size, 64, action_size)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.memory = ReplayBuffer(buffer_size)

    def get_name(self):
        return "SAC"

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state):
        state = torch.from_numpy(state.flatten()).float().unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action_probs = self.actor(state)
        self.actor.train()
        action = torch.distributions.Categorical(action_probs).sample().item()
        return action + 1

    def _calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-6)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_next = self.target_critic1(next_states)
        q2_next = self.target_critic2(next_states)
        min_qvalue = torch.sum(
            next_probs * torch.min(q1_next, q2_next), dim=1, keepdim=True
        )
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack([s.flatten() for s in states])).float()
        actions = torch.from_numpy(
            np.vstack([a - 1 for a in actions])
        ).long()  # -1 因为我们的动作从1开始，但索引从0开始
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(
            np.vstack([ns.flatten() for ns in next_states])
        ).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        td_target = self._calc_target(rewards, next_states, dones)
        critic1_q_values = self.critic1(states).gather(1, actions)
        critic1_loss = torch.mean(F.mse_loss(critic1_q_values, td_target.detach()))
        critic2_q_values = self.critic2(states).gather(1, actions)
        critic2_loss = torch.mean(F.mse_loss(critic2_q_values, td_target.detach()))
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-6)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_values = self.critic1(states)
        q2_values = self.critic2(states)
        min_qvalue = torch.sum(
            probs * torch.min(q1_values, q2_values), dim=1, keepdim=True
        )
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
        for target_param, local_param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            },
            filename,
        )
        print(f"模型已保存到 {filename}")

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
            self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critic1_optimizer.load_state_dict(
                checkpoint["critic1_optimizer_state_dict"]
            )
            self.critic2_optimizer.load_state_dict(
                checkpoint["critic2_optimizer_state_dict"]
            )
            print(f"从 {filename} 加载了模型")
            return True
        return False
