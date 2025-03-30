# main.py
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn

# ==================== 导入自定义模块 ====================
from dqn import EnhancedDQN
from flappy_env import FlappyBirdEnv

# ==================== 设置随机种子 ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(SEED)

# ==================== 定义优先经验回放 ====================
class PrioritizedReplayBuffer:
    """优先级经验回放"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, transition):
        max_p = np.max(self.priorities[:self.size]) if self.size > 0 else 1.0
        # 确保新经验的优先级不低于现有最大值
        priority = max(max_p, 1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority  # 修正索引位置
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        clipped_errors = np.clip(np.abs(errors), 1e-4, 10)  # 数值裁剪
        for idx, error in zip(indices, clipped_errors):
            self.priorities[idx] = error ** 0.8 + 1e-6  # 平滑优先级

# ==================== 初始化环境和网络 ====================
env = FlappyBirdEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = EnhancedDQN(state_dim, action_dim).to(device)
target_net = EnhancedDQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
memory = PrioritizedReplayBuffer(capacity=10000)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

scores = []
round = 0
total_rewards = []
# ==================== 超参数设置 ====================
num_episodes = 500
batch_size = 512
target_update_freq = 4  # 目标网络更新频率
gamma = 0.95
epsilon = 0.9
epsilon_start = epsilon
epsilon_min = 0.01
epsilon_decay = 0.98
tau = 0.005

# ==================== 预填充经验回放缓冲区 ====================
while memory.size < 10000:  # 确保有足够的样本
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        # noinspection PyTupleAssignmentBalance
        next_state, reward, done = env.step(action)
        memory.add((state, action, reward, next_state, done))
        state = next_state


# ==================== 开始训练 ====================
hidden = None
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        # ε-greedy策略选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(
                    torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
                )
                hidden = (hidden[0].detach(), hidden[1].detach()) if hidden else None
                action = q_values.argmax().item()

        # 执行动作并存储经验
        # noinspection PyTupleAssignmentBalance
        next_state, reward, done = env.step(action)
        memory.add((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        # 训练网络
        if memory.size >= batch_size:
            samples, indices, weights = memory.sample(batch_size)
            states = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32).to(device)
            actions = torch.tensor([s[1] for s in samples], dtype=torch.int64).to(device).unsqueeze(1)
            rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32).to(device).unsqueeze(1)
            next_states = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32).to(device)
            dones = torch.tensor([s[4] for s in samples], dtype=torch.float32).to(device).unsqueeze(1)

            current_q = policy_net(states)
            with torch.no_grad():
                # 使用target_net计算next_q_values
                next_q_values = target_net(next_states)
                next_actions = next_q_values.max(1)[1]
                target_q = target_net(next_states)
                next_q = target_q.gather(1, next_actions.unsqueeze(1))

            target = rewards + gamma * next_q * (1 - dones)
            ttd_errors = torch.abs(current_q - target).detach().cpu().numpy().flatten()
            memory.update_priorities(indices, ttd_errors)

            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device).unsqueeze(1)
            loss = (weights_tensor * nn.MSELoss(reduction='none')(current_q, target)).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # 添加梯度裁剪
            optimizer.step()

    # 硬更新目标网络

    if episode % target_update_freq == 0:
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    # 衰减探索率
    epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))

    if episode > int(num_episodes * 0.8):  # 80%训练时间后停止探索
        epsilon = epsilon_min
    if episode > 50 and np.mean(scores[-10:]) < 1:
        epsilon = min(0.5, epsilon + 0.2)  # 当连续10局得分<1时增强探索

    if episode % 10 == 0:
        avg_q = current_q.mean().item()
        max_q = current_q.max().item()
        td_error = ttd_errors.mean()
        print(f"Avg Q: {avg_q:.3f} | Max Q: {max_q:.3f} | TD Error: {td_error:.3f}")


    print(f"Episode {episode}: Score={env.score}, Total Reward={total_reward:.2f}")
    if env.score > 30:
        round = episode
        break

    scores.append(env.score)
    total_rewards.append(total_reward)

# 绘制训练曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(scores, label='Score')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(total_rewards, label='Total Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()

plt.tight_layout()
plt.show()

torch.save(policy_net.state_dict(), f'flappy_best_model(256-128-64){round}.pth')
print("模型已保存为"f'flappy_best_model{round}.pth')
