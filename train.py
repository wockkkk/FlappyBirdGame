import random
import numpy as np
import torch
from flappy_env import FlappyBirdEnv
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
from dqn128_64_32 import EnhancedDQN as DQN128
from dqn256_128_64 import EnhancedDQN as DQN256
from dqn256_128_64_32 import EnhancedDQN as DQN256_2

# ==================== 设置随机种子 ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(SEED)


# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 初始化环境和网络
env = FlappyBirdEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN256_2(state_dim, action_dim).to(device)
target_net = DQN256_2(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)
memory = ReplayBuffer(100000)

# 超参数设置
num_episodes = 5000
batch_size = 512
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99

# 训练循环
scores = []
clock_list = []
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
                q_values = policy_net(torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0))
                action = q_values.argmax().item()

        # 执行动作并存储经验
        next_state, reward, done, _, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # 从经验池中采样并训练网络
        if len(memory) >= batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.int64).to(device).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device).unsqueeze(1)
            current_q = policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target = rewards + gamma * next_q * (1 - dones)

            loss = nn.MSELoss()(current_q, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 更新目标网络
    if episode % 10 == 0 or env.score != 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}: Total Reward={total_reward:.2f}, score={env.score}, clock={env.clock}")
    if env.score >= 30:
        break
    # 衰减探索率
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # 记录分数
    scores.append(env.score)
    clock_list.append(env.clock)

model_save_path = f'256_2_model{episode}round_{env.score}score.pth'
torch.save(policy_net.state_dict(), model_save_path)
print(f"模型已保存到 {model_save_path}")
# 绘制训练曲线
# 绘制训练曲线
plt.figure(figsize=(12, 6))

# 绘制 score 曲线
plt.subplot(1, 2, 1)
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Score over Episodes')

# 绘制 clock 曲线
plt.subplot(1, 2, 2)
plt.plot(clock_list)
plt.xlabel('Episode')
plt.ylabel('Clock')
plt.title('Clock over Episodes')

plt.tight_layout()
plt.show()


