import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnhancedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        _1 = 256
        _2 = 128
        _3 = 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, _1),
            nn.LayerNorm(_1),
            nn.LeakyReLU(0.1),
            nn.Linear(_1, _2),
            nn.LayerNorm(_2),
            nn.LeakyReLU(0.1),
            nn.Linear(_2, _3),
            nn.LayerNorm(_3),
            nn.LeakyReLU(0.1),
            nn.Linear(_3, output_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')


    def forward(self, x):
        return self.net(x)


def train_dqn(env, policy_net, target_net, optimizer, memory, batch_size=256, gamma=0.95):
    # 随机采样
    indices = np.random.choice(memory['size'], batch_size, replace=False)
    batch_states = memory['states'][indices]
    batch_actions = memory['actions'][indices]
    batch_next_states = memory['next_states'][indices]
    batch_rewards = memory['rewards'][indices]
    batch_dones = memory['dones'][indices]

    # 转换为张量
    states = torch.tensor(batch_states, dtype=torch.float32).to(device)
    actions = torch.tensor(batch_actions, dtype=torch.int64).to(device).unsqueeze(1)
    next_states = torch.tensor(batch_next_states, dtype=torch.float32).to(device)
    rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device).unsqueeze(1)
    dones = torch.tensor(batch_dones, dtype=torch.float32).to(device).unsqueeze(1)

    # 计算当前Q值
    current_q = policy_net(states).gather(1, actions)

    # 计算目标Q值（关闭梯度计算）
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + gamma * next_q * (1 - dones)

    # 损失计算与优化
    loss = nn.MSELoss()(current_q, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
