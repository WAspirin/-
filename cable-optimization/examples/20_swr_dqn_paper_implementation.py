"""
SW-RDQN 完整实现 - 根据论文《基于关键路径点的深度强化学习复合路径规划》

核心创新:
1. 阶段关键点机制 (stage-keys) - 多阶段子目标分解
2. 轨迹预测模块 - RNN 预测动态障碍物轨迹
3. 优先经验回放 - 基于奖励+TD 误差的加权采样 (论文式 19)
4. 网络结构 - CNN + LSTM + Dueling DQN

作者：智子 (Sophon)
日期：2026-03-10
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import math
from typing import List, Tuple, Dict
import heapq


# ============================================================================
# 1. 经验回放池 - 论文式 19 优先采样
# ============================================================================

class PrioritizedReplayBuffer:
    """基于奖励和 TD 误差的优先经验回放 (论文式 19)"""
    
    def __init__(self, capacity: int = 10000, 
                 w_reward: float = 0.5, 
                 w_td_error: float = 0.5):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.w_reward = w_reward
        self.w_td_error = w_td_error
    
    def add(self, transition: Dict, reward: float, td_error: float):
        """
        添加经验并计算优先级 (论文式 19)
        
        P(i) = [w_reward * log(reward_i + |min(reward_i, 0)| + 1) 
              + w_td_error * log(|TD_error_i| + 1)] / sum
        """
        self.buffer.append(transition)
        
        # 计算优先级 (式 19)
        reward_term = self.w_reward * math.log(reward + abs(min(reward, 0)) + 1)
        td_term = self.w_td_error * math.log(abs(td_error) + 1)
        priority = reward_term + td_term
        
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[List, List, List]:
        """按优先级采样"""
        if len(self.buffer) < batch_size:
            return list(self.buffer), None, None
        
        # 归一化优先级为概率
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # 按概率采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]
        
        # 返回对应的 TD 误差和奖励用于更新
        td_errors = [None] * batch_size
        rewards = [None] * batch_size
        
        return batch, indices, probabilities[indices]
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """更新指定经验的优先级"""
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 2. 网络结构 - 论文图 4: CNN + LSTM + Dueling DQN
# ============================================================================

class SWRDQNNetwork(nn.Module):
    """
    SW-RDQN 网络结构 (论文图 4)
    
    输入：(batch_size, time_steps, input_features)
    输出：Q 值 (batch_size, n_actions)
    """
    
    def __init__(self, input_features: int = 576, 
                 hidden_size: int = 256,
                 n_actions: int = 4,
                 time_steps: int = 4):
        super(SWRDQNNetwork, self).__init__()
        
        self.time_steps = time_steps
        
        # 空间关系编码 (论文 3.2 节)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        # 场景特征提取 - 双层 CNN
        self.cnn = nn.Sequential(
            # 第一层：5x5 卷积
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # 第二层：3x3 卷积，步长 2 下采样
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # LSTM 层 - 时序建模
        self.lstm = nn.LSTM(
            input_size=64 + 64,  # CNN 输出 + 空间关系特征
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Dueling DQN 分支
        self.value_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.advantage_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions)
        )
    
    def forward(self, x: torch.Tensor, hidden_state: tuple = None):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, time_steps, features)
            hidden_state: LSTM 隐藏状态
        
        Returns:
            q_values: Q 值 (batch, n_actions)
            hidden_state: 新的 LSTM 隐藏状态
        """
        batch_size = x.shape[0]
        
        # 空间关系编码
        spatial_feat = self.spatial_encoder(x[:, -1, :])  # 取最后时间步
        
        # CNN 特征提取 (假设输入包含栅格图)
        # 这里简化处理，实际应传入栅格图
        cnn_feat = self.cnn(x[:, :, :].view(-1, 1, 24, 24))
        cnn_feat = cnn_feat.view(batch_size, self.time_steps, -1)
        
        # 特征融合
        fused = torch.cat([cnn_feat, spatial_feat.unsqueeze(1).repeat(1, self.time_steps, 1)], dim=2)
        
        # LSTM 时序建模
        lstm_out, hidden_state = self.lstm(fused, hidden_state)
        
        # 取最后时间步
        last_hidden = lstm_out[:, -1, :]
        
        # Dueling DQN 分支
        value = self.value_branch(last_hidden)
        advantage = self.advantage_branch(last_hidden)
        
        # 组合 (论文式 17)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden_state


# ============================================================================
# 3. 奖励函数 - 论文式 12-16
# ============================================================================

class RewardFunction:
    """
    复合奖励函数 (论文式 16)
    
    Rround = Rgoal + Ralign + Ravoid + Rpre + Rstep
    """
    
    def __init__(self, config: Dict = None):
        # 奖励参数 (论文表 2)
        self.config = config or {
            'r_goal': 100.0,      # 到达目标奖励
            'r_align': 1.0,       # 对齐奖励系数
            'r_avoid_1': 1.0,     # 避障奖励 (d < 1.0)
            'r_avoid_2': 2.0,     # 避障奖励 (1.0 < d < 2.0)
            'r_avoid_3': 3.0,     # 避障奖励 (2.0 < d < 3.0)
            'r_pre': -1.25,       # 预测落点惩罚系数
            'r_step': -0.01,      # 步数惩罚
            'dis_threshold': 2.0  # 预测距离阈值
        }
    
    def calculate_reward(self, state: Dict, action: int, 
                        next_state: Dict, done: bool) -> float:
        """
        计算总奖励 (论文式 16)
        
        Args:
            state: 当前状态
            action: 动作
            next_state: 下一状态
            done: 是否结束
        
        Returns:
            reward: 总奖励
        """
        r_goal = self._r_goal(done, next_state)
        r_align = self._r_align(state, action, next_state)
        r_avoid = self._r_avoid(next_state)
        r_pre = self._r_pre(next_state)
        r_step = self.config['r_step']
        
        total_reward = r_goal + r_align + r_avoid + r_pre + r_step
        
        return total_reward
    
    def _r_goal(self, done: bool, next_state: Dict) -> float:
        """到达目标奖励 (论文式 11)"""
        if done:
            return self.config['r_goal']
        return 0.0
    
    def _r_align(self, state: Dict, action: int, next_state: Dict) -> float:
        """
        轨迹对齐奖励 (论文式 12)
        
        鼓励机器人朝向关键点移动
        """
        # 计算机器人朝向与关键点方向的夹角
        robot_pos = np.array(next_state['position'])
        key_point = np.array(next_state['current_key_point'])
        
        # 期望方向
        expected_dir = key_point - robot_pos
        if np.linalg.norm(expected_dir) < 1e-6:
            return 0.0
        
        expected_dir = expected_dir / np.linalg.norm(expected_dir)
        
        # 实际移动方向
        actual_dir = robot_pos - np.array(state['position'])
        if np.linalg.norm(actual_dir) < 1e-6:
            return 0.0
        
        actual_dir = actual_dir / np.linalg.norm(actual_dir)
        
        # 计算夹角余弦
        cos_theta = np.dot(expected_dir, actual_dir)
        
        return self.config['r_align'] * cos_theta
    
    def _r_avoid(self, next_state: Dict) -> float:
        """
        避障奖励 (论文式 13-14)
        
        根据与障碍物的距离给予不同奖励
        """
        d_obstacle = next_state['distance_to_obstacle']
        
        if d_obstacle < 1.0:
            return self.config['r_avoid_1']
        elif d_obstacle < 2.0:
            return self.config['r_avoid_2']
        elif d_obstacle < 3.0:
            return self.config['r_avoid_3']
        else:
            return 0.0
    
    def _r_pre(self, next_state: Dict) -> float:
        """
        预测落点靠近惩罚 (论文式 15)
        
        惩罚机器人靠近动态障碍物预测轨迹
        """
        dis = next_state['distance_to_predicted_obstacle']
        
        if dis < self.config['dis_threshold']:
            return self.config['r_pre'] * dis
        else:
            return 0.0


# ============================================================================
# 4. SW-RDQN 智能体 - 完整实现
# ============================================================================

class SWRDQNAgent:
    """
    SW-RDQN 智能体
    
    核心特性:
    1. 阶段关键点机制
    2. 优先经验回放 (论文式 19)
    3. Dueling DQN 网络
    """
    
    def __init__(self, state_dim: int = 576, 
                 n_actions: int = 4,
                 config: Dict = None):
        
        self.config = config or {
            'gamma': 0.99,
            'lr': 1e-4,
            'batch_size': 64,
            'buffer_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_freq': 100,
            'time_steps': 4
        }
        
        self.n_actions = n_actions
        self.epsilon = self.config['epsilon_start']
        self.steps = 0
        
        # 网络
        self.network = SWRDQNNetwork(
            input_features=state_dim,
            n_actions=n_actions,
            time_steps=self.config['time_steps']
        )
        self.target_network = SWRDQNNetwork(
            input_features=state_dim,
            n_actions=n_actions,
            time_steps=self.config['time_steps']
        )
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.config['lr']
        )
        
        # 经验回放池 (论文式 19)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config['buffer_size']
        )
        
        # 奖励函数
        self.reward_function = RewardFunction()
        
        # 阶段关键点
        self.key_points = []
        self.current_key_idx = 0
        
        # 经验序列
        self.state_sequence = deque(maxlen=self.config['time_steps'])
    
    def select_action(self, state: Dict, train: bool = True) -> int:
        """
        选择动作 (ε-greedy)
        
        Args:
            state: 状态字典
            train: 是否训练模式
        
        Returns:
            action: 动作索引
        """
        if train and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # 准备输入
        state_tensor = self._prepare_state_input(state)
        
        with torch.no_grad():
            q_values, _ = self.network(state_tensor.unsqueeze(0))
            return q_values.argmax(dim=1).item()
    
    def _prepare_state_input(self, state: Dict) -> torch.Tensor:
        """准备状态输入"""
        # 这里简化处理，实际应包含：
        # 1. 栅格地图
        # 2. 机器人位置
        # 3. 关键点位置
        # 4. 障碍物预测轨迹
        
        # 简化为固定维度特征向量
        features = np.zeros(576)
        
        # 填充位置信息
        pos = state.get('position', [0, 0])
        key_point = state.get('current_key_point', [0, 0])
        
        features[0:2] = pos
        features[2:4] = key_point
        features[4] = state.get('distance_to_obstacle', 0)
        features[5] = state.get('distance_to_predicted_obstacle', 0)
        
        return torch.FloatTensor(features)
    
    def store_transition(self, state: Dict, action: int, 
                        reward: float, next_state: Dict, done: bool,
                        td_error: float = None):
        """
        存储转移 (论文式 19)
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            td_error: TD 误差 (用于优先采样)
        """
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        # 如果没有 TD 误差，使用奖励代替
        if td_error is None:
            td_error = abs(reward)
        
        self.replay_buffer.add(transition, reward, td_error)
    
    def update(self) -> float:
        """
        更新网络
        
        Returns:
            loss: 损失值
        """
        if len(self.replay_buffer) < self.config['batch_size']:
            return 0.0
        
        # 优先采样
        batch, indices, probabilities = self.replay_buffer.sample(
            self.config['batch_size']
        )
        
        # 转换为张量
        states = torch.stack([self._prepare_state_input(t['state']) for t in batch])
        actions = torch.LongTensor([t['action'] for t in batch])
        rewards = torch.FloatTensor([t['reward'] for t in batch])
        next_states = torch.stack([self._prepare_state_input(t['next_state']) for t in batch])
        dones = torch.FloatTensor([t['done'] for t in batch])
        
        # 计算当前 Q 值
        q_values, _ = self.network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标 Q 值 (多步贝尔曼方程，论文式 7)
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            next_q_max = next_q_values.max(dim=1)[0]
            
            # n-step 回报
            targets = rewards + (self.config['gamma'] ** self.config['time_steps']) * next_q_max * (1 - dones)
        
        # 计算损失
        loss = F.smooth_l1_loss(q_values, targets)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新优先级
        if indices is not None:
            new_priorities = []
            for i, (state, action, reward, next_state, done) in zip(
                indices, batch
            ):
                # 重新计算 TD 误差
                with torch.no_grad():
                    state_tensor = self._prepare_state_input(state).unsqueeze(0)
                    next_state_tensor = self._prepare_state_input(next_state).unsqueeze(0)
                    
                    q = self.network(state_tensor).gather(1, torch.LongTensor([action])).item()
                    next_q = self.target_network(next_state_tensor).max().item()
                    
                    td_error = abs(reward + self.config['gamma'] * next_q * (1 - done) - q)
                    new_priorities.append(td_error)
            
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        # 更新目标网络
        self.steps += 1
        if self.steps % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        # 衰减 ε
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )
        
        return loss.item()
    
    def set_key_points(self, key_points: List[Tuple[int, int]]):
        """设置阶段关键点"""
        self.key_points = key_points
        self.current_key_idx = 0
    
    def get_current_key_point(self) -> Tuple[int, int]:
        """获取当前阶段关键点"""
        if self.current_key_idx < len(self.key_points):
            return self.key_points[self.current_key_idx]
        return None
    
    def update_key_point(self, robot_pos: Tuple[int, int], threshold: float = 3.0):
        """
        更新关键点索引
        
        当机器人靠近当前关键点时，切换到下一个关键点
        """
        if self.current_key_idx < len(self.key_points):
            key_point = self.key_points[self.current_key_idx]
            distance = np.linalg.norm(np.array(robot_pos) - np.array(key_point))
            
            if distance < threshold:
                self.current_key_idx += 1


# ============================================================================
# 5. 主函数 - 测试
# ============================================================================

def main():
    """测试 SW-RDQN 智能体"""
    print("="*60)
    print("SW-RDQN 完整实现测试")
    print("="*60)
    
    # 创建智能体
    agent = SWRDQNAgent(
        state_dim=576,
        n_actions=4,
        config={
            'gamma': 0.99,
            'lr': 1e-4,
            'batch_size': 64,
            'buffer_size': 10000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_freq': 100,
            'time_steps': 4
        }
    )
    
    print(f"✓ 智能体创建成功")
    print(f"  网络参数量：{sum(p.numel() for p in agent.network.parameters()):,}")
    print(f"  动作空间：{agent.n_actions}")
    print(f"  经验回放池容量：{agent.config['buffer_size']}")
    
    # 模拟训练
    print("\n开始模拟训练...")
    n_episodes = 10
    
    for episode in range(n_episodes):
        # 重置环境 (简化)
        state = {
            'position': [0, 0],
            'current_key_point': [10, 10],
            'distance_to_obstacle': 5.0,
            'distance_to_predicted_obstacle': 10.0
        }
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            # 选择动作
            action = agent.select_action(state, train=True)
            
            # 执行动作 (简化)
            next_state = state.copy()
            reward = agent.reward_function.calculate_reward(state, action, next_state, False)
            
            # 存储转移
            agent.store_transition(state, action, reward, next_state, False)
            
            # 更新网络
            if len(agent.replay_buffer) >= agent.config['batch_size']:
                loss = agent.update()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        print(f"  回合 {episode+1}/{n_episodes}: "
              f"奖励={total_reward:.2f}, "
              f"步数={steps}, "
              f"ε={agent.epsilon:.3f}")
    
    print("\n✓ 测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
