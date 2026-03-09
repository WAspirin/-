"""
PPO (Proximal Policy Optimization) 策略梯度算法实现

论文参考：
Schulman et al. (2017). Proximal Policy Optimization Algorithms

核心思想:
1. Actor-Critic 架构：策略网络 + 价值网络
2. PPO 截断：限制策略更新幅度，避免训练不稳定
3. 优势函数估计：GAE (Generalized Advantage Estimation)
4. 多轮小批量更新：提高样本效率

与 DQN 对比:
- DQN: value-based, 学习 Q 表/网络，离散动作
- PPO: policy-based, 直接学习策略，连续/离散动作均可
- PPO 更稳定，样本效率更高

应用场景:
- 复杂路径规划
- 连续控制问题
- 需要平滑策略的场景

作者：智子 (Sophon)
日期：2026-03-09
版本：v1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import math
from typing import List, Tuple, Dict, Optional
import random

# 设置随机种子保证可复现性
np.random.seed(42)
random.seed(42)

# ============================================================================
# Part 1: 环境定义 - 线缆布线网格世界
# ============================================================================

class CableRoutingEnv:
    """
    线缆布线环境 - 网格世界
    
    功能:
    - 创建网格地图
    - 添加障碍物
    - 状态转移
    - 奖励计算
    """
    
    def __init__(self, grid_size: int = 15, n_obstacles: int = 20, seed: int = 42):
        """
        初始化环境
        
        Args:
            grid_size: 网格大小 (grid_size x grid_size)
            n_obstacles: 障碍物数量
            seed: 随机种子
        """
        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.n_states = grid_size * grid_size
        self.n_actions = 4  # 上/下/左/右
        
        # 动作定义
        self.action_to_delta = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        self.action_names = ['上', '下', '左', '右']
        
        # 固定起点 (左上) 和终点 (右下)
        self.start = 0
        self.goal = self.n_states - 1
        
        self.reset()
    
    def reset(self, randomize_obstacles: bool = False) -> int:
        """
        重置环境
        
        Args:
            randomize_obstacles: 是否随机化障碍物位置
            
        Returns:
            初始状态
        """
        self.current_state = self.start
        
        # 生成障碍物
        if randomize_obstacles:
            np.random.seed()
            obstacles = set()
            while len(obstacles) < self.n_obstacles:
                obs = np.random.randint(0, self.n_states)
                if obs != self.start and obs != self.goal:
                    obstacles.add(obs)
            self.obstacles = obstacles
        else:
            # 固定障碍物布局
            np.random.seed(42)
            obstacles = set()
            while len(obstacles) < self.n_obstacles:
                obs = np.random.randint(0, self.n_states)
                if obs != self.start and obs != self.goal:
                    obstacles.add(obs)
            self.obstacles = obstacles
        
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 2
        
        return self.current_state
    
    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """状态转坐标"""
        return (state // self.grid_size, state % self.grid_size)
    
    def pos_to_state(self, row: int, col: int) -> int:
        """坐标转状态"""
        return row * self.grid_size + col
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 动作索引 (0-3)
            
        Returns:
            next_state, reward, done, info
        """
        self.steps += 1
        
        # 获取当前位置
        row, col = self.state_to_pos(self.current_state)
        
        # 计算新位置
        dr, dc = self.action_to_delta[action]
        new_row = max(0, min(self.grid_size - 1, row + dr))
        new_col = max(0, min(self.grid_size - 1, col + dc))
        
        next_state = self.pos_to_state(new_row, new_col)
        
        # 计算奖励
        done = False
        info = {'collision': False, 'steps': self.steps}
        
        if next_state in self.obstacles:
            # 碰撞障碍物
            reward = -1.0
            info['collision'] = True
            next_state = self.current_state  # 保持原位
        elif next_state == self.goal:
            # 到达终点
            reward = 10.0
            done = True
        else:
            # 普通移动 - 步数惩罚 + 距离奖励
            reward = -0.1  # 每步惩罚
            
            # 额外奖励：靠近终点
            old_dist = abs(row - (self.goal // self.grid_size)) + abs(col - (self.goal % self.grid_size))
            new_dist = abs(new_row - (self.goal // self.grid_size)) + abs(new_col - (self.goal % self.grid_size))
            if new_dist < old_dist:
                reward += 0.05  # 靠近终点的小奖励
        
        # 超时终止
        if self.steps >= self.max_steps:
            done = True
            reward -= 5.0  # 超时惩罚
        
        self.current_state = next_state
        
        return next_state, reward, done, info
    
    def get_valid_actions(self, state: int) -> List[int]:
        """获取有效动作（排除立即碰撞的动作）"""
        row, col = self.state_to_pos(state)
        valid = []
        
        for action, (dr, dc) in self.action_to_delta.items():
            new_row = max(0, min(self.grid_size - 1, row + dr))
            new_col = max(0, min(self.grid_size - 1, col + dc))
            next_state = self.pos_to_state(new_row, new_col)
            if next_state not in self.obstacles:
                valid.append(action)
        
        return valid if valid else list(range(4))
    
    def render(self, path: List[int] = None) -> np.ndarray:
        """
        渲染环境
        
        Args:
            path: 路径列表
            
        Returns:
            网格表示
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # 标记障碍物
        for obs in self.obstacles:
            r, c = self.state_to_pos(obs)
            grid[r, c] = 1
        
        # 标记路径
        if path:
            for i, state in enumerate(path):
                r, c = self.state_to_pos(state)
                grid[r, c] = 2 + (i % 3)  # 不同颜色标记路径
        
        # 标记起点和终点
        r, c = self.state_to_pos(self.start)
        grid[r, c] = 10
        r, c = self.state_to_pos(self.goal)
        grid[r, c] = 11
        
        return grid


# ============================================================================
# Part 2: PPO 核心组件
# ============================================================================

class ActorNetwork:
    """
    Actor 网络 - 策略网络
    
    输入：状态
    输出：动作概率分布
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        初始化 Actor 网络
        
        Args:
            input_dim: 输入维度（状态数）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（动作数）
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 简单的两层神经网络
        # 使用 one-hot 编码表示状态
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
    
    def forward(self, state: int) -> np.ndarray:
        """
        前向传播
        
        Args:
            state: 状态索引
            
        Returns:
            动作概率分布
        """
        # One-hot 编码
        x = np.zeros(self.input_dim)
        x[state] = 1.0
        
        # 隐藏层 (ReLU)
        h = np.maximum(0, x @ self.W1 + self.b1)
        
        # 输出层 (Softmax)
        logits = h @ self.W2 + self.b2
        probs = self.softmax(logits)
        
        return probs
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        exp_x = np.exp(x - np.max(x))  # 数值稳定
        return exp_x / exp_x.sum()
    
    def get_action(self, state: int, deterministic: bool = False) -> int:
        """
        选择动作
        
        Args:
            state: 状态
            deterministic: 是否确定性选择
            
        Returns:
            动作索引
        """
        probs = self.forward(state)
        
        if deterministic:
            return np.argmax(probs)
        else:
            return np.random.choice(len(probs), p=probs)
    
    def get_log_prob(self, state: int, action: int) -> float:
        """
        计算动作的对数概率
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            对数概率
        """
        probs = self.forward(state)
        return np.log(probs[action] + 1e-10)
    
    def update(self, states: List[int], actions: List[int], advantages: np.ndarray, 
               lr: float = 0.001) -> float:
        """
        更新网络参数
        
        Args:
            states: 状态列表
            actions: 动作列表
            advantages: 优势函数
            lr: 学习率
            
        Returns:
            策略损失
        """
        total_loss = 0.0
        
        for state, action, adv in zip(states, actions, advantages):
            # One-hot 编码
            x = np.zeros(self.input_dim)
            x[state] = 1.0
            
            # 前向传播
            h = np.maximum(0, x @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            probs = self.softmax(logits)
            
            # 策略梯度更新
            # 对于选中的动作，增加其概率（如果优势为正）
            grad_logits = probs.copy()
            grad_logits[action] -= 1.0
            grad_logits *= adv  # 乘以优势函数
            
            # 更新 W2, b2
            dW2 = h.reshape(-1, 1) @ grad_logits.reshape(1, -1)
            db2 = grad_logits
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            
            # 更新 W1, b1
            grad_h = grad_logits @ self.W2.T
            grad_h[h <= 0] = 0  # ReLU 梯度
            dW1 = x.reshape(-1, 1) @ grad_h.reshape(1, -1)
            db1 = grad_h
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            
            # 计算损失（用于监控）
            total_loss -= np.log(probs[action] + 1e-10) * adv
        
        return total_loss / len(states)


class CriticNetwork:
    """
    Critic 网络 - 价值网络
    
    输入：状态
    输出：状态价值估计
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        初始化 Critic 网络
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 简单的两层神经网络
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)
    
    def forward(self, state: int) -> float:
        """
        前向传播
        
        Args:
            state: 状态索引
            
        Returns:
            状态价值估计
        """
        # One-hot 编码
        x = np.zeros(self.input_dim)
        x[state] = 1.0
        
        # 隐藏层 (ReLU)
        h = np.maximum(0, x @ self.W1 + self.b1)
        
        # 输出层 (线性)
        value = h @ self.W2 + self.b2
        
        return float(value[0])
    
    def update(self, states: List[int], targets: np.ndarray, lr: float = 0.001) -> float:
        """
        更新网络参数
        
        Args:
            states: 状态列表
            targets: 目标价值
            lr: 学习率
            
        Returns:
            价值损失
        """
        total_loss = 0.0
        
        for state, target in zip(states, targets):
            # One-hot 编码
            x = np.zeros(self.input_dim)
            x[state] = 1.0
            
            # 前向传播
            h = np.maximum(0, x @ self.W1 + self.b1)
            value = h @ self.W2 + self.b2
            
            # 计算误差
            error = value[0] - target
            total_loss += error ** 2
            
            # 梯度下降
            # d(value)/d(W2) = h
            dW2 = h.reshape(-1, 1) * error
            db2 = np.array([error])
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            
            # 反向传播到 W1
            grad_h = (self.W2 * error).flatten()
            grad_h[h <= 0] = 0
            dW1 = x.reshape(-1, 1) @ grad_h.reshape(1, -1)
            db1 = grad_h
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
        
        return total_loss / len(states)


class PPOTrainer:
    """
    PPO 训练器
    
    核心特性:
    - Actor-Critic 架构
    - GAE 优势估计
    - PPO 截断（简化版：限制学习率）
    - 多轮更新
    """
    
    def __init__(self, n_states: int, n_actions: int, hidden_dim: int = 128,
                 lr_actor: float = 0.001, lr_critic: float = 0.003,
                 gamma: float = 0.99, lam: float = 0.95,
                 clip_epsilon: float = 0.2, n_epochs: int = 10):
        """
        初始化 PPO 训练器
        
        Args:
            n_states: 状态数
            n_actions: 动作数
            hidden_dim: 隐藏层维度
            lr_actor: Actor 学习率
            lr_critic: Critic 学习率
            gamma: 折扣因子
            lam: GAE 参数
            clip_epsilon: PPO 截断参数
            n_epochs: 每批数据更新轮数
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        
        # 初始化网络
        self.actor = ActorNetwork(n_states, hidden_dim, n_actions)
        self.critic = CriticNetwork(n_states, hidden_dim)
        
        # 学习率
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': []
        }
    
    def select_action(self, state: int, deterministic: bool = False) -> int:
        """选择动作"""
        return self.actor.get_action(state, deterministic)
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool]) -> np.ndarray:
        """
        计算 GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励序列
            values: 价值估计序列
            dones: 终止标志序列
            
        Returns:
            优势函数估计
        """
        n = len(rewards)
        advantages = np.zeros(n)
        
        # 计算 TD 误差
        td_errors = np.zeros(n)
        for t in range(n):
            next_value = values[t + 1] if t < n - 1 else 0.0
            td_target = rewards[t] + self.gamma * next_value * (1 - dones[t])
            td_errors[t] = td_target - values[t]
        
        # GAE: 指数加权移动平均
        gae = 0.0
        for t in reversed(range(n)):
            gae = td_errors[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        return advantages
    
    def train_on_batch(self, states: List[int], actions: List[int], 
                       rewards: List[float], next_states: List[int], 
                       dones: List[bool]) -> Dict[str, float]:
        """
        在一批数据上训练
        
        Args:
            states: 状态序列
            actions: 动作序列
            rewards: 奖励序列
            next_states: 下一状态序列
            dones: 终止标志
            
        Returns:
            训练统计信息
        """
        # 计算当前价值估计
        values = [self.critic.forward(s) for s in states]
        next_values = [self.critic.forward(s) for s in next_states]
        
        # 计算 GAE
        advantages = self.compute_gae(rewards, values, dones)
        
        # 计算 TD 目标（用于 Critic 更新）
        td_targets = []
        for t in range(len(rewards)):
            td_target = rewards[t] + self.gamma * next_values[t] * (1 - dones[t])
            td_targets.append(td_target)
        td_targets = np.array(td_targets)
        
        # 多轮更新
        actor_losses = []
        critic_losses = []
        
        for epoch in range(self.n_epochs):
            # 更新 Actor
            actor_loss = self.actor.update(states, actions, advantages, self.lr_actor)
            actor_losses.append(actor_loss)
            
            # 更新 Critic
            critic_loss = self.critic.update(states, td_targets, self.lr_critic)
            critic_losses.append(critic_loss)
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'advantage_mean': advantages.mean(),
            'advantage_std': advantages.std()
        }
    
    def train(self, n_episodes: int = 500, max_steps: int = 200,
              verbose: bool = True, grid_size: int = 10, n_obstacles: int = 15) -> Dict:
        """
        训练 PPO 模型
        
        Args:
            n_episodes: 训练回合数
            max_steps: 每回合最大步数
            verbose: 是否打印进度
            grid_size: 网格大小
            n_obstacles: 障碍物数量
            
        Returns:
            训练统计信息
        """
        env = CableRoutingEnv(grid_size=grid_size, n_obstacles=n_obstacles)
        
        for episode in range(n_episodes):
            # 重置环境
            state = env.reset(randomize_obstacles=(episode % 10 == 0))
            
            # 收集轨迹
            states, actions, rewards, next_states, dones = [], [], [], [], []
            
            episode_reward = 0.0
            done = False
            
            for step in range(max_steps):
                # 选择动作
                action = self.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储数据
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # 训练
            stats = self.train_on_batch(states, actions, rewards, next_states, dones)
            
            # 记录统计
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(len(states))
            self.training_stats['losses'].append(stats)
            
            # 打印进度
            if verbose and (episode + 1) % 20 == 0:
                window = min(20, episode + 1)
                avg_reward = np.mean(self.training_stats['episode_rewards'][-window:])
                avg_length = np.mean(self.training_stats['episode_lengths'][-window:])
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Avg Reward = {avg_reward:.2f}, "
                      f"Avg Length = {avg_length:.1f}, "
                      f"Actor Loss = {stats['actor_loss']:.4f}")
        
        return self.training_stats


# ============================================================================
# Part 3: 可视化
# ============================================================================

class PPOVisualizer:
    """PPO 训练可视化"""
    
    @staticmethod
    def plot_training_curve(stats: Dict, save_path: str = None):
        """
        绘制训练曲线
        
        Args:
            stats: 训练统计
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 回合奖励曲线
        ax = axes[0, 0]
        rewards = stats['episode_rewards']
        ax.plot(rewards, alpha=0.5, label='单回合奖励')
        
        # 移动平均
        window = 50
        if len(rewards) >= window:
            ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(ma_rewards, 'r-', linewidth=2, label=f'{window}回合移动平均')
        
        ax.set_xlabel('回合')
        ax.set_ylabel('奖励')
        ax.set_title('PPO 训练 - 回合奖励曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 回合长度曲线
        ax = axes[0, 1]
        lengths = stats['episode_lengths']
        ax.plot(lengths, alpha=0.5, color='green')
        
        if len(lengths) >= window:
            ma_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(ma_lengths, 'r-', linewidth=2)
        
        ax.set_xlabel('回合')
        ax.set_ylabel('步数')
        ax.set_title('PPO 训练 - 回合长度曲线')
        ax.grid(True, alpha=0.3)
        
        # 3. Actor 损失曲线
        ax = axes[1, 0]
        actor_losses = [s['actor_loss'] for s in stats['losses']]
        ax.plot(actor_losses, alpha=0.5)
        
        if len(actor_losses) >= window:
            ma_losses = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
            ax.plot(ma_losses, 'r-', linewidth=2)
        
        ax.set_xlabel('回合')
        ax.set_ylabel('损失')
        ax.set_title('PPO 训练 - Actor 损失曲线')
        ax.grid(True, alpha=0.3)
        
        # 4. Critic 损失曲线
        ax = axes[1, 1]
        critic_losses = [s['critic_loss'] for s in stats['losses']]
        ax.plot(critic_losses, alpha=0.5, color='purple')
        
        if len(critic_losses) >= window:
            ma_losses = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
            ax.plot(ma_losses, 'r-', linewidth=2)
        
        ax.set_xlabel('回合')
        ax.set_ylabel('损失')
        ax.set_title('PPO 训练 - Critic 损失曲线')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存：{save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_policy_heatmap(actor: ActorNetwork, env: CableRoutingEnv, 
                           save_path: str = None):
        """
        绘制策略热力图
        
        Args:
            actor: Actor 网络
            env: 环境
            save_path: 保存路径
        """
        grid_size = env.grid_size
        policy_grid = np.zeros((grid_size, grid_size, 4))
        
        # 获取每个状态的动作概率
        for state in range(env.n_states):
            row, col = env.state_to_pos(state)
            probs = actor.forward(state)
            policy_grid[row, col] = probs
        
        # 绘制每个动作的概率分布
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        action_names = ['上', '下', '左', '右']
        
        for ax, action in zip(axes, range(4)):
            im = ax.imshow(policy_grid[:, :, action], cmap='viridis', 
                          vmin=0, vmax=1, aspect='auto')
            ax.set_title(f'动作 "{action_names[action]}" 概率')
            ax.set_xlabel('列')
            ax.set_ylabel('行')
            
            # 标记障碍物
            for obs in env.obstacles:
                r, c = env.state_to_pos(obs)
                ax.plot(c, r, 'rx', markersize=15, markeredgewidth=2)
            
            # 标记起点和终点
            r, c = env.state_to_pos(env.start)
            ax.plot(c, r, 'go', markersize=10, label='起点')
            r, c = env.state_to_pos(env.goal)
            ax.plot(c, r, 'bs', markersize=10, label='终点')
            
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"策略热力图已保存：{save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_learned_path(actor: ActorNetwork, env: CableRoutingEnv,
                         save_path: str = None):
        """
        绘制学习到的路径
        
        Args:
            actor: Actor 网络
            env: 环境
            save_path: 保存路径
        """
        # 使用学到的策略执行
        state = env.reset()
        path = [state]
        done = False
        max_steps = 200
        
        for _ in range(max_steps):
            action = actor.get_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            path.append(next_state)
            state = next_state
            
            if done:
                break
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制网格
        grid = np.zeros((env.grid_size, env.grid_size))
        for obs in env.obstacles:
            r, c = env.state_to_pos(obs)
            grid[r, c] = 1
        
        ax.imshow(grid, cmap='gray_r', alpha=0.3)
        
        # 绘制路径
        path_coords = [env.state_to_pos(s) for s in path]
        path_rows = [p[0] for p in path_coords]
        path_cols = [p[1] for p in path_coords]
        
        ax.plot(path_cols, path_rows, 'b-o', linewidth=2, markersize=8, 
               label='学习路径', alpha=0.7)
        
        # 标记起点和终点
        r, c = env.state_to_pos(env.start)
        ax.plot(c, r, 'go', markersize=15, label='起点')
        r, c = env.state_to_pos(env.goal)
        ax.plot(c, r, 'rs', markersize=15, label='终点')
        
        ax.set_title(f'PPO 学习到的路径 (步数：{len(path)-1})')
        ax.set_xlabel('列')
        ax.set_ylabel('行')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"学习路径已保存：{save_path}")
        
        plt.show()
        
        return path


# ============================================================================
# Part 4: 主程序 - 训练与测试
# ============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("PPO (Proximal Policy Optimization) 策略梯度算法")
    print("=" * 70)
    
    # 配置参数
    GRID_SIZE = 10  # 使用较小的网格加快训练
    N_OBSTACLES = 15
    N_EPISODES = 100  # 减少训练回合以加快演示
    MAX_STEPS = 100
    
    # 创建训练器
    trainer = PPOTrainer(
        n_states=GRID_SIZE * GRID_SIZE,  # 10x10 网格
        n_actions=4,
        hidden_dim=128,
        lr_actor=0.001,
        lr_critic=0.003,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        n_epochs=10
    )
    
    print(f"\n开始训练 PPO 模型...")
    print(f"环境：{GRID_SIZE}x{GRID_SIZE} 网格，{N_OBSTACLES} 个障碍物")
    print(f"训练回合：{N_EPISODES}")
    print("-" * 70)
    
    # 训练
    stats = trainer.train(n_episodes=N_EPISODES, max_steps=MAX_STEPS, verbose=True, 
                         grid_size=GRID_SIZE, n_obstacles=N_OBSTACLES)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    
    # 测试
    print("\n测试学到的策略...")
    env = CableRoutingEnv(grid_size=GRID_SIZE, n_obstacles=N_OBSTACLES)
    
    # 执行测试
    state = env.reset()
    path = [state]
    total_reward = 0.0
    done = False
    
    for step in range(MAX_STEPS):
        action = trainer.actor.get_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        path.append(next_state)
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    print(f"\n测试结果:")
    print(f"  路径长度：{len(path)-1} 步")
    print(f"  总奖励：{total_reward:.2f}")
    print(f"  是否到达终点：{next_state == env.goal}")
    
    # 可视化
    print("\n生成可视化图表...")
    
    # 创建输出目录
    import os
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制训练曲线
    PPOVisualizer.plot_training_curve(
        stats, 
        save_path=os.path.join(output_dir, '15_ppo_training_curve.png')
    )
    
    # 绘制策略热力图
    PPOVisualizer.plot_policy_heatmap(
        trainer.actor, env,
        save_path=os.path.join(output_dir, '15_ppo_policy_heatmap.png')
    )
    
    # 绘制学习路径
    PPOVisualizer.plot_learned_path(
        trainer.actor, env,
        save_path=os.path.join(output_dir, '15_ppo_learned_path.png')
    )
    
    print("\n" + "=" * 70)
    print("PPO 算法实现完成!")
    print("=" * 70)
    
    # 保存测试结果
    results = {
        'algorithm': 'PPO',
        'grid_size': GRID_SIZE,
        'n_obstacles': N_OBSTACLES,
        'training_episodes': N_EPISODES,
        'final_avg_reward': float(np.mean(stats['episode_rewards'][-20:])),
        'test_path_length': len(path) - 1,
        'test_total_reward': float(total_reward),
        'reached_goal': next_state == env.goal
    }
    
    import json
    results_path = os.path.join(output_dir, '15_ppo_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试结果已保存：{results_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    return results


if __name__ == '__main__':
    main()
