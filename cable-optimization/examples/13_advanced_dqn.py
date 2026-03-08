"""
进阶 DQN 变体实现

包含:
1. Double DQN - 解决 Q-learning 过估计问题
2. Dueling DQN - 分离状态价值和优势函数
3. 大规模测试对比

作者：智子 (Sophon)
日期：2026-03-08
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from typing import List, Tuple, Dict
import math

# ============================================================================
# Part 1: 环境定义 (10x10 网格布线)
# ============================================================================

class CableRoutingEnv:
    """电缆布线环境 - 网格世界"""
    
    def __init__(self, grid_size: int = 10, seed: int = 42):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4  # 上/下/左/右
        
        # 动作定义
        self.action_to_delta = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        # 固定起点和终点
        np.random.seed(seed)
        self.start = 0
        self.goal = self.n_states - 1
        
        self.reset()
    
    def reset(self) -> int:
        """重置环境"""
        self.current_state = self.start
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """执行动作"""
        # 计算新位置
        row, col = divmod(self.current_state, self.grid_size)
        dr, dc = self.action_to_delta[action]
        new_row = max(0, min(self.grid_size - 1, row + dr))
        new_col = max(0, min(self.grid_size - 1, col + dc))
        new_state = new_row * self.grid_size + new_col
        
        # 奖励设计
        if new_state == self.goal:
            reward = 10.0
            done = True
        else:
            # 距离奖励 + 步数惩罚
            old_dist = abs(row - (self.goal // self.grid_size)) + abs(col - (self.goal % self.grid_size))
            new_dist = abs(new_row - (self.goal // self.grid_size)) + abs(new_col - (self.goal % self.grid_size))
            reward = (old_dist - new_dist) - 0.1  # 鼓励靠近终点 + 步数惩罚
            done = False
        
        self.current_state = new_state
        return new_state, reward, done


# ============================================================================
# Part 2: Double DQN 实现
# ============================================================================

class DoubleDQN:
    """Double DQN - 解决过估计问题"""
    
    def __init__(self, 
                 n_states: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 target_update_freq: int = 10):
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        
        # Q 表 (状态 - 动作值)
        self.q_table = np.zeros((n_states, n_actions))
        self.target_q_table = np.zeros((n_states, n_actions))
        
        # 经验回放
        self.memory = deque(maxlen=2000)
        
        self.training_history = []
    
    def get_action(self, state: int, training: bool = True) -> int:
        """ε-greedy 策略选择动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, batch_size: int = 32):
        """使用 Double DQN 更新 Q 表"""
        if len(self.memory) < batch_size:
            return
        
        # 采样 batch
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                # Double DQN 核心：用 online network 选择动作，target network 评估价值
                best_action = np.argmax(self.q_table[next_state])
                target = reward + self.gamma * self.target_q_table[next_state, best_action]
            
            # Q-learning 更新
            self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        
        # 衰减 ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_q_table = self.q_table.copy()
    
    def train(self, env, n_episodes: int = 500, batch_size: int = 32) -> List[float]:
        """训练 Double DQN"""
        print(f"\n{'='*60}")
        print(f"Double DQN 训练")
        print(f"{'='*60}")
        print(f"状态空间：{self.n_states}")
        print(f"动作空间：{self.n_actions}")
        print(f"训练回合：{n_episodes}")
        print(f"{'='*60}\n")
        
        rewards_per_episode = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # 选择动作
                action = self.get_action(state)
                
                # 执行动作
                next_state, reward, done = env.step(action)
                
                # 存储经验
                self.store_transition(state, action, reward, next_state, done)
                
                # 更新 Q 表
                self.update(batch_size)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # 更新目标网络
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            
            rewards_per_episode.append(total_reward)
            
            # 打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"回合 {episode+1}/{n_episodes}: 平均奖励 = {avg_reward:.2f}, ε = {self.epsilon:.3f}")
        
        self.training_history = rewards_per_episode
        return rewards_per_episode
    
    def test(self, env, n_episodes: int = 10) -> Dict:
        """测试训练好的模型"""
        print(f"\n{'='*60}")
        print(f"Double DQN 测试")
        print(f"{'='*60}\n")
        
        rewards = []
        steps_list = []
        
        for _ in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # 贪婪策略（不探索）
                action = np.argmax(self.q_table[state])
                next_state, reward, done = env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            rewards.append(total_reward)
            steps_list.append(steps)
        
        results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(steps_list),
            'min_steps': np.min(steps_list),
            'max_steps': np.max(steps_list)
        }
        
        print(f"测试 {n_episodes} 次:")
        print(f"  平均奖励：{results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  平均步数：{results['avg_steps']:.1f}")
        print(f"  最少步数：{results['min_steps']}")
        print(f"  最多步数：{results['max_steps']}")
        print()
        
        return results


# ============================================================================
# Part 3: Dueling DQN 实现
# ============================================================================

class DuelingDQN:
    """Dueling DQN - 分离状态价值和优势函数"""
    
    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 target_update_freq: int = 10):
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        
        # Dueling DQN 核心：分离 V(s) 和 A(s,a)
        self.V_table = np.zeros(n_states)  # 状态价值
        self.A_table = np.zeros((n_states, n_actions))  # 优势函数
        self.target_V_table = np.zeros(n_states)
        self.target_A_table = np.zeros((n_states, n_actions))
        
        # 经验回放
        self.memory = deque(maxlen=2000)
        
        self.training_history = []
    
    def get_q_values(self) -> np.ndarray:
        """计算 Q 值：Q(s,a) = V(s) + A(s,a) - mean(A(s,:))"""
        Q = self.V_table.reshape(-1, 1) + self.A_table - np.mean(self.A_table, axis=1, keepdims=True)
        return Q
    
    def get_target_q_values(self) -> np.ndarray:
        """计算目标 Q 值"""
        Q = self.target_V_table.reshape(-1, 1) + self.target_A_table - np.mean(self.target_A_table, axis=1, keepdims=True)
        return Q
    
    def get_action(self, state: int, training: bool = True) -> int:
        """ε-greedy 策略"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            Q = self.get_q_values()
            return np.argmax(Q[state])
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, batch_size: int = 32):
        """使用 Dueling DQN 更新"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        Q = self.get_q_values()
        target_Q = self.get_target_q_values()
        
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(target_Q[next_state])
            
            # 分别更新 V 和 A
            td_error = target - Q[state, action]
            
            # 简化更新：同时更新 V 和 A
            self.V_table[state] += self.lr * td_error * 0.5
            self.A_table[state, action] += self.lr * td_error * 0.5
        
        # 衰减 ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_V_table = self.V_table.copy()
        self.target_A_table = self.A_table.copy()
    
    def train(self, env, n_episodes: int = 500, batch_size: int = 32) -> List[float]:
        """训练 Dueling DQN"""
        print(f"\n{'='*60}")
        print(f"Dueling DQN 训练")
        print(f"{'='*60}")
        print(f"状态空间：{self.n_states}")
        print(f"动作空间：{self.n_actions}")
        print(f"训练回合：{n_episodes}")
        print(f"{'='*60}\n")
        
        rewards_per_episode = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = self.get_action(state)
                next_state, reward, done = env.step(action)
                self.store_transition(state, action, reward, next_state, done)
                self.update(batch_size)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            
            rewards_per_episode.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"回合 {episode+1}/{n_episodes}: 平均奖励 = {avg_reward:.2f}, ε = {self.epsilon:.3f}")
        
        self.training_history = rewards_per_episode
        return rewards_per_episode
    
    def test(self, env, n_episodes: int = 10) -> Dict:
        """测试"""
        print(f"\n{'='*60}")
        print(f"Dueling DQN 测试")
        print(f"{'='*60}\n")
        
        rewards = []
        steps_list = []
        
        for _ in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = self.get_action(state, training=False)
                next_state, reward, done = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    break
            
            rewards.append(total_reward)
            steps_list.append(steps)
        
        results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(steps_list),
            'min_steps': np.min(steps_list),
            'max_steps': np.max(steps_list)
        }
        
        print(f"测试 {n_episodes} 次:")
        print(f"  平均奖励：{results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  平均步数：{results['avg_steps']:.1f}")
        print(f"  最少步数：{results['min_steps']}")
        print(f"  最多步数：{results['max_steps']}")
        print()
        
        return results


# ============================================================================
# Part 4: 对比实验
# ============================================================================

def compare_dqn_variants():
    """对比 DQN 变体"""
    print("\n" + "="*60)
    print("DQN 变体对比实验")
    print("="*60 + "\n")
    
    # 创建环境
    env = CableRoutingEnv(grid_size=10, seed=42)
    
    # 训练 Double DQN
    double_dqn = DoubleDQN(env.n_states, env.n_actions)
    double_dqn.train(env, n_episodes=500, batch_size=32)
    double_dqn_results = double_dqn.test(env, n_episodes=20)
    
    # 训练 Dueling DQN
    env.reset()
    dueling_dqn = DuelingDQN(env.n_states, env.n_actions)
    dueling_dqn.train(env, n_episodes=500, batch_size=32)
    dueling_dqn_results = dueling_dqn.test(env, n_episodes=20)
    
    # 对比结果
    print("\n" + "="*60)
    print("对比结果汇总")
    print("="*60)
    print(f"{'算法':<15} {'平均奖励':<12} {'标准差':<12} {'平均步数':<12}")
    print("-"*60)
    print(f"{'Double DQN':<15} {double_dqn_results['avg_reward']:<12.2f} {double_dqn_results['std_reward']:<12.2f} {double_dqn_results['avg_steps']:<12.1f}")
    print(f"{'Dueling DQN':<15} {dueling_dqn_results['avg_reward']:<12.2f} {dueling_dqn_results['std_reward']:<12.2f} {dueling_dqn_results['avg_steps']:<12.1f}")
    print("="*60 + "\n")
    
    return double_dqn, dueling_dqn, double_dqn_results, dueling_dqn_results


def visualize_comparison(double_dqn, dueling_dqn):
    """可视化对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 训练曲线对比
    ax = axes[0, 0]
    ax.plot(double_dqn.training_history, label='Double DQN', alpha=0.7)
    ax.plot(dueling_dqn.training_history, label='Dueling DQN', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Progress Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 收敛对比 (滑动平均)
    ax = axes[0, 1]
    window = 50
    double_dqn_smooth = np.convolve(double_dqn.training_history, np.ones(window)/window, mode='valid')
    dueling_dqn_smooth = np.convolve(dueling_dqn.training_history, np.ones(window)/window, mode='valid')
    ax.plot(double_dqn_smooth, label='Double DQN', linewidth=2)
    ax.plot(dueling_dqn_smooth, label='Dueling DQN', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Smoothed Reward')
    ax.set_title('Convergence Comparison (50-episode MA)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Q 值分布 (Double DQN)
    ax = axes[1, 0]
    q_values = double_dqn.q_table.flatten()
    q_values = q_values[q_values != 0]  # 过滤 0 值
    ax.hist(q_values, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Q-value')
    ax.set_ylabel('Frequency')
    ax.set_title('Q-value Distribution (Double DQN)')
    ax.grid(True, alpha=0.3)
    
    # 4. 价值函数可视化 (Dueling DQN)
    ax = axes[1, 1]
    im = ax.imshow(dueling_dqn.V_table.reshape(10, 10), cmap='viridis', origin='lower')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('State Value Function V(s) (Dueling DQN)')
    plt.colorbar(im, ax=ax, label='Value')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/.openclaw/workspace/cable-optimization/examples/outputs/dqn_variants_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ 对比图已保存")
    plt.show()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*60)
    print("进阶 DQN 变体 - Double DQN & Dueling DQN")
    print("="*60)
    
    # 对比实验
    double_dqn, dueling_dqn, double_results, dueling_results = compare_dqn_variants()
    
    # 可视化
    visualize_comparison(double_dqn, dueling_dqn)
    
    # 关键洞察
    print("\n" + "="*60)
    print("关键洞察")
    print("="*60)
    print("""
1. Double DQN 优势:
   - 解决 Q-learning 过估计问题
   - 使用 online network 选择动作，target network 评估价值
   - 更稳定的训练过程

2. Dueling DQN 优势:
   - 分离状态价值 V(s) 和优势函数 A(s,a)
   - 更好理解状态本身的价值
   - 在某些状态下动作选择不重要时更有效

3. 适用场景:
   - Double DQN: 通用场景，推荐使用
   - Dueling DQN: 状态价值差异明显的场景
   - 可以结合使用：Double Dueling DQN

4. 与 Week 1 启发式对比:
   - RL 学习"策略"而非直接优化"解"
   - 适应动态环境能力强
   - 训练成本高，但推理速度快
    """)
    print("="*60 + "\n")
    
    print("✅ Day 9 学习完成！进阶 DQN 变体实现成功！🎉")


if __name__ == "__main__":
    main()
