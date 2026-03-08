#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度 Q 网络 (DQN) / Q-Learning - 强化学习在线缆布线优化中的应用

作者：智子 (Sophon)
日期：2026-03-08
学习 Day：8 (Week 2 - 进阶算法)

算法核心：
- Q-Learning: 时序差分学习，更新 Q 值表
- DQN: 使用神经网络近似 Q 值函数（本实现使用简化版）
- ε-greedy 策略平衡探索与利用
- 经验回放思想（简化实现）

应用场景：
- 动态环境下的路径规划
- 多约束布线优化
- 在线学习与自适应调整

注意：由于环境限制，本实现使用 NumPy 实现简化版 Q-Learning，
     展示核心思想。完整 DQN 需要 PyTorch/TensorFlow。
"""

import numpy as np
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证可复现性
np.random.seed(42)
random.seed(42)

# ============================================================================
# 配置类
# ============================================================================

class QLearningConfig:
    """Q-Learning 算法配置参数"""
    
    def __init__(self):
        # 状态离散化
        self.state_bins = 5  # 每个维度的分箱数
        
        # 训练参数
        self.alpha = 0.1  # 学习率
        self.gamma = 0.99  # 折扣因子
        
        # 探索策略
        self.epsilon_start = 1.0  # 初始探索率
        self.epsilon_end = 0.01  # 最终探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        
        # 训练控制
        self.max_episodes = 500  # 最大训练轮次
        self.max_steps = 100  # 每轮最大步数


# ============================================================================
# 线缆布线环境
# ============================================================================

class CableRoutingEnv:
    """线缆布线环境 - 网格环境"""
    
    def __init__(self, grid_size: int = 10, num_obstacles: int = 15):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.action_space = 4  # 上、下、左、右
        self.action_names = ['上', '下', '左', '右']
        
        self.reset()
    
    def reset(self, seed: int = None) -> Tuple[int, int]:
        """重置环境"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 随机生成起点和终点
        self.agent_pos = [random.randint(0, self.grid_size-1), 
                         random.randint(0, self.grid_size-1)]
        self.goal_pos = [random.randint(0, self.grid_size-1), 
                        random.randint(0, self.grid_size-1)]
        
        # 确保起点和终点不同
        while self.agent_pos == self.goal_pos:
            self.goal_pos = [random.randint(0, self.grid_size-1), 
                           random.randint(0, self.grid_size-1)]
        
        # 随机生成障碍物
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            obs = (random.randint(0, self.grid_size-1), 
                  random.randint(0, self.grid_size-1))
            if obs != tuple(self.agent_pos) and obs != tuple(self.goal_pos):
                self.obstacles.add(obs)
        
        self.steps = 0
        self.path = [self.agent_pos.copy()]
        
        return tuple(self.agent_pos)
    
    def get_state_index(self, pos: List[int]) -> int:
        """将位置转换为状态索引"""
        return pos[0] * self.grid_size + pos[1]
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """执行动作"""
        self.steps += 1
        
        # 动作映射：0=上，1=下，2=左，3=右
        action_map = {
            0: [0, 1],   # 上
            1: [0, -1],  # 下
            2: [-1, 0],  # 左
            3: [1, 0]    # 右
        }
        
        # 计算新位置
        new_pos = [
            max(0, min(self.grid_size-1, self.agent_pos[0] + action_map[action][0])),
            max(0, min(self.grid_size-1, self.agent_pos[1] + action_map[action][1]))
        ]
        
        # 检查碰撞
        done = False
        reward = -0.1  # 每步惩罚，鼓励快速到达
        
        if tuple(new_pos) in self.obstacles:
            reward = -1.0  # 碰撞惩罚
            done = True
        elif new_pos == self.goal_pos:
            reward = 10.0  # 到达终点奖励
            done = True
        elif self.steps >= self.grid_size * 3:
            done = True  # 超时
            reward = -5.0
        else:
            self.agent_pos = new_pos
            self.path.append(self.agent_pos.copy())
        
        next_state = self.get_state_index(self.agent_pos)
        info = {'path': self.path.copy(), 'steps': self.steps, 
                'pos': self.agent_pos.copy()}
        
        return next_state, reward, done, info


# ============================================================================
# Q-Learning 训练器
# ============================================================================

class QLearningTrainer:
    """Q-Learning 训练器（简化版 DQN 思想）"""
    
    def __init__(self, env: CableRoutingEnv, config: QLearningConfig):
        self.env = env
        self.config = config
        
        # Q 表：state × action
        n_states = env.grid_size * env.grid_size
        n_actions = env.action_space
        self.q_table = np.zeros((n_states, n_actions))
        
        # 训练记录
        self.epsilon = config.epsilon_start
        self.episode_rewards = []
        self.episode_lengths = []
    
    def select_action(self, state: int) -> int:
        """ε-greedy 策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, 
                       next_state: int, done: bool):
        """Q-Learning 更新公式"""
        if done:
            target = reward
        else:
            target = reward + self.config.gamma * np.max(self.q_table[next_state])
        
        # Q 值更新
        self.q_table[state, action] += self.config.alpha * (target - self.q_table[state, action])
    
    def train_episode(self) -> Tuple[float, int]:
        """训练一个回合"""
        state = self.env.get_state_index(self.env.reset())
        total_reward = 0
        
        for step in range(self.config.max_steps):
            # 选择动作
            action = self.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 更新 Q 值
            self.update_q_value(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 更新探索率
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(info['steps'])
        
        return total_reward, info['steps']
    
    def train(self, verbose: bool = True) -> Dict:
        """完整训练流程"""
        if verbose:
            print(f"开始训练 Q-Learning...")
            print(f"状态空间：{self.env.grid_size}x{self.env.grid_size} = {self.env.grid_size**2}")
            print(f"动作空间：{self.env.action_space} ({self.env.action_names})")
            print(f"障碍物数量：{self.env.num_obstacles}")
            print("-" * 60)
        
        for episode in range(self.config.max_episodes):
            reward, length = self.train_episode()
            
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_length = np.mean(self.episode_lengths[-50:])
                print(f"Episode {episode+1}/{self.config.max_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Steps: {avg_length:.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        if verbose:
            print("-" * 60)
            print(f"训练完成！")
            print(f"最终平均奖励：{np.mean(self.episode_rewards[-50:]):.2f}")
            print(f"最终平均步数：{np.mean(self.episode_lengths[-50:]):.1f}")
            print(f"最佳奖励：{max(self.episode_rewards):.2f}")
        
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'epsilon': self.epsilon,
            'q_table': self.q_table.copy()
        }
    
    def get_best_action(self, state: int) -> int:
        """获取最优动作（贪婪策略）"""
        return np.argmax(self.q_table[state])


# ============================================================================
# 可视化
# ============================================================================

class RLVisualizer:
    """强化学习可视化"""
    
    @staticmethod
    def plot_training_results(rewards: List[float], lengths: List[int], 
                             save_path: str = None):
        """绘制训练结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 回合奖励曲线
        ax1 = axes[0, 0]
        ax1.plot(rewards, alpha=0.3, label='单回合奖励', color='gray')
        
        window = 50
        if len(rewards) >= window:
            ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), ma_rewards, 
                    label=f'{window}回合移动平均', color='blue', linewidth=2)
        
        ax1.set_xlabel('回合数')
        ax1.set_ylabel('奖励')
        ax1.set_title('训练奖励曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 回合长度曲线
        ax2 = axes[0, 1]
        ax2.plot(lengths, alpha=0.3, label='单回合步数', color='gray')
        
        if len(lengths) >= window:
            ma_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(lengths)), ma_lengths, 
                    label=f'{window}回合移动平均', color='green', linewidth=2)
        
        ax2.set_xlabel('回合数')
        ax2.set_ylabel('步数')
        ax2.set_title('路径长度曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 奖励分布
        ax3 = axes[1, 0]
        ax3.hist(rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(rewards), color='red', linestyle='--', 
                   label=f'平均：{np.mean(rewards):.2f}')
        ax3.set_xlabel('奖励')
        ax3.set_ylabel('频数')
        ax3.set_title('奖励分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 探索率变化
        ax4 = axes[1, 1]
        episodes = range(len(rewards))
        epsilon_curve = [max(0.01, 1.0 * (0.995 ** ep)) for ep in episodes]
        ax4.plot(episodes, epsilon_curve, color='purple', linewidth=2)
        ax4.set_xlabel('回合数')
        ax4.set_ylabel('探索率 ε')
        ax4.set_title('探索率衰减曲线')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存：{save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_q_table(q_table: np.ndarray, env: CableRoutingEnv, 
                    save_path: str = None):
        """绘制 Q 表热力图（显示每个状态的最优动作）"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 创建动作方向映射
        action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        action_colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
        
        grid = np.zeros((env.grid_size, env.grid_size))
        arrows = []
        
        for state in range(len(q_table)):
            x = state // env.grid_size
            y = state % env.grid_size
            best_action = np.argmax(q_table[state])
            best_q = q_table[state, best_action]
            
            grid[x, y] = best_q
            arrows.append((x, y, action_arrows[best_action], action_colors[best_action]))
        
        # 绘制热力图
        im = ax.imshow(grid, cmap='RdYlGn', origin='lower')
        
        # 绘制障碍物
        for obs in env.obstacles:
            rect = plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, 
                                color='red', alpha=0.6)
            ax.add_patch(rect)
        
        # 绘制箭头
        for x, y, arrow, color in arrows:
            if (y, x) not in env.obstacles:
                ax.text(y, x, arrow, ha='center', va='center', 
                       fontsize=8, color=color, fontweight='bold')
        
        # 标记起点和终点（示例位置）
        ax.plot(-0.5, -0.5, 'go', markersize=15, label='起点示例')
        ax.plot(env.grid_size-0.5, env.grid_size-0.5, 'rs', 
               markersize=15, label='终点示例')
        
        plt.colorbar(im, label='Q 值')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_title('Q 表策略热力图')
        ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Q 表图已保存：{save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_path(env: CableRoutingEnv, path: List[List[int]], 
                 title: str = "规划路径", save_path: str = None):
        """绘制路径"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制网格
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True, linestyle='-', linewidth=0.5)
        
        # 绘制障碍物
        for obs in env.obstacles:
            rect = plt.Rectangle((obs[0]-0.4, obs[1]-0.4), 0.8, 0.8, 
                                color='red', alpha=0.6)
            ax.add_patch(rect)
        
        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-o', linewidth=2, markersize=8, 
                   label='规划路径', alpha=0.7)
            
            ax.plot(path[0][0], path[0][1], 'go', markersize=15, 
                   label='起点', zorder=5)
            ax.plot(path[-1][0], path[-1][1], 'rs', markersize=15, 
                   label='终点', zorder=5)
        
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"路径图已保存：{save_path}")
        
        plt.show()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 训练和测试 Q-Learning"""
    
    print("=" * 70)
    print("Q-Learning / 简化 DQN - 线缆布线路径规划")
    print("=" * 70)
    
    # 创建环境
    env = CableRoutingEnv(grid_size=10, num_obstacles=15)
    
    # 创建配置
    config = QLearningConfig()
    config.max_episodes = 300
    config.max_steps = 50
    
    # 创建训练器
    trainer = QLearningTrainer(env, config)
    
    # 训练
    results = trainer.train(verbose=True)
    
    # 可视化训练结果
    print("\n生成训练可视化图表...")
    RLVisualizer.plot_training_results(
        results['rewards'], 
        results['lengths'],
        save_path='examples/outputs/12_dqn_training.png'
    )
    
    # 可视化 Q 表
    print("\n生成 Q 表策略图...")
    RLVisualizer.plot_q_table(
        results['q_table'],
        env,
        save_path='examples/outputs/12_dqn_qtable.png'
    )
    
    # 测试：运行几个回合展示效果
    print("\n" + "=" * 70)
    print("测试训练好的模型")
    print("=" * 70)
    
    test_episodes = 5
    test_rewards = []
    test_lengths = []
    
    for i in range(test_episodes):
        state = env.get_state_index(env.reset(seed=i+100))
        total_reward = 0
        done = False
        path = [env.agent_pos.copy()]
        
        while not done:
            action = trainer.get_best_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            path = info['path']
        
        test_rewards.append(total_reward)
        test_lengths.append(info['steps'])
        print(f"测试回合 {i+1}: 奖励 = {total_reward:.2f}, 步数 = {info['steps']}")
    
    # 绘制最后一个测试回合的路径
    print("\n绘制测试路径...")
    RLVisualizer.plot_path(
        env, path,
        title=f"Q-Learning 测试路径 (奖励：{test_rewards[-1]:.2f}, 步数：{test_lengths[-1]})",
        save_path='examples/outputs/12_dqn_test_path.png'
    )
    
    print(f"\n平均测试奖励：{np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"平均测试步数：{np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")
    
    # 保存训练结果
    results_summary = {
        'algorithm': 'Q-Learning (简化 DQN)',
        'total_episodes': len(results['rewards']),
        'final_avg_reward': float(np.mean(results['rewards'][-50:])),
        'best_reward': float(max(results['rewards'])),
        'test_avg_reward': float(np.mean(test_rewards)),
        'test_avg_steps': float(np.mean(test_lengths)),
        'final_epsilon': float(trainer.epsilon),
        'grid_size': env.grid_size,
        'num_obstacles': env.num_obstacles,
        'state_space': env.grid_size ** 2,
        'action_space': env.action_space
    }
    
    with open('examples/outputs/12_dqn_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("Q-Learning 训练完成！")
    print("=" * 70)
    
    return results_summary


if __name__ == "__main__":
    main()
