"""
复合 DRL 路径规划系统 - 基于关键路径点

论文参考：
Sun 等 (2026). Deep reinforcement learning-based composite path planning with key path points

核心思想:
1. 层次化规划：高层选择关键点 + 低层点间规划
2. 搜索空间缩减：O(n²) → O(k²), k<<n
3. DRL + 传统算法结合：DRL 选点 + A*规划

作者：智子 (Sophon)
日期：2026-03-08
版本：v1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
import math
from typing import List, Tuple, Dict, Optional
import random

# ============================================================================
# Part 1: 环境定义
# ============================================================================

class GridEnvironment:
    """
    网格环境类
    
    功能:
    - 创建网格地图
    - 添加障碍物
    - 碰撞检测
    - 可视化
    """
    
    def __init__(self, width: int = 20, height: int = 20, seed: int = 42):
        """
        初始化环境
        
        Args:
            width: 网格宽度
            height: 网格高度
            seed: 随机种子
        """
        self.width = width
        self.height = height
        self.seed = seed
        
        # 网格地图 (0=可通行，1=障碍物)
        np.random.seed(seed)
        self.grid = np.zeros((height, width), dtype=np.int8)
        
        # 起点和终点
        self.start = (0, 0)
        self.goal = (width - 1, height - 1)
        
        # 关键点列表
        self.key_points = []
    
    def add_obstacles(self, num_obstacles: int = 20, 
                     obstacle_size: int = 2) -> None:
        """
        随机添加障碍物
        
        Args:
            num_obstacles: 障碍物数量
            obstacle_size: 障碍物大小
        """
        for _ in range(num_obstacles):
            # 随机位置
            x = np.random.randint(0, self.width - obstacle_size)
            y = np.random.randint(0, self.height - obstacle_size)
            
            # 避开起点和终点
            if abs(x - self.start[0]) < 3 and abs(y - self.start[1]) < 3:
                continue
            if abs(x - self.goal[0]) < 3 and abs(y - self.goal[1]) < 3:
                continue
            
            # 添加障碍物
            for dx in range(obstacle_size):
                for dy in range(obstacle_size):
                    if 0 <= x+dx < self.width and 0 <= y+dy < self.height:
                        self.grid[y+dy, x+dx] = 1
    
    def is_valid(self, x: int, y: int) -> bool:
        """
        检查位置是否有效
        
        Args:
            x: x 坐标
            y: y 坐标
            
        Returns:
            是否有效 (在边界内且无障碍)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 0
        return False
    
    def generate_key_points(self, num_points: int = 5) -> List[Tuple[int, int]]:
        """
        生成关键点
        
        Args:
            num_points: 关键点数量
            
        Returns:
            关键点列表
        """
        key_points = []
        
        # 添加起点和终点作为关键点
        key_points.append(self.start)
        
        # 随机选择可通行位置作为关键点
        attempts = 0
        while len(key_points) < num_points + 1 and attempts < 1000:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            
            # 检查是否可通行
            if not self.is_valid(x, y):
                attempts += 1
                continue
            
            # 检查是否与已有关键点太近
            too_close = False
            for kp in key_points:
                dist = math.hypot(x - kp[0], y - kp[1])
                if dist < 3:
                    too_close = True
                    break
            
            if not too_close:
                key_points.append((x, y))
            
            attempts += 1
        
        # 添加终点
        key_points.append(self.goal)
        
        self.key_points = key_points
        return key_points
    
    def visualize(self, path: List[Tuple[int, int]] = None,
                 key_points: List[Tuple[int, int]] = None,
                 save_path: str = None) -> None:
        """
        可视化环境
        
        Args:
            path: 路径
            key_points: 关键点
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制网格
        ax.imshow(self.grid, cmap='binary', origin='lower')
        
        # 绘制关键点
        if key_points:
            kpx = [kp[0] for kp in key_points]
            kpy = [kp[1] for kp in key_points]
            ax.scatter(kpx, kpy, c='red', s=200, marker='*', 
                      label='Key Points', zorder=5)
            for i, kp in enumerate(key_points):
                ax.annotate(f'KP{i}', (kp[0]+0.2, kp[1]+0.2), 
                           fontsize=12, color='red', weight='bold')
        
        # 绘制路径
        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, 'b-', linewidth=2, label='Path', zorder=4)
            ax.scatter([path[0][0]], [path[0][1]], c='green', 
                      s=200, marker='o', label='Start', zorder=5)
            ax.scatter([path[-1][0]], [path[-1][1]], c='blue', 
                      s=200, marker='s', label='Goal', zorder=5)
        
        # 设置
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Grid Environment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图像已保存到：{save_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================================================
# Part 2: A* 路径规划器 (低层规划)
# ============================================================================

class AStarPlanner:
    """
    A* 路径规划器
    
    功能:
    - 点间路径规划
    - 启发式搜索
    - 路径平滑
    """
    
    def __init__(self, env: GridEnvironment):
        """
        初始化 A* 规划器
        
        Args:
            env: 网格环境
        """
        self.env = env
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        启发式函数 (欧氏距离)
        
        Args:
            a: 点 a
            b: 点 b
            
        Returns:
            欧氏距离
        """
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        获取邻居节点 (8 方向)
        
        Args:
            x: x 坐标
            y: y 坐标
            
        Returns:
            邻居节点列表
        """
        neighbors = []
        # 8 方向
        directions = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.env.is_valid(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def plan(self, start: Tuple[int, int], 
            goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* 路径规划
        
        Args:
            start: 起点
            goal: 终点
            
        Returns:
            路径列表，如果无路径返回 None
        """
        # 优先队列：(f_score, x, y)
        open_set = [(0, start[0], start[1])]
        
        # 记录访问
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, x, y = heapq.heappop(open_set)
            current = (x, y)
            
            # 到达终点
            if current == goal:
                # 重建路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            # 遍历邻居
            for nx, ny in self.get_neighbors(x, y):
                neighbor = (nx, ny)
                
                # 计算 g_score
                move_cost = math.hypot(nx - x, ny - y)
                tentative_g = g_score[current] + move_cost
                
                # 如果找到更好的路径
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, nx, ny))
        
        # 无路径
        return None
    
    def plan_through_key_points(self, key_points: List[Tuple[int, int]]) \
                                -> Optional[List[Tuple[int, int]]]:
        """
        通过关键点的路径规划
        
        Args:
            key_points: 关键点列表
            
        Returns:
            完整路径，如果任何段无路径返回 None
        """
        if len(key_points) < 2:
            return None
        
        full_path = []
        
        # 逐段规划
        for i in range(len(key_points) - 1):
            start = key_points[i]
            goal = key_points[i + 1]
            
            segment = self.plan(start, goal)
            if segment is None:
                print(f"⚠️  关键点 {i} 到 {i+1} 无路径")
                return None
            
            # 添加段路径 (避免重复点)
            if i == 0:
                full_path.extend(segment)
            else:
                full_path.extend(segment[1:])
        
        return full_path


# ============================================================================
# Part 3: DRL 关键点选择器 (高层规划)
# ============================================================================

class DQNKeyPointSelector:
    """
    DQN 关键点选择器
    
    功能:
    - 学习选择最优关键点序列
    - Q-learning 更新
    - ε-greedy 策略
    """
    
    def __init__(self, env: GridEnvironment,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化 DQN 关键点选择器
        
        Args:
            env: 网格环境
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q 表：状态 -> 动作 -> 价值
        # 状态：当前关键点索引
        # 动作：下一个关键点索引
        self.q_table = {}
        
        # 训练历史
        self.training_history = []
    
    def get_state(self, current_kp_idx: int, visited: tuple) -> tuple:
        """
        获取状态表示
        
        Args:
            current_kp_idx: 当前关键点索引
            visited: 已访问的关键点
            
        Returns:
            状态元组
        """
        return (current_kp_idx, visited)
    
    def get_action(self, state: tuple, available_actions: List[int]) -> int:
        """
        ε-greedy 策略选择动作
        
        Args:
            state: 状态
            available_actions: 可用动作列表
            
        Returns:
            选择的动作
        """
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.choice(available_actions)
        else:
            # 利用：选择 Q 值最大的动作
            if state not in self.q_table:
                self.q_table[state] = {a: 0.0 for a in available_actions}
            
            q_values = self.q_table[state]
            max_q = max([q_values.get(a, 0.0) for a in available_actions])
            
            # 从最大值中随机选择 (打破平局)
            best_actions = [a for a in available_actions 
                          if q_values.get(a, 0.0) == max_q]
            return np.random.choice(best_actions)
    
    def update_q(self, state: tuple, action: int, reward: float, 
                next_state: tuple, next_actions: List[int]) -> None:
        """
        Q-learning 更新
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 奖励
            next_state: 下一状态
            next_actions: 下一状态的可用动作
        """
        # 初始化 Q 值
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # 计算最大 Q 值
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in next_actions}
        
        max_next_q = max([self.q_table[next_state].get(a, 0.0) 
                         for a in next_actions])
        
        # Q-learning 更新
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.lr * (
            reward + self.gamma * max_next_q - old_q
        )
    
    def calculate_reward(self, path: Optional[List[Tuple[int, int]]],
                        num_key_points: int) -> float:
        """
        计算奖励
        
        Args:
            path: 规划的路径
            num_key_points: 关键点数量
            
        Returns:
            奖励值
        """
        if path is None:
            # 无路径：大惩罚
            return -1000.0
        
        # 路径长度奖励 (越短越好)
        path_length = sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) 
                         for p1, p2 in zip(path, path[1:]))
        length_reward = -path_length * 0.1
        
        # 关键点数量奖励 (适中为好)
        kp_reward = -abs(num_key_points - 5) * 10.0
        
        # 平滑度奖励 (转弯少为好)
        smoothness_reward = 0.0
        if len(path) > 2:
            turns = 0
            for i in range(1, len(path) - 1):
                v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                # 检查是否转弯
                if v1 != v2:
                    turns += 1
            smoothness_reward = -turns * 5.0
        
        total_reward = length_reward + kp_reward + smoothness_reward
        return total_reward
    
    def train(self, planner: AStarPlanner, n_episodes: int = 100) -> List[float]:
        """
        训练 DQN 关键点选择器
        
        Args:
            planner: A* 规划器
            n_episodes: 训练回合数
            
        Returns:
            每回合奖励历史
        """
        print(f"\n{'='*60}")
        print(f"DQN 关键点选择器训练")
        print(f"{'='*60}")
        print(f"关键点数量：{len(self.env.key_points)}")
        print(f"训练回合：{n_episodes}")
        print(f"{'='*60}\n")
        
        rewards_per_episode = []
        
        for episode in range(n_episodes):
            # 重置环境
            key_points = self.env.generate_key_points(num_points=5)
            
            # 初始化
            current_kp_idx = 0
            visited = (0,)
            state = self.get_state(current_kp_idx, visited)
            
            episode_reward = 0.0
            selected_kps = [key_points[0]]
            
            # 选择关键点序列
            while current_kp_idx < len(key_points) - 1:
                # 可用动作：未访问的关键点
                available_actions = [i for i in range(len(key_points)) 
                                    if i not in visited]
                
                if not available_actions:
                    break
                
                # 选择动作
                action = self.get_action(state, available_actions)
                
                # 执行动作
                next_kp_idx = action
                visited = visited + (next_kp_idx,)
                next_state = self.get_state(next_kp_idx, visited)
                
                # 规划路径
                temp_kps = [key_points[i] for i in visited]
                path = planner.plan_through_key_points(temp_kps)
                
                # 计算奖励
                reward = self.calculate_reward(path, len(visited))
                
                # Q-learning 更新
                next_available = [i for i in range(len(key_points)) 
                                 if i not in visited]
                if next_available:
                    self.update_q(state, action, reward, next_state, 
                                 next_available)
                
                # 更新状态
                state = next_state
                current_kp_idx = next_kp_idx
                selected_kps.append(key_points[next_kp_idx])
                episode_reward += reward
            
            # 最终路径规划
            final_path = planner.plan_through_key_points(selected_kps)
            final_reward = self.calculate_reward(final_path, len(selected_kps))
            episode_reward += final_reward
            
            rewards_per_episode.append(episode_reward)
            
            # 衰减 ε
            self.epsilon = max(self.epsilon_min, 
                              self.epsilon * self.epsilon_decay)
            
            # 打印进度
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(rewards_per_episode[-20:])
                print(f"回合 {episode+1}/{n_episodes}: "
                      f"平均奖励 = {avg_reward:.2f}, "
                      f"ε = {self.epsilon:.3f}")
        
        self.training_history = rewards_per_episode
        return rewards_per_episode
    
    def select_best_sequence(self, planner: AStarPlanner) \
                             -> Tuple[List[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
        """
        选择最优关键点序列
        
        Args:
            planner: A* 规划器
            
        Returns:
            关键点序列，完整路径
        """
        key_points = self.env.key_points
        
        # 贪婪选择
        current_kp_idx = 0
        visited = [0]
        selected_kps = [key_points[0]]
        
        while current_kp_idx < len(key_points) - 1:
            # 可用动作
            available_actions = [i for i in range(len(key_points)) 
                                if i not in visited]
            
            if not available_actions:
                break
            
            # 选择 Q 值最大的动作
            state = self.get_state(current_kp_idx, tuple(visited))
            
            if state not in self.q_table:
                # 未训练过，随机选择
                next_kp_idx = np.random.choice(available_actions)
            else:
                q_values = self.q_table[state]
                next_kp_idx = max(available_actions, 
                                 key=lambda a: q_values.get(a, 0.0))
            
            visited.append(next_kp_idx)
            selected_kps.append(key_points[next_kp_idx])
            current_kp_idx = next_kp_idx
        
        # 规划完整路径
        full_path = planner.plan_through_key_points(selected_kps)
        
        return selected_kps, full_path


# ============================================================================
# Part 4: 复合 DRL 路径规划系统
# ============================================================================

class CompositeDRLPlanner:
    """
    复合 DRL 路径规划系统
    
    架构:
    - 高层：DQN 关键点选择
    - 低层：A* 点间规划
    
    优势:
    - 搜索空间缩减
    - 结合 DRL 和传统算法优势
    - 适应性强
    """
    
    def __init__(self, env: GridEnvironment):
        """
        初始化复合 DRL 规划器
        
        Args:
            env: 网格环境
        """
        self.env = env
        self.planner = AStarPlanner(env)
        self.selector = DQNKeyPointSelector(env)
    
    def train(self, n_episodes: int = 100) -> List[float]:
        """
        训练规划器
        
        Args:
            n_episodes: 训练回合数
            
        Returns:
            奖励历史
        """
        return self.selector.train(self.planner, n_episodes)
    
    def plan(self, train_first: bool = True, 
            n_train_episodes: int = 100) -> Tuple[List[Tuple[int, int]], 
                                                   Optional[List[Tuple[int, int]]]]:
        """
        路径规划
        
        Args:
            train_first: 是否先训练
            n_train_episodes: 训练回合数
            
        Returns:
            关键点序列，完整路径
        """
        # 生成关键点
        key_points = self.env.generate_key_points(num_points=5)
        
        print(f"\n{'='*60}")
        print(f"复合 DRL 路径规划")
        print(f"{'='*60}")
        print(f"关键点：{len(key_points)} 个")
        print(f"起点：{self.env.start}")
        print(f"终点：{self.env.goal}")
        print(f"{'='*60}\n")
        
        # 训练
        if train_first:
            self.train(n_train_episodes)
        
        # 选择最优序列
        selected_kps, full_path = self.selector.select_best_sequence(self.planner)
        
        # 输出结果
        print(f"✓ 关键点序列：{len(selected_kps)} 个")
        for i, kp in enumerate(selected_kps):
            print(f"  KP{i}: {kp}")
        
        if full_path:
            print(f"✓ 完整路径：{len(full_path)} 个点")
            print(f"✓ 路径长度：{sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in zip(full_path, full_path[1:])):.2f}")
        else:
            print(f"✗ 无可行路径")
        
        print(f"\n{'='*60}\n")
        
        return selected_kps, full_path
    
    def visualize(self, selected_kps: List[Tuple[int, int]], 
                 full_path: Optional[List[Tuple[int, int]]],
                 save_path: str = None) -> None:
        """
        可视化结果
        
        Args:
            selected_kps: 关键点序列
            full_path: 完整路径
            save_path: 保存路径
        """
        self.env.visualize(path=full_path, key_points=selected_kps, 
                          save_path=save_path)
    
    def compare_with_astar(self) -> Dict:
        """
        与纯 A* 对比
        
        Returns:
            对比结果字典
        """
        print(f"\n{'='*60}")
        print(f"与纯 A* 对比")
        print(f"{'='*60}\n")
        
        # 纯 A*
        import time
        start_time = time.time()
        astar_path = self.planner.plan(self.env.start, self.env.goal)
        astar_time = time.time() - start_time
        
        # 复合 DRL
        start_time = time.time()
        selected_kps, composite_path = self.plan(train_first=False)
        composite_time = time.time() - start_time
        
        # 对比
        results = {
            'astar': {
                'path_length': sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) 
                                  for p1, p2 in zip(astar_path, astar_path[1:])) if astar_path else None,
                'time': astar_time,
                'success': astar_path is not None
            },
            'composite_drl': {
                'path_length': sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) 
                                  for p1, p2 in zip(composite_path, composite_path[1:])) if composite_path else None,
                'time': composite_time,
                'success': composite_path is not None,
                'num_key_points': len(selected_kps)
            }
        }
        
        print(f"纯 A*:")
        print(f"  路径长度：{results['astar']['path_length']:.2f}" if results['astar']['path_length'] else "  无路径")
        print(f"  规划时间：{results['astar']['time']*1000:.2f} ms")
        print(f"  成功率：{'✓' if results['astar']['success'] else '✗'}")
        
        print(f"\n复合 DRL:")
        print(f"  关键点数量：{results['composite_drl']['num_key_points']}")
        print(f"  路径长度：{results['composite_drl']['path_length']:.2f}" if results['composite_drl']['path_length'] else "  无路径")
        print(f"  规划时间：{results['composite_drl']['time']*1000:.2f} ms")
        print(f"  成功率：{'✓' if results['composite_drl']['success'] else '✗'}")
        
        if results['astar']['path_length'] and results['composite_drl']['path_length']:
            length_diff = (results['composite_drl']['path_length'] - results['astar']['path_length']) / results['astar']['path_length'] * 100
            print(f"\n  路径长度差异：{length_diff:+.2f}%")
        
        print(f"\n{'='*60}\n")
        
        return results


# ============================================================================
# Part 5: 主函数与测试
# ============================================================================

def main():
    """
    主函数 - 演示复合 DRL 路径规划系统
    """
    print("="*60)
    print("复合 DRL 路径规划系统 v1.0")
    print("="*60)
    print()
    print("论文参考:")
    print("Sun 等 (2026). Deep reinforcement learning-based")
    print("composite path planning with key path points")
    print()
    print("="*60)
    print()
    
    # 创建环境
    print("1. 创建环境...")
    env = GridEnvironment(width=20, height=20, seed=42)
    env.add_obstacles(num_obstacles=15, obstacle_size=2)
    print(f"   ✓ 环境大小：{env.width}x{env.height}")
    print(f"   ✓ 障碍物：15 个")
    print()
    
    # 创建规划器
    print("2. 创建复合 DRL 规划器...")
    planner = CompositeDRLPlanner(env)
    print(f"   ✓ 高层：DQN 关键点选择")
    print(f"   ✓ 低层：A* 点间规划")
    print()
    
    # 训练与规划
    print("3. 训练与规划...")
    selected_kps, full_path = planner.plan(train_first=True, n_train_episodes=100)
    
    # 可视化
    print("4. 可视化结果...")
    planner.visualize(selected_kps, full_path, 
                     save_path='/root/.openclaw/workspace/cable-optimization/examples/outputs/composite_drl_path.png')
    print()
    
    # 对比实验
    print("5. 与纯 A* 对比...")
    results = planner.compare_with_astar()
    
    # 训练曲线
    print("6. 绘制训练曲线...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(planner.selector.training_history, 'b-', linewidth=1, alpha=0.5, label='Episode Reward')
    
    # 滑动平均
    window = 10
    if len(planner.selector.training_history) >= window:
        smoothed = np.convolve(planner.selector.training_history, 
                              np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'r-', linewidth=2, label=f'{window}-episode MA')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('DQN Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = '/root/.openclaw/workspace/cable-optimization/examples/outputs/dqn_training_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 训练曲线已保存到：{save_path}")
    plt.close()
    
    print()
    print("="*60)
    print("✅ 复合 DRL 路径规划系统演示完成！")
    print("="*60)
    print()
    print("输出文件:")
    print("  1. composite_drl_path.png - 路径可视化")
    print("  2. dqn_training_curve.png - 训练曲线")
    print()
    print("下一步:")
    print("  1. 调整参数优化性能")
    print("  2. 在更大规模问题上测试")
    print("  3. 与更多基线算法对比")
    print()


if __name__ == "__main__":
    main()
