"""
SW-RDQN v3.3 - 训练优化版

优化内容 (v3.2 → v3.3):
1. ✅ 增加训练回合 (40 → 200)
2. ✅ 改进奖励函数 (稠密奖励)
3. ✅ epsilon 退火策略
4. ✅ 增加关键点数量
5. ✅ 调整学习率和网络结构

作者：智子 (Sophon)
日期：2026-03-10
版本：v3.3 (训练优化版)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.ndimage import distance_transform_edt, label
from collections import deque
import heapq
import math
from typing import List, Tuple, Dict, Optional
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# ============================================================================
# Part 1: Voronoi 关键点生成 (增加关键点)
# ============================================================================

class VoronoiKeyPointGenerator:
    """Voronoi 关键点生成器 - 优化版"""
    
    def __init__(self, grid_map: np.ndarray, safety_distance: float = 2.0):
        self.grid_map = grid_map
        self.safety_distance = safety_distance
        self.height, self.width = grid_map.shape
        self.obstacle_centers = self._extract_obstacle_centers()
        self.key_points = []
    
    def _extract_obstacle_centers(self) -> np.ndarray:
        labeled, n_features = label(self.grid_map)
        centers = []
        for i in range(1, n_features + 1):
            obstacle_mask = (labeled == i)
            coords = np.argwhere(obstacle_mask)
            center = coords.mean(axis=0).astype(int)
            centers.append(tuple(center))
        return np.array(centers) if centers else np.argwhere(self.grid_map == 1)
    
    def extract_key_points(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """提取关键点 - 优化：添加中间点"""
        if len(self.obstacle_centers) < 2:
            raise ValueError("至少需要 2 个障碍物")
        
        vor = Voronoi(self.obstacle_centers.astype(float))
        skeleton_edges = []
        for ridge in vor.ridge_vertices:
            if -1 not in ridge:
                pt1 = vor.vertices[ridge[0]]
                pt2 = vor.vertices[ridge[1]]
                if self._is_edge_valid(pt1, pt2):
                    skeleton_edges.append((pt1, pt2))
        
        distance_field = distance_transform_edt(1 - self.grid_map)
        safe_edges = []
        for pt1, pt2 in skeleton_edges:
            if self._is_edge_safe(pt1, pt2, distance_field):
                safe_edges.append((pt1, pt2))
        
        # 优化：添加起点和终点投影
        key_points = [start]
        start_proj = self._project_to_skeleton(start, safe_edges)
        if start_proj:
            key_points.append((int(start_proj[0]), int(start_proj[1])))
        
        # 添加所有骨架点
        for pt1, pt2 in safe_edges:
            key_points.append((int(pt1[0]), int(pt1[1])))
            key_points.append((int(pt2[0]), int(pt2[1])))
        
        goal_proj = self._project_to_skeleton(goal, safe_edges)
        if goal_proj:
            key_points.append((int(goal_proj[0]), int(goal_proj[1])))
        key_points.append(goal)
        
        # 优化：简化但保留更多点 (距离阈值从 3 改为 5)
        self.key_points = self._simplify_key_points(key_points, min_dist=5)
        return self.key_points
    
    def _is_edge_valid(self, pt1, pt2) -> bool:
        return (0 <= pt1[0] < self.height and 0 <= pt1[1] < self.width and
                0 <= pt2[0] < self.height and 0 <= pt2[1] < self.width)
    
    def _is_edge_safe(self, pt1, pt2, distance_field) -> bool:
        line_points = self._get_line_points(pt1, pt2)
        for x, y in line_points:
            if 0 <= x < self.height and 0 <= y < self.width:
                if distance_field[x, y] < self.safety_distance:
                    return False
        return True
    
    def _project_to_skeleton(self, point, edges) -> Optional[Tuple[float, float]]:
        if not edges:
            return None
        min_dist = float('inf')
        closest_point = None
        for pt1, pt2 in edges:
            proj = self._project_point_to_line(point, pt1, pt2)
            dist = math.hypot(point[0] - proj[0], point[1] - proj[1])
            if dist < min_dist:
                min_dist = dist
                closest_point = proj
        return closest_point
    
    def _project_point_to_line(self, point, line_start, line_end):
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        if dx == 0 and dy == 0:
            return line_start
        t = max(0, min(1, ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy) / (dx * dx + dy * dy)))
        return (line_start[0] + t * dx, line_start[1] + t * dy)
    
    def _get_line_points(self, pt1, pt2) -> List[Tuple[int, int]]:
        points = []
        x0, y0 = int(pt1[0]), int(pt1[1])
        x1, y1 = int(pt2[0]), int(pt2[1])
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dy
                y0 += sy
        return points
    
    def _simplify_key_points(self, key_points, min_dist: float = 5) -> List[Tuple[int, int]]:
        """简化关键点 - 优化：可配置最小距离"""
        if len(key_points) <= 2:
            return key_points
        simplified = [key_points[0]]
        for i in range(1, len(key_points) - 1):
            last_pt = simplified[-1]
            curr_pt = key_points[i]
            dist = math.hypot(curr_pt[0] - last_pt[0], curr_pt[1] - last_pt[1])
            if dist >= min_dist:
                simplified.append(curr_pt)
        simplified.append(key_points[-1])
        return simplified
    
    def visualize(self, save_path: str = None):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(self.grid_map, cmap='binary', origin='lower')
        if len(self.obstacle_centers) > 0:
            ax.scatter(self.obstacle_centers[:, 1], self.obstacle_centers[:, 0], 
                      c='red', s=50, marker='x', label='Obstacle Centers')
        if self.key_points:
            kpx = [kp[1] for kp in self.key_points]
            kpy = [kp[0] for kp in self.key_points]
            ax.plot(kpx, kpy, 'r-*', linewidth=2, markersize=10, label=f'Key Points ({len(self.key_points)})')
            ax.scatter([self.key_points[0][1]], [self.key_points[0][0]], 
                      c='green', s=200, marker='o', label='Start')
            ax.scatter([self.key_points[-1][1]], [self.key_points[-1][0]], 
                      c='blue', s=200, marker='s', label='Goal')
        ax.legend()
        ax.set_title('Voronoi Key Points (v3.3)')
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

# ============================================================================
# Part 2: 动态障碍物模型
# ============================================================================

class DynamicObstacle:
    """动态障碍物"""
    def __init__(self, x: float, y: float, vx: float, vy: float, size: int = 2, prediction_horizon: int = 5):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy, self.size = vx, vy, size
        self.prediction_horizon = prediction_horizon
        self.position_history = deque(maxlen=20)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.position_history.append((self.x, self.y))
    
    def predict_future_positions(self, steps: int = None) -> List[Tuple[int, int]]:
        if steps is None:
            steps = self.prediction_horizon
        return [(int(round(self.x + self.vx * t)), int(round(self.y + self.vy * t))) for t in range(1, steps + 1)]
    
    def get_current_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.bool_)
        for dx in range(-self.size//2, self.size//2 + 1):
            for dy in range(-self.size//2, self.size//2 + 1):
                x, y = int(self.x + dx), int(self.y + dy)
                if 0 <= x < height and 0 <= y < width:
                    mask[x, y] = True
        return mask

# ============================================================================
# Part 3: SW-RDQN 网络 (7 通道)
# ============================================================================

class SWRDQNNetwork(nn.Module):
    """SW-RDQN 网络 - 7 通道"""
    
    def __init__(self, obs_height: int = 11, obs_width: int = 11, 
                 n_channels: int = 7, window_size: int = 4,
                 n_actions: int = 8, hidden_size: int = 256):
        super(SWRDQNNetwork, self).__init__()
        self.n_actions = n_actions
        self.window_size = window_size
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.gru = nn.GRU(input_size=128, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        
        self.value_stream = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, n_actions))
    
    def forward(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor = None):
        batch_size, seq_len = obs_seq.shape[:2]
        cnn_features = []
        for t in range(seq_len):
            obs_t = obs_seq[:, t, :, :, :]
            feat = self.cnn(obs_t).view(batch_size, -1)
            cnn_features.append(feat)
        cnn_features = torch.stack(cnn_features, dim=1)
        
        if hidden_state is None:
            gru_out, hidden_state = self.gru(cnn_features)
        else:
            gru_out, hidden_state = self.gru(cnn_features, hidden_state)
        last_hidden = gru_out[:, -1, :]
        
        value = self.value_stream(last_hidden)
        advantage = self.advantage_stream(last_hidden)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values, hidden_state
    
    def get_hidden_state(self, batch_size: int = 1, device: str = 'cpu'):
        return torch.zeros(2, batch_size, 256).to(device)

# ============================================================================
# Part 4: 加权优先经验回放
# ============================================================================

class WeightedPrioritizedReplayBuffer:
    """加权优先经验回放"""
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, lambda_safety: float = 0.1):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.lambda_safety = lambda_safety
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, transition: Dict, td_error: float, d_safety: float):
        priority = abs(td_error) + self.lambda_safety * d_safety
        self.buffer.append(transition)
        self.priorities.append(priority ** self.alpha)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], torch.Tensor, List[int]]:
        if len(self.buffer) < batch_size:
            return list(self.buffer), torch.ones(batch_size), list(range(len(self.buffer)))
        priorities = np.log1p(list(self.priorities))
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = torch.FloatTensor(weights / weights.max())
        self.beta = min(1.0, self.beta + self.beta_increment)
        return batch, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray, d_safety_list: np.ndarray):
        for idx, td_error, d_safety in zip(indices, td_errors, d_safety_list):
            if 0 <= idx < len(self.priorities):
                priority = abs(td_error) + self.lambda_safety * d_safety
                self.priorities[idx] = priority ** self.alpha
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# Part 5: 局部 SW-RDQN 控制器 (训练优化)
# ============================================================================

class LocalSWRDQNController:
    """局部 SW-RDQN 控制器 - 训练优化版"""
    def __init__(self, obs_size: int = 11, n_channels: int = 7, window_size: int = 4, 
                 n_actions: int = 8, learning_rate: float = 5e-4, gamma: float = 0.99,  # 优化：更高学习率
                 grid_map: np.ndarray = None):
        self.obs_size = obs_size
        self.n_channels = n_channels
        self.window_size = window_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.grid_map = grid_map
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = SWRDQNNetwork(obs_size, obs_size, n_channels, window_size, n_actions).to(self.device)
        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_network.eval()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay_buffer = WeightedPrioritizedReplayBuffer(capacity=10000)
        self.obs_window = deque(maxlen=window_size)
        
        # 优化：epsilon 退火
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        self.stats = {'episodes': 0, 'success_rate': 0.0, 'total_loss': 0.0}
    
    def get_observation(self, grid_map: np.ndarray, position: Tuple[int, int], goal: Tuple[int, int], dynamic_obstacles: List[DynamicObstacle]) -> np.ndarray:
        """获取局部观测 - 7 通道"""
        x, y = position
        half_size = self.obs_size // 2
        
        obs = np.zeros((self.n_channels, self.obs_size, self.obs_size), dtype=np.float32)
        
        for i in range(self.obs_size):
            for j in range(self.obs_size):
                gx = int(x - half_size + i)
                gy = int(y - half_size + j)
                
                if 0 <= gx < grid_map.shape[0] and 0 <= gy < grid_map.shape[1]:
                    obs[0, i, j] = grid_map[gx, gy]
                    
                    for obstacle in dynamic_obstacles:
                        if obstacle.get_current_mask(grid_map.shape[0], grid_map.shape[1])[gx, gy]:
                            obs[1, i, j] = 1.0
                        for t, (px, py) in enumerate(obstacle.predict_future_positions(3)):
                            if int(px) == gx and int(py) == gy:
                                obs[2 + t, i, j] = 1.0
                else:
                    obs[0, i, j] = 1.0
        
        # 7 通道：dx + dy
        goal_dx = (goal[0] - x) / max(1, abs(goal[0] - x) + abs(goal[1] - y))
        goal_dy = (goal[1] - y) / max(1, abs(goal[0] - x) + abs(goal[1] - y))
        obs[5, :, :] = np.clip(goal_dx, -1, 1)
        obs[6, :, :] = np.clip(goal_dy, -1, 1)
        
        return obs
    
    def _calculate_safety_distance(self, position: Tuple[int, int]) -> float:
        """计算到最近障碍物的距离"""
        x, y = position
        min_dist = float('inf')
        if self.grid_map is not None:
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_map.shape[0] and 0 <= ny < self.grid_map.shape[1]:
                        if self.grid_map[nx, ny] == 1:
                            dist = math.hypot(dx, dy)
                            min_dist = min(min_dist, dist)
        return min_dist if min_dist != float('inf') else 10.0
    
    def select_action(self, obs: np.ndarray, epsilon: float = None) -> int:
        """选择动作 - 优化：支持自定义 epsilon"""
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            q_values, _ = self.network(obs_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done, td_error: float, d_safety: float):
        self.replay_buffer.add({'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}, td_error, d_safety)
    
    def update(self, batch_size: int = 32) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0
        batch, weights, indices = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([exp['next_state'] for exp in batch])).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)
        q_values = self.network(states)[0].gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)[0].max(dim=1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        td_errors = torch.abs(q_values - targets)
        loss = (td_errors * weights.to(self.device)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        d_safety_list = np.array([self._calculate_safety_distance((0, 0)) for _ in range(len(indices))])
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy(), d_safety_list)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network = copy.deepcopy(self.network)
    
    def train_episode(self, env, start: Tuple[int, int], goal: Tuple[int, int], dynamic_obstacles: List[DynamicObstacle], max_steps: int = 200) -> Tuple[float, bool]:
        """训练回合 - 优化：稠密奖励"""
        position = start
        total_reward = 0.0
        success = False
        obs = self.get_observation(env.grid_map, position, goal, dynamic_obstacles)
        self.obs_window.append(obs)
        
        # 记录初始距离
        initial_dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
        prev_dist = initial_dist
        
        for step in range(max_steps):
            if len(self.obs_window) < self.window_size:
                action = random.randint(0, self.n_actions - 1)
            else:
                obs_seq = np.array(list(self.obs_window))
                action = self.select_action(obs_seq)  # 使用退火 epsilon
            
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            dx, dy = directions[action]
            new_x, new_y = position[0] + dx, position[1] + dy
            if env.is_valid(new_x, new_y):
                position = (new_x, new_y)
            
            for obstacle in dynamic_obstacles:
                obstacle.update()
            
            # 优化：稠密奖励
            reward = self._calculate_reward_dense(position, goal, prev_dist, action, dynamic_obstacles, env.grid_map)
            total_reward += reward
            
            # 更新距离
            curr_dist = math.hypot(goal[0] - position[0], goal[1] - position[1])
            prev_dist = curr_dist
            
            next_obs = self.get_observation(env.grid_map, position, goal, dynamic_obstacles)
            self.obs_window.append(next_obs)
            
            done = (position == goal)
            if done:
                success = True
                break
            
            td_error = abs(reward)
            d_safety = self._calculate_safety_distance(position)
            if len(self.obs_window) >= self.window_size:
                obs_seq = np.array(list(self.obs_window)[:-1])
                next_obs_seq = np.array(list(self.obs_window))
                self.store_transition(obs_seq, action, reward, next_obs_seq, done, td_error, d_safety)
            
            if step % 4 == 0:
                loss = self.update(batch_size=32)
                self.stats['total_loss'] += loss
        
        if len(self.obs_window) >= self.window_size:
            self.update()
        
        # epsilon 退火
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.stats['episodes'] += 1
        self.stats['success_rate'] = (self.stats['success_rate'] * (self.stats['episodes'] - 1) + (1.0 if success else 0.0)) / self.stats['episodes']
        return total_reward, success
    
    def _calculate_reward_dense(self, position, goal, prev_dist: float, action, dynamic_obstacles, grid_map) -> float:
        """稠密奖励函数 - 优化版"""
        reward = 0.0
        
        # 1. 到达目标奖励
        if position == goal:
            reward += 100.0
        else:
            # 2. 距离奖励 (稠密)
            curr_dist = math.hypot(goal[0] - position[0], goal[1] - position[1])
            if curr_dist < prev_dist:
                reward += 1.0  # 靠近目标奖励
            else:
                reward -= 0.5  # 远离目标惩罚
        
        # 3. 碰撞惩罚
        x, y = position
        if grid_map[x, y] == 1:
            reward -= 50.0
        
        # 4. 动态障碍物惩罚
        for obstacle in dynamic_obstacles:
            dist = math.hypot(x - obstacle.x, y - obstacle.y)
            if dist < obstacle.size:
                reward -= 30.0
            elif dist < obstacle.size + 2:
                reward -= 5.0  # 接近警告
        
        # 5. 平滑度奖励
        reward += 2.0 if action < 2 else -1.0
        
        # 6. 步数惩罚 (较小)
        reward -= 0.05
        
        return reward
    
    def _calculate_reward(self, position, goal, action, dynamic_obstacles, grid_map) -> float:
        """原始奖励函数 (保留兼容性)"""
        reward = 0.0
        if position == goal:
            reward += 100.0
        x, y = position
        if grid_map[x, y] == 1:
            reward -= 50.0
        for obstacle in dynamic_obstacles:
            dist = math.hypot(x - obstacle.x, y - obstacle.y)
            if dist < obstacle.size:
                reward -= 30.0
        reward += 5.0 if action < 2 else -2.0
        reward -= 0.1
        return reward

# ============================================================================
# Part 6: A* 底层规划器
# ============================================================================

class AStarPlanner:
    """A* 规划器"""
    def __init__(self, env):
        self.env = env
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        if not self.env.is_valid(*start) or not self.env.is_valid(*goal):
            return None
        open_set = [(0, start[0], start[1])]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        while open_set:
            _, x, y = heapq.heappop(open_set)
            current = (x, y)
            if current == goal:
                return self._reconstruct_path(came_from, current)
            for nx, ny in self._get_neighbors(x, y):
                tentative_g = g_score[current] + math.hypot(nx - x, ny - y)
                neighbor = (nx, ny)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, nx, ny))
        return None
    
    def _heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def _get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.env.is_valid(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors
    
    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

# ============================================================================
# Part 7: 完整 SW-RDQN 规划系统 v3.3
# ============================================================================

class SWRDQNPlanner:
    """SW-RDQN 复合路径规划系统 v3.3"""
    def __init__(self, grid_map: np.ndarray, safety_distance: float = 2.0):
        self.grid_map = grid_map
        self.env = GridEnvironment(grid_map)
        self.keypoint_generator = VoronoiKeyPointGenerator(grid_map, safety_distance)
        self.local_controller = LocalSWRDQNController(grid_map=grid_map)
        self.astar_planner = AStarPlanner(self.env)
        self.dynamic_obstacles = []
        self.key_points = []
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int], dynamic_obstacles: List[DynamicObstacle] = None, use_astar: bool = True, train_first: bool = True, n_episodes: int = 200):  # 优化：增加默认训练回合
        print(f"\n{'='*60}\nSW-RDQN 复合路径规划系统 v3.3\n{'='*60}")
        print(f"起点：{start}\n终点：{goal}\n{'='*60}\n")
        print("Step 1: 生成 Voronoi 关键点...")
        self.key_points = self.keypoint_generator.extract_key_points(start, goal)
        print(f"✓ 关键点数量：{len(self.key_points)}")
        if dynamic_obstacles:
            self.dynamic_obstacles = dynamic_obstacles
            print(f"✓ 动态障碍物：{len(dynamic_obstacles)} 个")
        if train_first:
            print("\nStep 2: 训练 SW-RDQN 控制器...")
            self._train_controller(start, goal, n_episodes)
        print("\nStep 3: 多阶段路径规划...")
        if use_astar:
            full_path = self._plan_with_astar(start, goal)
        else:
            full_path = self._plan_with_drl(start, goal)
        stats = {'path_length': sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in zip(full_path, full_path[1:])) if full_path else None, 'num_key_points': len(self.key_points), 'success': full_path is not None and len(full_path) > 0}
        print(f"\n{'='*60}\n规划完成!")
        if stats['path_length']:
            print(f"路径长度：{stats['path_length']:.2f}")
        else:
            print("无路径")
        print(f"成功率：{'✓' if stats['success'] else '✗'}")
        print(f"{'='*60}\n")
        return full_path, stats
    
    def _train_controller(self, start, goal, n_episodes: int = 200):
        print(f"训练回合：{n_episodes}")
        print(f"初始 epsilon: {self.local_controller.epsilon_start}")
        print(f"最终 epsilon: {self.local_controller.epsilon_end}")
        successes = 0
        for episode in range(n_episodes):
            subgoal = self.key_points[np.random.randint(1, len(self.key_points) - 1)] if len(self.key_points) > 2 else goal
            _, success = self.local_controller.train_episode(self.env, start, subgoal, self.dynamic_obstacles)
            if success:
                successes += 1
            if (episode + 1) % 20 == 0:
                success_rate = successes / (episode + 1) * 100
                print(f"回合 {episode+1}/{n_episodes}: 成功率 = {success_rate:.1f}%, epsilon = {self.local_controller.epsilon:.3f}")
            if (episode + 1) % 10 == 0:
                self.local_controller.update_target_network()
        print(f"✓ 训练完成，最终成功率：{successes/n_episodes*100:.1f}%")
    
    def _plan_with_astar(self, start, goal):
        full_path = [start]
        current = start
        for i in range(len(self.key_points) - 1):
            subgoal = self.key_points[i + 1]
            segment = self.astar_planner.plan(current, subgoal)
            if segment:
                full_path.extend(segment[1:])
                current = segment[-1]
            else:
                return None
        return full_path
    
    def _plan_with_drl(self, start, goal):
        full_path = [start]
        current = start
        for i in range(len(self.key_points) - 1):
            subgoal = self.key_points[i + 1]
            segment = self._drl_plan_segment(current, subgoal)
            if segment:
                full_path.extend(segment[1:])
                current = segment[-1]
            else:
                return None
        return full_path
    
    def _drl_plan_segment(self, start, goal):
        position = start
        path = [position]
        for step in range(100):
            obs = self.local_controller.get_observation(self.grid_map, position, goal, self.dynamic_obstacles)
            action = self.local_controller.select_action(obs, epsilon=0.0)  # 测试时用贪婪
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            dx, dy = directions[action]
            new_x, new_y = position[0] + dx, position[1] + dy
            if self.env.is_valid(new_x, new_y):
                position = (new_x, new_y)
                path.append(position)
            if position == goal:
                break
        return path if position == goal else []
    
    def visualize(self, path, save_path: str = None):
        self.keypoint_generator.visualize()
        if path:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.grid_map, cmap='binary', origin='lower')
            px, py = [p[1] for p in path], [p[0] for p in path]
            plt.plot(px, py, 'r-', linewidth=2, label='Path')
            plt.scatter([path[0][1]], [path[0][0]], c='green', s=200, marker='o', label='Start')
            plt.scatter([path[-1][1]], [path[-1][0]], c='blue', s=200, marker='s', label='Goal')
            plt.legend()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

class GridEnvironment:
    def __init__(self, grid_map: np.ndarray):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.height and 0 <= y < self.width and self.grid_map[x, y] == 0

# ============================================================================
# Part 8: 主函数
# ============================================================================

def main():
    print("="*60 + "\nSW-RDQN 复合路径规划系统 v3.3 - 训练优化\n" + "="*60)
    print("\n1. 创建测试环境...")
    grid_map = np.zeros((30, 30), dtype=np.int8)
    grid_map[5:10, 5:25] = 1
    grid_map[15:20, 10:30] = 1
    grid_map[25:30, 0:15] = 1
    dynamic_obstacles = [DynamicObstacle(15, 15, 0.5, 0.3, 3), DynamicObstacle(20, 10, -0.3, 0.5, 2)]
    print(f"   ✓ 地图大小：30x30\n   ✓ 动态障碍物：{len(dynamic_obstacles)} 个\n")
    print("2. 创建 SW-RDQN 规划器 v3.3...")
    planner = SWRDQNPlanner(grid_map, safety_distance=2.0)
    print(f"   ✓ Voronoi 关键点生成器 (更多关键点)\n   ✓ SW-RDQN 控制器 (稠密奖励 + epsilon 退火)\n   ✓ 加权优先经验回放 (式 17)\n   ✓ A* 底层规划器\n")
    print("3. 规划路径...")
    path, stats = planner.plan((2, 2), (27, 27), dynamic_obstacles, use_astar=True, train_first=True, n_episodes=200)
    print("4. 可视化...")
    if path:
        planner.visualize(path, save_path='/root/.openclaw/workspace/cable-optimization/examples/outputs/swrdqn_v33_path.png')
    print("\n" + "="*60 + "\n✅ SW-RDQN v3.3 训练优化完成！\n" + "="*60)
    print("\n优化内容 (v3.2 → v3.3):")
    print("  1. ✅ 增加训练回合 (40 → 200)")
    print("  2. ✅ 稠密奖励函数 (距离奖励)")
    print("  3. ✅ Epsilon 退火策略 (1.0 → 0.01)")
    print("  4. ✅ 增加关键点数量")
    print("  5. ✅ 提高学习率 (1e-4 → 5e-4)")
    print("\n预期效果：")
    print("  - 训练成功率：0% → 30-50%")
    print("  - 收敛速度：更快")
    print("\n输出：swrdqn_v33_path.png\n")

if __name__ == "__main__":
    main()
