import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import copy
import math
import random
from typing import List, Tuple, Optional, Dict


# ============================================================================
# 动态障碍物类
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
        for dx in range(-self.size // 2, self.size // 2 + 1):
            for dy in range(-self.size // 2, self.size // 2 + 1):
                x, y = int(self.x + dx), int(self.y + dy)
                if 0 <= x < height and 0 <= y < width:
                    mask[x, y] = True
        return mask


# ============================================================================
# Part 3: SW-RDQN 网络 (严格对齐文献图 4)
# ============================================================================

class SWRDQNNetwork(nn.Module):
    """严格对齐文献图 4：CNN + MLP 特征融合 -> LSTM -> Dueling DQN"""

    def __init__(self, obs_height: int = 11, obs_width: int = 11,
                 img_channels: int = 5, window_size: int = 4,
                 n_actions: int = 8, hidden_size: int = 512):
        super().__init__()
        self.n_actions = n_actions
        self.img_channels = img_channels
        self.obs_height = obs_height
        self.obs_width = obs_width

        # 1. CNN for image observation
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=5, stride=2, padding=2),
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

        # 2. MLP for target feature [dist, theta]
        self.target_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )

        # 3. LSTM over time steps
        self.lstm = nn.LSTM(input_size=128 + 64, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)

        # 4. Dueling DQN streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions)
        )

    def forward(self, img_seq: torch.Tensor, target_seq: torch.Tensor):
        """
        Args:
            img_seq: [B, T, C, H, W]
            target_seq: [B, T, 2]
        Returns:
            q_values: [B, n_actions]
        """
        B, T = img_seq.shape[:2]

        # Vectorized feature extraction
        img_flat = img_seq.view(B * T, self.img_channels, self.obs_height, self.obs_width)
        cnn_feat = self.cnn(img_flat).view(B * T, -1)  # [B*T, 128]

        tgt_flat = target_seq.view(B * T, 2)
        mlp_feat = self.target_mlp(tgt_flat)  # [B*T, 64]

        fused = torch.cat([cnn_feat, mlp_feat], dim=1)  # [B*T, 192]
        fused_seq = fused.view(B, T, -1)  # [B, T, 192]

        lstm_out, _ = self.lstm(fused_seq)
        last_hidden = lstm_out[:, -1, :]  # [B, hidden_size]

        value = self.value_stream(last_hidden)  # [B, 1]
        advantage = self.advantage_stream(last_hidden)  # [B, n_actions]
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# ============================================================================
# Part 4: n-step 加权优先经验回放 (对齐文献公式 19)
# ============================================================================

class NStepPrioritizedReplayBuffer:
    """严格对齐文献 3.3 节公式 (19)：基于 Reward 和 TD Error 对数归一化的加权优先回放"""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, w_reward: float = 0.15, w_td: float = 0.85):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.w_reward = w_reward
        self.w_td = w_td

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def _calc_paper_priority(self, td_error: float, reward: float) -> float:
        # 严格按照公式 (19): r + |min(r, 0)| + 1
        adjusted_reward = reward + abs(min(reward, 0.0)) + 1.0
        p_reward = self.w_reward * np.log(adjusted_reward)
        p_td = self.w_td * np.log(abs(td_error) + 1.0)
        return p_reward + p_td

    def add(self, transition: Dict, td_error: float, reward: float):
        priority = self._calc_paper_priority(td_error, reward)
        self.buffer.append(transition)
        self.priorities.append(priority)  # 不再使用 max_prio

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            indices = list(range(len(self.buffer)))
            batch = [self.buffer[i] for i in indices]
            weights = torch.ones(len(batch))
            return batch, weights, indices

        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化到 [0, 1]
        self.beta = min(1.0, self.beta + self.beta_increment)
        return batch, torch.FloatTensor(weights), indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray, rewards: np.ndarray):
        for idx, td_err, rew in zip(indices, td_errors, rewards):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = self._calc_paper_priority(td_err, rew)

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Part 5: 局部 SW-RDQN 控制器 (对齐文献)
# ============================================================================

class LocalSWRDQNController:
    """局部 SW-RDQN 控制器 - 论文公式复现版"""

    def __init__(self, obs_size: int = 11, img_channels: int = 5, window_size: int = 4,
                 n_actions: int = 8, learning_rate: float = 1e-3, gamma: float = 0.99,
                 grid_map: np.ndarray = None, key_path_points: List[Tuple[int, int]] = None):
        self.obs_size = obs_size
        self.img_channels = img_channels
        self.window_size = window_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.grid_map = grid_map
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.network = SWRDQNNetwork(obs_size, obs_size, img_channels, window_size, n_actions).to(self.device)
        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_network.eval()

        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20,
        )

        # 回放池调整为对数优先级版本
        self.replay_buffer = NStepPrioritizedReplayBuffer(capacity=10000)

        # 观测窗口改为存放 (img_obs, target_obs) 元组
        self.obs_window = deque(maxlen=window_size)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        self.best_success_rate = 0.0
        self.patience_counter = 0
        self.max_patience = 50

        self.stats = {'episodes': 0, 'success_rate': 0.0, 'total_loss': 0.0}

        self.key_path_points = key_path_points if key_path_points is not None else []
        self.current_stage = 0

    def get_observation(self, grid_map: np.ndarray, position: Tuple[int, int], goal: Tuple[int, int],
                        dynamic_obstacles: List[DynamicObstacle]) -> Tuple[np.ndarray, np.ndarray]:
        """严格分离图像观测 (env) 和目标特征 (fet)"""
        x, y = position
        half_size = self.obs_size // 2

        # 1. 图像特征 (5通道: 静态, 动态当前, 预测未来3步)
        img_obs = np.zeros((5, self.obs_size, self.obs_size), dtype=np.float32)
        for i in range(self.obs_size):
            for j in range(self.obs_size):
                gx = int(x - half_size + i)
                gy = int(y - half_size + j)

                if 0 <= gx < grid_map.shape[0] and 0 <= gy < grid_map.shape[1]:
                    img_obs[0, i, j] = grid_map[gx, gy]
                    for obstacle in dynamic_obstacles:
                        if obstacle.get_current_mask(grid_map.shape[0], grid_map.shape[1])[gx, gy]:
                            img_obs[1, i, j] = 1.0
                        for t, (px, py) in enumerate(obstacle.predict_future_positions(3)):
                            if int(px) == gx and int(py) == gy:
                                img_obs[2 + t, i, j] = 1.0
                else:
                    img_obs[0, i, j] = 1.0

        # 2. 目标特征向量 [d, theta]
        dist = math.hypot(goal[0] - x, goal[1] - y)
        theta = math.atan2(goal[1] - y, goal[0] - x)
        target_obs = np.array([dist, theta], dtype=np.float32)

        return img_obs, target_obs

    def _calculate_paper_reward(self, position, prev_position, goal, action, dynamic_obstacles, grid_map) -> float:
        """严格对齐文献公式 12-16 的奖励函数"""
        r_a, r_b, r_c, r_e = 0.6, -0.4, -10.0, -0.2
        r1, r2, r3, r4 = -2.0, -1.5, -0.2, -0.01
        alpha = 0.5

        x, y = position
        px, py = prev_position
        gx, gy = goal

        dt = math.hypot(gx - x, gy - y)
        dt_1 = math.hypot(gx - px, gy - py)

        # 1. R_goal (公式 12)
        if dt < 0.5:
            R_goal = abs(r_c)  # 到达目标
        elif dt < dt_1:
            R_goal = r_a
        else:
            R_goal = r_b

        # 2. R_align (公式 13)
        dx, dy = x - px, y - py
        denom = math.hypot(dx, dy) * math.hypot(gx - px, gy - py)
        if denom > 0:
            R_align = alpha * (dx * (gx - px) + dy * (gy - py)) / denom
        else:
            R_align = 0.0

        # 3. R_avoid & R_pre (公式 14, 15)
        R_avoid = 0.0
        R_pre = 0.0

        if grid_map[int(x), int(y)] == 1:
            R_avoid += -10.0

        for obs in dynamic_obstacles:
            dist_obs = math.hypot(x - obs.x, y - obs.y)
            if dist_obs == 0:
                R_avoid += -10.0
            elif 0 < dist_obs <= 1.0:
                R_avoid += r1
            elif 1.0 < dist_obs <= 2.0:
                R_avoid += r2
            elif 2.0 < dist_obs <= 3.0:
                R_avoid += r3
            else:
                R_avoid += r4

            for future_pt in obs.predict_future_positions(3):
                dis_to_future = math.hypot(x - future_pt[0], y - future_pt[1])
                if dis_to_future < 2.0:
                    R_pre += -1.25 * dis_to_future

        # 4. R_step
        R_step = r_e

        # 公式 16: 总奖励
        R_round = R_goal + R_align + R_avoid + R_pre + R_step
        return R_round

    def select_action(self, img_seq: np.ndarray, target_seq: np.ndarray, epsilon: float = None) -> int:
        """选择动作，支持双输入"""
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            img_tensor = torch.FloatTensor(img_seq).unsqueeze(0).to(self.device)
            target_tensor = torch.FloatTensor(target_seq).unsqueeze(0).to(self.device)
            q_values, _ = self.network(img_tensor, target_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, img_seq, target_seq, action, reward, next_img_seq, next_target_seq, done,
                         td_error: float):
        self.replay_buffer.add(
            {'img_state': img_seq, 'target_state': target_seq, 'action': action,
             'reward': reward, 'next_img_state': next_img_seq, 'next_target_state': next_target_seq, 'done': done},
            td_error, reward)

    def update(self, batch_size: int = 32) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch, weights, indices = self.replay_buffer.sample(batch_size)

        img_states = torch.FloatTensor(np.array([exp['img_state'] for exp in batch])).to(self.device)
        target_states = torch.FloatTensor(np.array([exp['target_state'] for exp in batch])).to(self.device)

        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)

        next_img_states = torch.FloatTensor(np.array([exp['next_img_state'] for exp in batch])).to(self.device)
        next_target_states = torch.FloatTensor(np.array([exp['next_target_state'] for exp in batch])).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)

        q_values = self.network(img_states, target_states)[0].gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_img_states, next_target_states)[0].max(dim=1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        td_errors = torch.abs(q_values - targets)
        loss = (td_errors * weights.to(self.device)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step(loss)

        # 使用文献公式 19 要求的 td_error 和 reward 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy(), rewards.cpu().numpy())

        return loss.item()

    def update_target_network(self):
        self.target_network = copy.deepcopy(self.network)

    def train_episode(self, env, start: Tuple[int, int], goal: Tuple[int, int],
                      dynamic_obstacles: List[DynamicObstacle], max_steps: int = 200) -> Tuple[float, bool]:
        position = start
        total_reward = 0.0
        success = False

        # 如果有关键路径点，则使用第一个作为初始目标
        if self.key_path_points:
            current_goal = self.key_path_points[self.current_stage]
        else:
            current_goal = goal

        img_obs, target_obs = self.get_observation(env.grid_map, position, current_goal, dynamic_obstacles)
        self.obs_window.append((img_obs, target_obs))

        prev_position = position

        for step in range(max_steps):
            if len(self.obs_window) < self.window_size:
                action = random.randint(0, self.n_actions - 1)
            else:
                img_seq = np.array([o[0] for o in self.obs_window])
                target_seq = np.array([o[1] for o in self.obs_window])
                action = self.select_action(img_seq, target_seq)

            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            dx, dy = directions[action]
            new_x, new_y = position[0] + dx, position[1] + dy
            if env.is_valid(new_x, new_y):
                prev_position = position
                position = (new_x, new_y)

            for obstacle in dynamic_obstacles:
                obstacle.update()

            reward = self._calculate_paper_reward(position, prev_position, current_goal, action, dynamic_obstacles,
                                                  env.grid_map)

            if position == current_goal:
                reward += 100.0  # 到达当前子目标的额外奖励

                # 切换到下一个子目标
                if self.current_stage + 1 < len(self.key_path_points):
                    self.current_stage += 1
                    current_goal = self.key_path_points[self.current_stage]
                    # 更新观测值为目标位置变化后的值
                    img_obs, target_obs = self.get_observation(env.grid_map, position, current_goal, dynamic_obstacles)
                    self.obs_window[-1] = (img_obs, target_obs)  # 更新最后一个观测窗口为新目标
                else:
                    success = True  # 所有子目标均已完成

            total_reward += reward

            next_img_obs, next_target_obs = self.get_observation(env.grid_map, position, current_goal,
                                                                 dynamic_obstacles)
            self.obs_window.append((next_img_obs, next_target_obs))

            done = (position == current_goal or success)
            if done and not success:
                break

            td_error = abs(reward)  # 初始预估TD误差

            if len(self.obs_window) >= self.window_size:
                prev_window = list(self.obs_window)[:-1]
                img_seq = np.array([o[0] for o in prev_window])
                target_seq = np.array([o[1] for o in prev_window])

                curr_window = list(self.obs_window)
                next_img_seq = np.array([o[0] for o in curr_window])
                next_target_seq = np.array([o[1] for o in curr_window])

                self.store_transition(img_seq, target_seq, action, reward, next_img_seq, next_target_seq, done,
                                      td_error)

            if step % 4 == 0:
                loss = self.update(batch_size=32)
                self.stats['total_loss'] += loss

        if len(self.obs_window) >= self.window_size:
            self.update()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if success:
            if self.stats['success_rate'] > self.best_success_rate:
                self.best_success_rate = self.stats['success_rate']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

        self.stats['episodes'] += 1
        self.stats['success_rate'] = (self.stats['success_rate'] * (self.stats['episodes'] - 1) + (
            1.0 if success else 0.0)) / self.stats['episodes']

        return total_reward, success

    def should_early_stop(self) -> bool:
        return self.patience_counter >= self.max_patience

# ============================================================================
# 测试脚本: 小规模案例验证
# ============================================================================
