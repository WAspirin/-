# Week 2 Day 10 - 进阶 DQN 变体与 GNN 入门

_线缆布线优化中的深度强化学习进阶_

---

## 📚 一、Double DQN (2015)

### 核心问题
标准 DQN 存在 **Q 值高估问题** (overestimation bias)：
- 原因：max 操作倾向于选择被高估的 Q 值
- 公式：`y = r + γ * max_a' Q(s', a')`
- 影响：导致策略次优，训练不稳定

### Double DQN 解法
**解耦动作选择与价值评估**：
```python
# 标准 DQN
target = reward + gamma * target_network.max_q(next_state)

# Double DQN
action = online_network.select_action(next_state)  # 用在线网络选动作
target = reward + gamma * target_network.q(next_state, action)  # 用目标网络评估
```

### 关键代码实现
```python
class DoubleDQN:
    def compute_target(self, states, actions, rewards, next_states, dones):
        # 用在线网络选择动作
        next_actions = self.online_network(states).argmax(dim=1)
        
        # 用目标网络评估价值
        next_q_values = self.target_network(states).gather(1, next_actions)
        
        # TD target
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        return targets
```

### 在线缆布线中的应用
- **状态**: 当前布线位置 + 已布线段 + 约束状态
- **动作**: 下一步移动方向 (上/下/左/右/转弯)
- **优势**: 更准确的 Q 值估计 → 更稳定的路径优化策略

---

## 📚 二、Dueling DQN (2016)

### 核心思想
**分离状态价值与动作优势**：
- 有些状态本身就好 (靠近目标)，与动作无关
- 有些动作在某些状态下更好 (优势)

### 网络架构
```
                    ┌─→ Value V(s) ──┐
State s ─→ CNN/FC ──┤                ├─→ Q(s,a)
                    └─→ Advantage A(s,a) ──┘

Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
```

### 关键代码实现
```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(...)  # 共享特征提取
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 单值输出 V(s)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # 每动作优势 A(s,a)
        )
    
    def forward(self, x):
        features = self.shared(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 聚合：Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
```

### 在线缆布线中的应用
- **Value 流**: 评估当前位置的好坏 (距离目标多远，是否靠近障碍)
- **Advantage 流**: 评估各动作的相对优势 (哪个方向更优)
- **优势**: 更快收敛，更好泛化

---

## 📚 三、图神经网络 (GNN) 基础

### 为什么需要 GNN？
线缆布线问题本质是**图结构问题**：
- 节点：布线点、连接器、障碍角点
- 边：可能的布线路径
- 标准 CNN/RNN 无法直接处理图结构

### GNN 核心思想
**消息传递机制** (Message Passing)：
```
每个节点聚合邻居信息 → 更新自身表示 → 传递更新
```

### 图卷积网络 (GCN)
```python
# 简化的 GCN 层
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj_matrix):
        # x: 节点特征 [N, F]
        # adj_matrix: 邻接矩阵 [N, N]
        
        # 消息传递：聚合邻居特征
        neighbor_agg = torch.matmul(adj_matrix, x)
        
        # 更新节点表示
        output = self.linear(neighbor_agg)
        return torch.relu(output)
```

### 图注意力网络 (GAT)
```python
# 注意力机制加权聚合
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(in_features, num_heads)
        
    def forward(self, x, adj_matrix):
        # 计算节点间注意力权重
        attn_output, _ = self.attention(x, x, x, attn_mask=adj_matrix)
        return attn_output
```

---

## 📚 四、GNN 在线缆布线中的应用

### 问题建模
```
图 G = (V, E)
- V: 所有可能的布线点 (网格点 + 连接器位置)
- E: 相邻点之间的连接 (考虑障碍约束)
- 节点特征：位置坐标、是否障碍、是否已布线、到目标距离
- 边特征：距离、转弯次数、是否穿过障碍
```

### GNN + RL 混合架构
```
状态编码：
  布线图 → GNN → 图嵌入 → RL 策略网络 → 动作选择

训练流程：
  1. GNN 学习图结构表示
  2. RL 学习基于图嵌入的决策策略
  3. 端到端联合优化
```

### 代码框架
```python
class GNN_RL_CableRouter:
    def __init__(self):
        self.gnn = GCN(num_layers=3, hidden_dim=128)
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, graph):
        # graph: (node_features, adj_matrix)
        node_embeds = self.gnn(graph.node_features, graph.adj_matrix)
        
        # 全局图嵌入 (所有节点平均)
        graph_embed = node_embeds.mean(dim=0)
        
        # 策略输出
        action_probs = self.policy_net(graph_embed)
        return action_probs
```

### 优势
- **结构感知**: 直接处理图结构，无需网格化
- **泛化能力**: 学习到的策略可迁移到不同规模的布线问题
- **可解释性**: 注意力权重显示关键路径

---

## 🎯 五、今日实践任务

### 任务 1: 实现 Double DQN (60 分钟)
```bash
cd cable-optimization/week2/
python 10_double_dqn.py
```

**目标**:
- 对比 DQN vs Double DQN 的 Q 值估计差异
- 验证 Double DQN 减少高估的效果

### 任务 2: 实现 Dueling DQN (60 分钟)
```bash
python 11_dueling_dqn.py
```

**目标**:
- 实现 Dueling 网络架构
- 对比标准 DQN 的收敛速度

### 任务 3: GNN 基础实验 (60 分钟)
```bash
python 12_gnn_basic.py
```

**目标**:
- 构建布线问题的图表示
- 实现简单的 GCN 层
- 可视化节点嵌入

---

## 📊 六、预期结果

| 算法 | Q 值高估 | 收敛速度 | 最终性能 |
|------|---------|---------|---------|
| DQN | 高 | 中等 | 基准 |
| Double DQN | 低 | 中等 | +5-10% |
| Dueling DQN | 中 | 快 | +10-15% |
| GNN+RL | 中 | 慢 (训练) | +20-30% (泛化) |

---

## 🔗 七、参考资料

1. **Double DQN**: Hasselt et al. (2015) - "Deep Reinforcement Learning with Double Q-learning"
2. **Dueling DQN**: Wang et al. (2016) - "Dueling Network Architectures for Deep Reinforcement Learning"
3. **GCN**: Kipf & Welling (2017) - "Semi-Supervised Classification with Graph Convolutional Networks"
4. **GAT**: Veličković et al. (2018) - "Graph Attention Networks"

**代码参考**:
- https://github.com/pranz24/pytorch-double-dqn
- https://github.com/KaiyangZhou/pytorch-ddpg-naf (Dueling 实现)
- https://github.com/rusty1s/pytorch_geometric (GNN 库)

---

_创建时间：2026-03-10 18:17_
_智子主动准备 - Week 2 Day 10 学习材料_
