# Sun et al. (2026) SW-RDQN 完整复现计划

**论文**: Deep reinforcement learning-based composite path planning with key path points  
**作者**: Sun, X., et al.  
**期刊**: IEEE Transactions on Robotics, 2026

---

## 📊 SW-RDQN 架构解析

### 整体架构 (Figure 2)

```
┌─────────────────────────────────────────┐
│  全局规划层 (Global Planner)             │
│  - Voronoi 图关键点提取 (Algorithm 1)    │
│  - 安全距离剪枝                          │
│  - 输出：关键点序列 KP = {kp_0, ..., kp_n}│
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  局部控制层 (Local SW-RDQN Controller)   │
│  - Sliding Window 观测 (11x11 栅格)      │
│  - CNN 编码器 (3 层卷积)                 │
│  - GRU 时序记忆 (2 层，hidden=256)       │
│  - Dueling DQN 头部                      │
│  - 输出：8 方向移动动作                   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  A* 底层规划器 (可选)                     │
│  - 关键点间精确路径                      │
│  - 动态重规划触发                        │
└─────────────────────────────────────────┘
```

---

## 🔑 核心组件详解

### 1. Voronoi 关键点生成 (Algorithm 1)

**输入**: 
- 栅格地图 M
- 起点 S，终点 G
- 安全距离 d_safe

**输出**: 
- 关键点序列 KP = {S, kp_1, kp_2, ..., kp_n, G}

**步骤**:
1. 计算障碍物集合 O = {o_1, o_2, ..., o_m}
2. 生成 Voronoi 图 Vor(O)
3. 提取 Voronoi 边 E = {e_1, e_2, ..., e_k}
4. 安全距离剪枝:
   - 对每条边 e_i，计算其上各点到最近障碍物的距离
   - 如果存在点 p 使得 dist(p, O) < d_safe，移除 e_i
5. 连接 S 和 G 到最近的骨架点
6. 简化关键点 (移除共线点)

**关键公式**:
```
dist(p, O) = min_{o∈O} ||p - o||_2
```

---

### 2. SW-RDQN 网络架构 (Figure 3)

#### 2.1 输入表示

**观测窗口**: 11x11 局部栅格图

**通道**:
- 通道 0: 静态障碍物 (0=自由，1=障碍)
- 通道 1: 动态障碍物当前位置
- 通道 2: 动态障碍物预测位置 (t+1)
- 通道 3: 动态障碍物预测位置 (t+2)
- 通道 4: 子目标方向 (相对位置)

**Sliding Window**: 保存最近 4 帧观测

**输入张量**: (batch, window_size=4, channels=5, height=11, width=11)

#### 2.2 CNN 编码器

```
Conv1: 5x5 kernel, 32 filters, stride=2 → ReLU → BatchNorm
       输出：(batch, 4, 32, 5, 5)

Conv2: 3x3 kernel, 64 filters, stride=1 → ReLU → BatchNorm
       输出：(batch, 4, 64, 3, 3)

Conv3: 3x3 kernel, 128 filters, stride=1 → ReLU → BatchNorm
       输出：(batch, 4, 128, 1, 1)

Flatten: (batch, 4, 128)
```

#### 2.3 GRU 时序处理

```
GRU: input_size=128, hidden_size=256, num_layers=2, batch_first=True
     输出：(batch, 4, 256)
     
取最后一步：(batch, 256)
```

#### 2.4 Dueling DQN 头部

```
价值流 V(s):
  Linear(256 → 128) → ReLU
  Linear(128 → 1)

优势流 A(s, a):
  Linear(256 → 128) → ReLU
  Linear(128 → n_actions=8)

Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
```

**输出**: (batch, 8) - 8 个动作的 Q 值

---

### 3. 奖励函数 (式 12-16)

**总奖励**:
```
r_total = r_goal + r_obstacle + r_dynamic + r_smooth + r_step
```

**各项定义**:

1. **目标奖励**:
```
r_goal = +100,  如果到达子目标
r_goal = 0,     否则
```

2. **静态障碍物惩罚**:
```
r_obstacle = -50,  如果碰撞静态障碍
r_obstacle = -10,  如果距离障碍 < 2 栅格
r_obstacle = 0,    否则
```

3. **动态障碍物惩罚**:
```
r_dynamic = -30,  如果预测碰撞 (t+δt)
r_dynamic = 0,    否则
```

4. **平滑度奖励**:
```
r_smooth = +5,  如果动作与上一步相同 (直线移动)
r_smooth = -2,  如果转向 (45 度)
r_smooth = -5,  如果急转 (90 度)
```

5. **步数惩罚**:
```
r_step = -0.1  每步
```

---

### 4. 加权优先经验回放 (式 17-19)

**优先级定义**:
```
priority_i = |TD-error_i| + ε_priority
           = |r_i + γ*max_a' Q(s', a') - Q(s, a)| + ε_priority
```

**加权采样**:
```
P(i) = log(1 + priority_i) / Σ_j log(1 + priority_j)
```

**重要性采样权重**:
```
w_i = (N * P(i))^-β / max_j(w_j)
```

**β退火**:
```
β = β_0 + (1 - β_0) * (step / max_steps)
```

---

### 5. 多阶段训练 (Algorithm 2)

**阶段划分**:
```
Stage 0: S → KP_1
Stage 1: KP_1 → KP_2
...
Stage n: KP_n → G
```

**训练流程**:
1. 对每个阶段 k = 0, ..., n:
   - 设置子目标 goal = KP_{k+1}
   - 训练局部控制器 N_episodes 回合
   - 保存训练好的网络参数
2. 部署时按阶段切换子目标

---

## 🔧 实现清单

### Part 1: Voronoi 关键点生成 ✅
- [x] 距离场计算
- [x] Voronoi 图生成
- [x] 安全距离剪枝
- [ ] 起点/终点连接到骨架
- [ ] 关键点简化

### Part 2: SW-RDQN 网络
- [ ] CNN 编码器 (3 层卷积+BN)
- [ ] GRU 时序处理 (2 层)
- [ ] Dueling DQN 头部
- [ ] Sliding Window 缓存

### Part 3: 训练系统
- [ ] 加权优先经验回放
- [ ] 重要性采样
- [ ] β退火
- [ ] 多阶段训练

### Part 4: 奖励函数
- [ ] 目标奖励
- [ ] 障碍物惩罚
- [ ] 动态障碍物惩罚
- [ ] 平滑度奖励
- [ ] 步数惩罚

### Part 5: 集成测试
- [ ] 静态环境测试
- [ ] 动态环境测试
- [ ] 对比实验 (vs A*, vs v1.0)
- [ ] 消融实验

---

## 📅 时间计划

| 时间 | 任务 | 预计代码量 |
|------|------|-----------|
| 10:30-11:00 | Voronoi 完善 | 100 行 |
| 11:00-11:45 | SW-RDQN 网络 | 300 行 |
| 11:45-12:00 | 奖励函数 | 100 行 |
| 14:00-14:45 | 经验回放 + 训练 | 200 行 |
| 14:45-15:15 | 集成测试 | 150 行 |
| 15:15-15:30 | 文档 + 提交 | 50 行 |
| **总计** | | **~900 行** |

---

## 📚 关键参考文献

1. **Voronoi 图**: Fortune (1987) - Sweepline 算法
2. **DQN**: Mnih et al. (2015) - Nature
3. **Double DQN**: Van Hasselt et al. (2016) - AAAI
4. **Dueling DQN**: Wang et al. (2016) - ICML
5. **PER**: Schaul et al. (2016) - ICLR
6. **GRU**: Cho et al. (2014) - EMNLP

---

**开始时间**: 10:25  
**目标完成**: 12:00 (第一部分)  
**状态**: 准备就绪！🚀
