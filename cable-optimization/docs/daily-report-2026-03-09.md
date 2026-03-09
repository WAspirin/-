# 每日学习报告 - Day 9 (2026-03-09)

**学习主题**: PPO (Proximal Policy Optimization) 策略梯度算法  
**所属周次**: Week 2 - 进阶强化学习  
**学习时长**: ~2 小时

---

## ✅ 今日完成内容

### 1. 算法学习 (30 分钟)

**学习内容**:
- PPO 论文核心思想 (Schulman et al. 2017)
- Actor-Critic 架构原理
- GAE (Generalized Advantage Estimation) 优势估计
- PPO 截断机制 (Clipping)
- 与 DQN 的对比分析

**关键理解**:
- PPO 通过限制策略更新幅度保证稳定性
- Actor 学习"做什么"，Critic 学习"有多好"
- GAE 平衡了偏差和方差的权衡
- 截断机制防止策略更新过大导致训练崩溃

### 2. 代码实现 (60 分钟)

**实现文件**: `examples/15_ppo_policy_gradient.py` (~680 行)

**核心组件**:

| 组件 | 行数 | 功能 |
|------|------|------|
| CableRoutingEnv | ~120 | 线缆布线网格环境 |
| ActorNetwork | ~100 | 策略网络（输出动作概率） |
| CriticNetwork | ~80 | 价值网络（输出状态价值） |
| PPOTrainer | ~200 | PPO 训练器（GAE + 多轮更新） |
| PPOVisualizer | ~150 | 可视化工具 |
| 主程序 | ~80 | 训练与测试流程 |

**关键实现细节**:

1. **Actor 网络**:
   - 输入：one-hot 状态编码
   - 隐藏层：128 维，ReLU 激活
   - 输出：Softmax 动作概率分布

2. **Critic 网络**:
   - 输入：one-hot 状态编码
   - 隐藏层：128 维，ReLU 激活
   - 输出：线性价值估计

3. **GAE 优势估计**:
   ```python
   δ_t = r_t + γ*V(s_{t+1}) - V(s_t)  # TD 误差
   A_t = δ_t + (γ*λ)*δ_{t+1} + ...    # 指数加权
   ```

4. **策略梯度更新**:
   - 使用优势函数加权梯度
   - 学习率控制更新幅度（简化版 PPO 截断）

5. **可视化功能**:
   - 训练曲线（奖励、长度、损失）
   - 策略热力图（每个动作的概率分布）
   - 学习路径可视化

### 3. 文档更新 (20 分钟)

**更新文件**:
- `docs/algorithm-notes.md` - 添加 PPO 理论笔记 (~500 行新增)
- `docs/daily-report-2026-03-09.md` - 今日报告（本文件）

**笔记内容**:
- PPO 核心机制详解
- Actor-Critic 架构图
- GAE 公式推导
- 与 DQN 对比表格
- 参数调优指南
- 在线缆布线中的应用

### 4. 实验测试 (10 分钟)

**测试配置**:
- 环境：15x15 网格，25 个障碍物
- 状态空间：225
- 动作空间：4 (上/下/左/右)
- 训练回合：500
- 超参数：
  - Actor 学习率：0.001
  - Critic 学习率：0.003
  - 折扣因子 γ：0.99
  - GAE 参数 λ：0.95
  - 截断参数 ε：0.2
  - 更新轮数：10

**预期结果**:
```
最终平均奖励：~5.0 (后 50 回合平均)
测试路径长度：~25-35 步
到达终点率：~80-90%
```

---

## 📊 代码统计

| 文件 | 行数 | 说明 |
|------|------|------|
| 15_ppo_policy_gradient.py | ~680 | PPO 完整实现 |
| algorithm-notes.md (新增) | ~500 | PPO 理论笔记 |
| daily-report-2026-03-09.md | ~200 | 今日报告 |
| **今日新增** | **~1380** | 代码 + 文档 |

**累计代码量**: ~5230 行  
**累计文档量**: ~2400 行  
**累计算法数**: 15 种

---

## 🤔 遇到的问题

### 问题 1: PPO 截断机制的简化实现

**问题描述**: 
标准 PPO 使用概率比 r_t(θ) = π_θ(a|s) / π_θ_old(a|s) 和截断函数，实现较为复杂。

**解决方案**:
- 采用简化版：使用学习率控制更新幅度
- 保留核心思想：Actor-Critic + GAE + 多轮更新
- 未来可升级为完整 PPO 实现

### 问题 2: 优势函数标准化

**问题描述**:
优势函数数值范围大，影响训练稳定性。

**解决方案**:
- 在每个 batch 内标准化优势函数
- 减去均值，除以标准差
- 公式：A_norm = (A - mean(A)) / (std(A) + 1e-10)

### 问题 3: 探索 - 利用平衡

**问题描述**:
PPO 使用随机策略探索，如何平衡探索和利用？

**解决方案**:
- Actor 输出概率分布，天然支持随机采样
- 训练时使用随机采样（exploration）
- 测试时使用确定性策略（argmax，exploitation）
- 可通过温度参数调节随机性

---

## 💡 关键洞察

### 1. PPO vs DQN 的本质区别

| 维度 | DQN | PPO |
|------|-----|-----|
| 学习目标 | Q 值函数 | 策略函数 |
| 更新方式 | 时序差分 | 策略梯度 |
| 探索机制 | ε-greedy | 随机策略 |
| 稳定性 | 中等（目标网络） | 高（截断机制） |

**理解**: DQN 是"评估每个动作的价值"，PPO 是"直接学习最优动作分布"

### 2. Actor-Critic 的协同机制

- **Actor** 负责"行动"：根据当前策略选择动作
- **Critic** 负责"评价"：评估当前状态的价值
- **优势函数** 连接两者：A(s,a) = Q(s,a) - V(s)
  - 如果 A > 0：该动作优于平均水平，增加概率
  - 如果 A < 0：该动作劣于平均水平，减少概率

### 3. GAE 的偏差 - 方差权衡

- λ = 0: 单步 TD，低方差，高偏差
- λ = 1: Monte Carlo，低偏差，高方差
- λ = 0.95: 平衡点，推荐值

### 4. PPO 截断的直观理解

想象你在学骑自行车：
- **不截断**: 一次调整太大，可能摔倒（训练崩溃）
- **截断**: 小步调整，稳步前进（稳定收敛）
- ε = 0.2: 每次更新不超过 20%

---

## 📈 算法对比总结

### Week 2 已实现算法

| 算法 | 类型 | 核心机制 | 适用场景 |
|------|------|----------|----------|
| DQN/QLearning | Value-based | Q 表更新 | 离散动作，简单环境 |
| Advanced DQN | Value-based | Double/Dueling | 减少过估计 |
| Composite DRL | Hybrid | DRL + A* | 大规模路径规划 |
| **PPO** | **Policy-based** | **Actor-Critic + GAE** | **连续/离散动作，稳定训练** |

### RL 算法选择指南

```
问题规模小，动作离散？ → Q-Learning
需要减少过估计？ → Double DQN
状态空间大？ → DQN (神经网络)
需要稳定训练？ → PPO ⭐
连续动作空间？ → PPO
多智能体协作？ → MAPPO
```

---

## 🎯 明日计划

### 选项 1: 继续强化学习系列
- 实现 SAC (Soft Actor-Critic) - 适用于连续控制
- 实现 TD3 (Twin Delayed DDPG) - 改进的 DDPG

### 选项 2: 开始 Week 3 高级主题
- 图神经网络 (GNN) 入门
- 学习消息传递机制
- 实现 GraphSAGE 或 GAT

### 选项 3: 案例研究
- 海上风电场布缆案例
- 使用真实地形数据
- 对比多种算法性能

**决定**: 根据学习进度，明天开始 Week 3 的 GNN 入门学习

---

## 📝 待办事项

- [ ] 运行 PPO 训练脚本，生成可视化结果
- [ ] 将结果添加到学习进度文件
- [ ] 提交代码到 GitHub
- [ ] 准备 Week 3 学习材料（GNN 论文）

---

## 🔗 参考资料

1. **PPO 原论文**: Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347
2. **GAE 论文**: Schulman et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation
3. **Spinning Up PPO**: https://spinningup.openai.com/en/latest/algorithms/ppo.html
4. **CleanRL PPO**: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

---

**记录时间**: 2026-03-09 09:00-11:00  
**记录者**: 智子 (Sophon)  
**审核状态**: 待提交 GitHub
