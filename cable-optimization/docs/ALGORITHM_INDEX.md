# 算法索引目录

> 线缆布线优化算法库 - 完整算法参考指南

_最后更新：2026-03-15 | 算法总数：22 种 | 代码行数：~10000 | 文档行数：~20000_

---

## 📑 快速导航

- [按类别浏览](#-按类别浏览)
- [按性能选择](#-按性能选择)
- [按场景选择](#-按场景选择)
- [文件位置索引](#-文件位置索引)

---

## 📚 按类别浏览

### 1️⃣ 数学规划法 (1 种)

| 算法 | 文件 | 行数 | 核心思想 | 状态 |
|------|------|------|---------|------|
| MILP | `01_milp_basic.py` | ~180 | 混合整数线性规划，保证全局最优 | ✅ |

**特点**: 理论保证最强，但计算复杂度高，适合小规模问题 (<50 节点)

**使用场景**: 
- 需要精确最优解
- 问题规模较小
- 有足够计算时间

---

### 2️⃣ 精确算法 (4 种)

| 算法 | 文件 | 行数 | 核心思想 | 状态 |
|------|------|------|---------|------|
| Dijkstra | `02_dijkstra.py` | ~170 | 贪心策略，单源最短路径 | ✅ |
| A* Search | `06_astar.py` | ~450 | Dijkstra + 启发式函数 | ✅ |
| Prim MST | `07_minimum_spanning_tree.py` | ~450 | 贪心构建最小生成树 | ✅ |
| Kruskal MST | `07_minimum_spanning_tree.py` | ~450 | 并查集 + 边排序 | ✅ |

**特点**: 速度快，有理论保证，但需要完整图结构

**使用场景**:
- 已知完整网络拓扑
- 需要快速获得可行解
- 作为其他算法的初始解

---

### 3️⃣ 元启发式算法 (7 种)

| 算法 | 文件 | 行数 | 核心思想 | 状态 |
|------|------|------|---------|------|
| 遗传算法 GA | `03_genetic_algorithm.py` | ~240 | 生物进化模拟（选择/交叉/变异） | ✅ |
| 粒子群 PSO | `04_pso.py` | ~300 | 鸟群觅食模拟（速度更新） | ✅ |
| 模拟退火 SA | `05_simulated_annealing.py` | ~380 | 金属退火模拟（Metropolis 准则） | ✅ |
| 变邻域 VNS | `08_variable_neighborhood_search.py` | ~450 | 系统性切换邻域结构 | ✅ |
| 禁忌搜索 TS | `09_tabu_search.py` | ~480 | 禁忌表避免循环 | ✅ |
| 蚁群 ACO | `10_ant_colony_optimization.py` | ~450 | 信息素正反馈 | ✅ |
| 膜算法 Memetic | `17_memetic_algorithm.py` | ~600 | GA + 局部搜索混合 | ✅ |

**特点**: 全局搜索能力强，适合大规模问题，无最优保证但实用性好

**性能对比** (30 节点测试):
| 算法 | 成本 | 时间 (s) | 推荐度 |
|------|------|---------|--------|
| VNS | ~350 | 1.6 | ⭐⭐⭐⭐⭐ |
| TS | ~350 | 0.4 | ⭐⭐⭐⭐⭐ |
| SA | ~353 | 13.0 | ⭐⭐⭐ |
| ACO | ~373 | 2.1 | ⭐⭐⭐⭐ |
| GA | ~380 | 2.5 | ⭐⭐⭐⭐ |

---

### 4️⃣ 强化学习 (4 种)

| 算法 | 文件 | 行数 | 核心思想 | 状态 |
|------|------|------|---------|------|
| Q-Learning | `12_dqn_reinforcement_learning.py` | ~450 | 时序差分学习，Q 表更新 | ✅ |
| 进阶 DQN | `13_advanced_dqn.py` | ~480 | 经验回放 + 目标网络 | ✅ |
| PPO | `15_ppo_policy_gradient.py` | ~680 | 近端策略优化（Actor-Critic） | ✅ |
| SW-RDQN | `20_swr_dqn_paper_implementation.py` | ~680 | 滑动窗口 + Double DQN | ✅ |

**特点**: 从零开始学习，适应动态环境，训练成本高但泛化能力强

**使用场景**:
- 环境动态变化
- 需要在线学习
- 有充足训练时间

---

### 5️⃣ 图神经网络 (1 种)

| 算法 | 文件 | 行数 | 核心思想 | 状态 |
|------|------|------|---------|------|
| GNN | `16_gnn_graph_neural_network.py` | ~650 | 消息传递 + 图卷积/注意力 | ✅ |

**特点**: 直接处理图结构，无需手工特征工程

**使用场景**:
- 图结构复杂
- 需要学习节点/边表示
- 可与其他方法混合使用

---

### 6️⃣ 多目标优化 (1 种)

| 算法 | 文件 | 行数 | 核心思想 | 状态 |
|------|------|------|---------|------|
| NSGA-II | `18_multiobjective_nsga2.py` | ~680 | 非支配排序 + 拥挤度 + 精英保留 | ✅ |

**特点**: 同时优化多个目标，输出 Pareto 前沿

**使用场景**:
- 成本 vs 可靠性权衡
- 多目标决策
- 需要多个候选解

---

### 7️⃣ 大规模问题求解 (1 种)

| 算法 | 文件 | 行数 | 核心思想 | 状态 |
|------|------|------|---------|------|
| 分解算法 | `19_large_scale_decomposition.py` | ~500 | K-Means 聚类 + 分治优化 | ✅ |

**特点**: 将大问题分解为小问题，加速比可达 1000-10000 倍

**使用场景**:
- 节点数 > 100
- 计算资源有限
- 可接受近似解

---

### 8️⃣ 其他算法 (3 种)

| 算法 | 文件 | 行数 | 核心思想 | 状态 |
|------|------|------|---------|------|
| 复合 DRL | `14_composite_drl_planner_v2.py` | ~700 | DRL + 启发式混合 | ✅ |
| Voronoi | `Voronoi_optimized.py` | ~450 | Voronoi 图划分空间 | ✅ |
| 算法对比 | `11_algorithm_comparison.py` | ~550 | 统一测试框架 | ✅ |

---

## 🏆 按性能选择

### 追求最优解质量
1. **Dijkstra/A*** - 精确算法，理论最优（需完整图）
2. **VNS/TS** - 元启发式中最佳（~350 成本）
3. **Memetic** - GA+LS 混合，质量更高

### 追求计算速度
1. **Dijkstra** - <1ms
2. **Prim/Kruskal** - <10ms
3. **TS** - ~0.4s
4. **VNS** - ~1.6s

### 大规模问题 (>100 节点)
1. **分解算法** - 分治策略，加速比 1000x+
2. **VNS/TS** - 元启发式可扩展性好
3. **GA/PSO** - 群体智能，并行性好

### 动态/未知环境
1. **Q-Learning/DQN** - 在线学习
2. **PPO** - 策略梯度，稳定性好
3. **GNN** - 图结构学习

### 多目标决策
1. **NSGA-II** - Pareto 前沿，知情选择

---

## 🎯 按场景选择

### 场景 1: 园区网络布线 (50-100 节点)
**推荐**: VNS 或 TS  
**理由**: 解质量好，计算时间可接受，实现简单

### 场景 2: 城市电网规划 (200+ 节点)
**推荐**: 分解算法 + VNS  
**理由**: 分治处理大规模，簇内用 VNS 优化

### 场景 3: 实时路径规划 (动态障碍物)
**推荐**: A* 或 DQN  
**理由**: A* 快速重规划，DQN 可学习动态策略

### 场景 4: 多约束优化 (成本 + 可靠性 + 美观)
**推荐**: NSGA-II  
**理由**: 多目标 Pareto 优化，提供多个候选解

### 场景 5: 科研实验对比
**推荐**: 运行 `11_algorithm_comparison.py`  
**理由**: 统一测试框架，公平对比所有算法

---

## 📁 文件位置索引

### examples/ 目录
```
examples/
├── 01_milp_basic.py                      # MILP
├── 02_dijkstra.py                        # Dijkstra
├── 03_genetic_algorithm.py               # GA
├── 04_pso.py                             # PSO
├── 05_simulated_annealing.py             # SA
├── 06_astar.py                           # A*
├── 07_minimum_spanning_tree.py           # MST (Prim+Kruskal)
├── 08_variable_neighborhood_search.py    # VNS
├── 09_tabu_search.py                     # TS
├── 10_ant_colony_optimization.py         # ACO
├── 11_algorithm_comparison.py            # 对比实验
├── 12_dqn_reinforcement_learning.py      # Q-Learning
├── 13_advanced_dqn.py                    # 进阶 DQN
├── 14_composite_drl_planner_v2.py        # 复合 DRL
├── 15_ppo_policy_gradient.py             # PPO
├── 16_gnn_graph_neural_network.py        # GNN
├── 17_memetic_algorithm.py               # Memetic
├── 18_multiobjective_nsga2.py            # NSGA-II
├── 19_large_scale_decomposition.py       # 分解算法
├── 20_swr_dqn_paper_implementation.py    # SW-RDQN
├── 20_swr_dqn_paper_implementation_fixed.py
├── SW_RDQN.py                            # SW-RDQN 原始
├── SW_RDQN_optimized.py                  # SW-RDQN 优化
├── Voronoi.py                            # Voronoi
└── Voronoi_optimized.py                  # Voronoi 优化
```

### docs/ 目录
```
docs/
├── ALGORITHM_INDEX.md                    # 本文件 - 算法索引
├── algorithm-notes.md                    # 详细算法笔记
├── technical-report.md                   # 技术报告
├── learning-progress.md                  # 学习进度
├── daily-task.md                         # 每日任务说明
├── week1-algorithm-comparison.md         # Week 1 总结
└── daily-report-2026-03-*.md             # 每日报告 (15 篇)
```

---

## 📊 项目统计

| 指标 | 数值 |
|------|------|
| 算法总数 | 22 种 |
| 代码文件 | 24 个 |
| 代码行数 | ~10,000 |
| 文档行数 | ~20,000 |
| 学习天数 | 15 天 |
| 周次完成 | 2.5/3 周 |

---

## 🔗 相关链接

- [项目 README](../README.md)
- [算法笔记](algorithm-notes.md)
- [技术报告](technical-report.md)
- [学习进度](learning-progress.md)
- [GitHub 仓库](https://github.com/WAspirin/-)

---

**维护者**: 智子 (Sophon) for WonderXi  
**最后更新**: 2026-03-15  
**状态**: ✅ 持续维护中
