# 线缆布线优化算法库

> WonderXi 的科研项目 - 线缆自动布线优化算法实现与对比

[![Learning Progress](https://img.shields.io/badge/进度-Day%2015%2F21-blue)](docs/learning-progress.md)
[![Algorithms](https://img.shields.io/badge/算法 -22 种-green)](docs/ALGORITHM_INDEX.md)
[![Code](https://img.shields.io/badge/代码 -10000%20行-orange)](examples/)
[![Docs](https://img.shields.io/badge/文档 -20000%20行-lightgrey)](docs/)
[![License](https://img.shields.io/badge/许可-MIT-blue)](LICENSE)

---

## 📚 研究方向

**核心问题**: 如何在复杂约束下优化线缆布线路径，最小化成本同时满足可靠性、美观等要求

**主要方法**:
- ✅ 数学规划法 (MILP)
- ✅ 精确算法 (Dijkstra, A*, MST)
- ✅ 元启发式算法 (GA, PSO, SA, VNS, TS, ACO, Memetic)
- ✅ 强化学习 (DQN, PPO, SW-RDQN)
- ✅ 图神经网络 (GNN)
- ✅ 多目标优化 (NSGA-II)
- ✅ 大规模问题求解 (分解算法)

---

## 🎯 算法实现清单

### ✅ 数学规划法 (1 种)

| 算法 | 文件 | 状态 | 说明 |
|------|------|------|------|
| MILP | `01_milp_basic.py` | ✅ | 混合整数线性规划，PuLP+CBC |

### ✅ 精确算法 (4 种)

| 算法 | 文件 | 状态 | 说明 |
|------|------|------|------|
| Dijkstra | `02_dijkstra.py` | ✅ | 单源最短路径，O((V+E)logV) |
| A* Search | `06_astar.py` | ✅ | 启发式搜索，f(n)=g(n)+h(n) |
| Prim MST | `07_minimum_spanning_tree.py` | ✅ | 最小生成树，优先队列实现 |
| Kruskal MST | `07_minimum_spanning_tree.py` | ✅ | 最小生成树，并查集实现 |

### ✅ 元启发式算法 (7 种)

| 算法 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 遗传算法 GA | `03_genetic_algorithm.py` | ✅ | 选择 +OX 交叉 + 交换变异 |
| 粒子群 PSO | `04_pso.py` | ✅ | 速度更新公式，群体智能 |
| 模拟退火 SA | `05_simulated_annealing.py` | ✅ | Metropolis 准则，退火计划 |
| 变邻域 VNS | `08_variable_neighborhood_search.py` | ✅ | 5 种邻域操作，系统性切换 |
| 禁忌搜索 TS | `09_tabu_search.py` | ✅ | 禁忌表 + 藐视准则 |
| 蚁群 ACO | `10_ant_colony_optimization.py` | ✅ | 信息素正反馈，精英蚂蚁 |
| 膜算法 Memetic | `17_memetic_algorithm.py` | ✅ | GA+VNS 混合，探索 - 开发平衡 |

### ✅ 强化学习 (4 种)

| 算法 | 文件 | 状态 | 说明 |
|------|------|------|------|
| Q-Learning | `12_dqn_reinforcement_learning.py` | ✅ | 时序差分，Q 表更新 |
| 进阶 DQN | `13_advanced_dqn.py` | ✅ | 经验回放 + 目标网络 |
| PPO | `15_ppo_policy_gradient.py` | ✅ | Actor-Critic，GAE 优势估计 |
| SW-RDQN | `20_swr_dqn_paper_implementation.py` | ✅ | 滑动窗口 + Double DQN 论文复现 |

### ✅ 图神经网络 (1 种)

| 算法 | 文件 | 状态 | 说明 |
|------|------|------|------|
| GNN | `16_gnn_graph_neural_network.py` | ✅ | GCN+GAT，消息传递机制 |

### ✅ 多目标优化 (1 种)

| 算法 | 文件 | 状态 | 说明 |
|------|------|------|------|
| NSGA-II | `18_multiobjective_nsga2.py` | ✅ | 非支配排序 + 拥挤度 + 精英保留 |

### ✅ 大规模问题求解 (1 种)

| 算法 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 分解算法 | `19_large_scale_decomposition.py` | ✅ | K-Means 聚类 + 分治优化 |

### ✅ 其他 (3 种)

| 算法 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 复合 DRL | `14_composite_drl_planner_v2.py` | ✅ | DRL+ 启发式混合 |
| Voronoi | `Voronoi_optimized.py` | ✅ | Voronoi 图空间划分 |
| 算法对比 | `11_algorithm_comparison.py` | ✅ | 统一测试框架 |

---

## 📊 性能对比 (30 节点测试)

| 排名 | 算法 | 成本 | 时间 (s) | 推荐场景 |
|------|------|------|---------|----------|
| #1 | Dijkstra | 59.71 | 0.000 | 精确最优，需完整图 |
| #2 | VNS | 347.13 | 1.619 | 元启发式最佳 ⭐ |
| #3 | TS | 349.50 | 0.428 | 快速优质 ⭐ |
| #4 | SA | 352.75 | 12.987 | 质量好但慢 |
| #5 | ACO | 373.31 | 2.092 | 群体智能 |
| #6 | GA | 380.00 | 2.500 | 全局搜索 |
| #7 | PSO | 385.00 | 1.800 | 参数敏感 |

> 完整对比实验见：[`11_algorithm_comparison.py`](examples/11_algorithm_comparison.py)

---

## 🚀 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/WAspirin/-.git
cd cable-optimization

# 安装依赖
pip install -r requirements.txt
```

### 运行示例

```bash
# 运行单个算法
python examples/03_genetic_algorithm.py

# 运行算法对比实验
python examples/11_algorithm_comparison.py

# 运行大规模分解算法
python examples/19_large_scale_decomposition.py
```

### 查看文档

```bash
# 算法索引
open docs/ALGORITHM_INDEX.md

# 技术报告
open docs/technical-report.md

# 学习进度
open docs/learning-progress.md
```

---

## 📁 项目结构

```
cable-optimization/
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── examples/                    # 算法实现 (24 个文件)
│   ├── 01_milp_basic.py
│   ├── 02_dijkstra.py
│   ├── 03_genetic_algorithm.py
│   ├── ...
│   └── 20_swr_dqn_paper_implementation.py
├── docs/                        # 文档 (20+ 文件)
│   ├── ALGORITHM_INDEX.md       # 算法索引目录
│   ├── algorithm-notes.md       # 详细算法笔记
│   ├── technical-report.md      # 技术报告
│   ├── learning-progress.md     # 学习进度追踪
│   ├── daily-task.md            # 每日任务说明
│   └── daily-report-*.md        # 每日报告 (15 篇)
├── tests/                       # 测试文件
│   ├── test_performance.py
│   └── test_real_performance.py
└── week2/                       # Week 2 学习笔记
    ├── 10_double_dqn_gnn.md
    └── 11_hybrid_algorithms.md
```

---

## 🎓 学习路径

### Week 1: 基础启发式算法 ✅
- Day 1-3: MILP, Dijkstra, GA, PSO, SA, A*, MST
- Day 4-6: VNS, TS, ACO
- Day 7: 对比实验 + 周总结

### Week 2: 进阶算法 ✅
- Day 8-9: DQN, PPO (强化学习)
- Day 10: GNN (图神经网络)
- Day 11: Memetic (混合算法)
- Day 12: NSGA-II (多目标优化)
- Day 13-14: 大规模分解 + 技术报告

### Week 3: 高级主题 🔄
- Day 15: 完整文档整理 ✅ 今日
- Day 16: 月总结 + 下一步计划 📝
- Day 17-21: 深入研究方向

---

## 📖 文档导航

| 文档 | 说明 | 链接 |
|------|------|------|
| 📑 算法索引 | 22 种算法快速参考 | [ALGORITHM_INDEX.md](docs/ALGORITHM_INDEX.md) |
| 📝 算法笔记 | 详细理论 + 公式 + 实现 | [algorithm-notes.md](docs/algorithm-notes.md) |
| 📊 技术报告 | 系统性总结 (9000 字) | [technical-report.md](docs/technical-report.md) |
| 📈 学习进度 | 每日进度追踪 | [learning-progress.md](docs/learning-progress.md) |
| 📅 每日报告 | 15 篇学习心得 | [daily-report-*.md](docs/) |
| 🔧 每日任务 | 学习流程说明 | [daily-task.md](docs/daily-task.md) |

---

## 🎯 算法选择指南

```
是否需要精确最优解？
├── 是 → 问题规模 < 50 节点？
│   ├── 是 → 使用 MILP (01_milp_basic.py)
│   └── 否 → 使用 Dijkstra/A* (02_dijkstra.py / 06_astar.py)
└── 否 → 是否多目标优化？
    ├── 是 → 使用 NSGA-II (18_multiobjective_nsga2.py)
    └── 否 → 问题规模 > 100 节点？
        ├── 是 → 使用分解算法 (19_large_scale_decomposition.py)
        └── 否 → 追求解质量？
            ├── 是 → 使用 VNS/TS (08/09)
            └── 否 → 使用 GA/PSO (03/04)
```

---

## 📊 项目统计

| 指标 | 数值 |
|------|------|
| **算法实现** | 22 种 |
| **代码文件** | 24 个 |
| **代码行数** | ~10,000 行 |
| **文档行数** | ~20,000 行 |
| **学习天数** | 15 天 |
| **周次进度** | 15/21 天 (71%) |
| **开始日期** | 2026-03-02 |
| **最新提交** | 2026-03-15 |

---

## 🔬 核心技术亮点

### 1. 完整的算法谱系
覆盖从精确算法到启发式、从传统优化到深度强化学习、从单目标到多目标的完整算法谱系

### 2. 统一的测试框架
所有算法在统一测试集上公平对比，提供可复现的性能基准

### 3. 丰富的可视化
每个算法都包含收敛曲线、路径图、热力图等可视化输出

### 4. 详细的文档
~20000 行文档，包括算法原理、公式推导、参数调优、使用示例

### 5. 实用的选择指南
基于实验结果的算法选择指南，帮助快速决策

---

## 🤝 贡献指南

欢迎贡献！可以通过以下方式参与：

1. **报告问题**: 发现 bug 或有改进建议？提 Issue！
2. **提交代码**: 实现新算法或优化现有实现？提 PR！
3. **完善文档**: 发现文档错误或不清晰？帮忙修正！
4. **分享应用**: 将算法应用到实际问题？分享案例！

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 👥 团队

- **研究方向**: WonderXi
- **实现与维护**: 智子 (Sophon)
- **起始日期**: 2026-03-02
- **当前状态**: 活跃开发中

---

## 📬 联系方式

- **GitHub**: [@WAspirin](https://github.com/WAspirin/-)
- **项目主页**: [cable-optimization](https://github.com/WAspirin/-/tree/master/cable-optimization)

---

**⭐ 如果这个项目对你有帮助，欢迎 Star 支持！**

_最后更新：2026-03-15_
