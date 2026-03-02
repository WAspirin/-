# 线缆自动布线优化 - 学习资料库

_智子为 WonderXi 整理的研究方向相关资料_

---

## 📦 GitHub 开源项目

### 高相关度项目

| 项目 | 语言 | 描述 | 更新时间 |
|------|------|------|----------|
| **[caerbannogwhite/cables-routing-optimization](https://github.com/caerbannogwhite/cables-routing-optimization)** | C++ | 风电场涡轮机间最小成本连接方案 | 2020-01 |
| **[deadinternet2027/CableOptimization](https://github.com/deadinternet2027/CableOptimization)** | Python | 线缆布线优化 | 2025-05 |
| **[Will-iam-L/L-MVNS-for-urban-cable-routing](https://github.com/Will-iam-L/L-MVNS-for-urban-cable-routing)** | Python | 城市线缆布线的变邻域搜索算法 | 2026-01 |
| **[skyoon1109/Offshore-windfarm-cable-routing-optimization](https://github.com/skyoon1109/Offshore-windfarm-cable-routing-optimization)** | Jupyter | 海上风电场线缆路由 MIP 模型 | 2025-03 |
| **[skyoon1109/Solar-plant-cable-routing-optimization](https://github.com/skyoon1109/Solar-plant-cable-routing-optimization)** | Jupyter | 大型太阳能电站线缆布线优化 | 2025-03 |
| **[AbelKlaassens/Master-Thesis-Subsea-Cable-Routing-Optimization](https://github.com/AbelKlaassens/Master-Thesis-Subsea-Cable-Routing-Optimization)** | Python | 海底电缆路由多目标优化模型 | 2025-06 |
| **[mehedi160/Cable-route-optimization-using-MILP](https://github.com/mehedi160/Cable-route-optimization-using-MILP)** | Python | 混合整数线性规划 (MILP) 方法 | 2024-06 |

### 有趣的项目

- **hrahman12/Quantum-Optimization-for-Shark-Aware-Subsea-Cable-Routing** 🦈
  - 量子优化 + 鲨鱼感知的海底电缆布线
  - 这个研究方向太有创意了！

---

## 🔬 常用优化方法

### 1. 数学规划方法
- **MILP/MIP** - 混合整数线性规划
- **Linear Programming** - 线性规划
- **Multi-objective Optimization** - 多目标优化

### 2. 元启发式算法
- **MVNS** - 变邻域搜索 (Variable Neighborhood Search)
- **Path-Relinking** - 路径重连元启发式
- **Genetic Algorithm** - 遗传算法
- **Particle Swarm Optimization** - 粒子群优化
- **Simulated Annealing** - 模拟退火

### 3. 机器学习方法
- **Capacitated K-means** - 容量约束 K 均值聚类
- **Reinforcement Learning** - 强化学习（用于动态环境）

### 4. 路径规划算法
- **A*** - A star 算法
- **Dijkstra** - 最短路径
- **RRT** - 快速探索随机树

---

## 📊 应用场景

| 场景 | 特点 | 约束条件 |
|------|------|----------|
| 海上风电场 | 距离长、环境复杂 | 海床地形、渔业活动、海洋生物 |
| 太阳能电站 | 大规模、密集布线 | 容量约束、地形限制 |
| 城市电网 | 人口密集、空间有限 | 地下管道、交通规则 |
| 数据中心 | 高密度、散热要求 | 线缆长度、信号衰减 |
| 汽车线束 | 3D 空间、振动环境 | 空间约束、电磁干扰 |

---

## 🎯 研究热点

1. **多目标优化** - 成本、可靠性、环境影响的平衡
2. **动态环境适应** - 实时调整布线路径
3. **量子计算应用** - 量子优化算法
4. **AI/ML 集成** - 用机器学习预测最优解
5. **数字孪生** - 虚拟环境中的布线仿真

---

## 📚 待深入探索

- [ ] 阅读 caerbannogwhite 项目的 C++ 实现
- [ ] 学习 MVNS 算法原理
- [ ] 研究 MILP 在布线问题中的建模方法
- [ ] 了解 Gurobi/CPLEX 等优化求解器
- [ ] 探索强化学习在动态布线中的应用

---

## 💡 核心 MIP 模型示例

来自 `skyoon1109/Offshore-windfarm-cable-routing-optimization` 的海上风电场模型：

### 目标函数
```
min Σ(i,j)∈A Σt∈T c^t_ij * x^t_ij + Σk∈V0 a_k * u_k
```
- 最小化线缆总成本（涡轮机 - 变电站 + 变电站 - 陆地）

### 关键决策变量
- `x^t_ij`: 边 (i,j) 是否使用电缆类型 t
- `y_ij`: 边 (i,j) 是否连接（不限电缆类型）
- `f_ij`: 从节点 i 到 j 的能量流
- `u_k`: 是否选择候选变电站 k

### 核心约束
1. **流量守恒**: 每个涡轮机的流入 - 流出 = 1
2. **容量约束**: 电缆容量 ≥ 能量流
3. **拓扑约束**: 每个涡轮机出度 = 1
4. **变电站选择**: 9 个候选中选 1 个
5. **交叉避免**: 使用 Lazy Constraint + Callback 处理

### 拓扑类型
- **Branch topology** - 分支拓扑
- **Balanced Radial topology** - 平衡辐射拓扑

### 工具链
- **求解器**: Gurobi
- **语言**: Python + Jupyter Notebook
- **技巧**: Callback 函数添加 Lazy Constraint 避免线缆交叉

---

_最后更新：2026-03-02 by 智子_
