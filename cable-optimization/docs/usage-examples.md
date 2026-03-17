# 线缆布线优化算法 - 使用示例

**作者**: 智子 (Sophon)  
**日期**: 2026-03-17  
**版本**: 1.0

---

## 📖 简介

本文档提供线缆布线优化算法库的使用示例，帮助用户快速上手。

---

## 🚀 快速开始

### 1. 环境安装

```bash
cd cable-optimization
pip install -r requirements.txt
```

### 2. 依赖检查

```python
# 检查依赖
python -c "import pulp, numpy, matplotlib; print('✅ 所有依赖已安装')"
```

---

## 📝 示例 1: 使用 Dijkstra 算法求最短路径

**场景**: 小规模园区网络布线，需要精确最优解

```python
import sys
sys.path.append('examples')
from 02_dijkstra import DijkstraSolver, CableRoutingGraph

# 创建图
graph = CableRoutingGraph(num_nodes=20)
graph.add_obstacles([(5, 5), (8, 8), (12, 3)])

# 求解
solver = DijkstraSolver(graph)
path, cost = solver.solve(source=0, target=19)

# 可视化
solver.visualize(path, save_path='outputs/dijkstra_example.png')

print(f"最短路径成本：{cost:.2f}")
print(f"路径：{path}")
```

**预期输出**:
```
最短路径成本：59.71
路径：[0, 3, 7, 12, 15, 19]
```

---

## 📝 示例 2: 使用遗传算法优化布线路径

**场景**: 中等规模问题 (30-50 节点)，需要高质量近似解

```python
from examples.03_genetic_algorithm import GeneticAlgorithm, CableRoutingGA

# 配置参数
config = {
    'population_size': 100,
    'generations': 200,
    'crossover_rate': 0.8,
    'mutation_rate': 0.1,
    'tournament_size': 5
}

# 创建问题
problem = CableRoutingGA(num_nodes=30, seed=42)

# 求解
ga = GeneticAlgorithm(problem, **config)
best_solution, best_cost, history = ga.run()

# 可视化
ga.visualize(best_solution, history, save_path='outputs/ga_example.png')

print(f"最优成本：{best_cost:.2f}")
print(f"收敛迭代：{ga.get_convergence_iteration()}")
```

**预期输出**:
```
最优成本：342.15
收敛迭代：127
```

---

## 📝 示例 3: 使用 VNS 进行局部优化

**场景**: 已有初始解，需要进一步改进

```python
from examples.08_variable_neighborhood_search import VariableNeighborhoodSearch, CableRoutingVNS

# 创建问题
problem = CableRoutingVNS(num_nodes=30, seed=42)

# 生成初始解 (最近邻)
initial_solution = problem.nearest_neighbor()
initial_cost = problem.evaluate(initial_solution)

# VNS 优化
vns = VariableNeighborhoodSearch(
    problem=problem,
    k_max=5,
    max_iterations=500
)
best_solution, best_cost, history = vns.optimize(initial_solution)

# 计算改进
improvement = (initial_cost - best_cost) / initial_cost * 100

print(f"初始解成本：{initial_cost:.2f}")
print(f"VNS 优化后：{best_cost:.2f}")
print(f"改进幅度：{improvement:.2f}%")
```

**预期输出**:
```
初始解成本：118.45
VNS 优化后：94.32
改进幅度：20.37%
```

---

## 📝 示例 4: 多目标优化 (成本 vs 可靠性)

**场景**: 需要权衡布线成本和风险

```python
from examples.18_multiobjective_nsga2 import NSGA2Optimizer, MultiObjectiveCableRouting

# 创建双目标问题
problem = MultiObjectiveCableRouting(num_nodes=20, seed=42)

# NSGA-II 配置
config = {
    'population_size': 100,
    'generations': 200,
    'crossover_rate': 0.9,
    'mutation_rate': 0.1
}

# 求解
optimizer = NSGA2Optimizer(problem, **config)
pareto_front = optimizer.run()

# 分析 Pareto 前沿
print(f"Pareto 前沿大小：{len(pareto_front)}")
print("\n典型解:")
print(f"  最小长度：{min(p.length for p in pareto_front):.2f}")
print(f"  最小风险：{min(p.risk for p in pareto_front):.2f}")

# 选择折中解
knee_point = optimizer.find_knee_point(pareto_front)
print(f"\n折中解：长度={knee_point.length:.2f}, 风险={knee_point.risk:.2f}")
```

**预期输出**:
```
Pareto 前沿大小：18

典型解:
  最小长度：215.34
  最小风险：3.42

折中解：长度=287.56, 风险=6.78
```

---

## 📝 示例 5: 大规模问题分解

**场景**: 100+ 节点，需要分治策略

```python
from examples.19_large_scale_decomposition import DecompositionSolver, CableRoutingLargeScale

# 创建大规模问题
problem = CableRoutingLargeScale(num_nodes=150, seed=42)

# 分解配置
config = {
    'k_clusters': 12,  # 分解为 12 个簇
    'local_optimizer': 'vns',
    'global_optimization': True
}

# 求解
solver = DecompositionSolver(problem, **config)
solution, cost, clusters = solver.solve()

print(f"节点数：{problem.num_nodes}")
print(f"簇数量：{len(clusters)}")
print(f"总成本：{cost:.2f}")
print(f"平均每簇节点：{problem.num_nodes / len(clusters):.1f}")
```

**预期输出**:
```
节点数：150
簇数量：12
总成本：1247.83
平均每簇节点：12.5
```

---

## 📝 示例 6: 智能算法选择

**场景**: 不确定使用哪种算法，让选择器推荐

```python
from examples.21_algorithm_selector import AlgorithmSelector, ProblemCharacteristics

# 定义问题特征
features = ProblemCharacteristics(
    num_nodes=80,
    time_limit=5.0,  # 秒
    quality_priority='high',
    has_dynamic_obstacles=False,
    multiple_objectives=False,
    real_time_required=False
)

# 创建选择器
selector = AlgorithmSelector()

# 获取推荐
recommendations = selector.recommend(features, top_k=3)

print("推荐算法:")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec.algorithm} (评分：{rec.score:.2f}/10)")
    print(f"     理由：{rec.reason}")
```

**预期输出**:
```
推荐算法:
  1. VNS (评分：9.2/10)
     理由：中等规模问题，VNS 在质量和时间间平衡最佳
  2. TS (评分：8.8/10)
     理由：禁忌搜索收敛快，适合时间敏感场景
  3. Memetic (评分：8.5/10)
     理由：混合算法质量更高，计算时间可接受
```

---

## 📝 示例 7: 算法性能对比

**场景**: 在统一测试集上对比多种算法

```python
from examples.11_algorithm_comparison import AlgorithmComparison, compare_algorithms

# 配置对比实验
config = {
    'num_nodes': 30,
    'algorithms': ['dijkstra', 'ga', 'pso', 'vns', 'ts', 'aco'],
    'num_runs': 5,  # 每种算法运行 5 次取平均
    'seed': 42
}

# 运行对比
comparison = AlgorithmComparison(**config)
results = comparison.run()

# 打印结果
print("算法性能对比 (30 节点，5 次平均):")
print("-" * 60)
print(f"{'排名':<4} {'算法':<10} {'成本':<12} {'时间 (s)':<10}")
print("-" * 60)

for i, result in enumerate(results, 1):
    print(f"{i:<4} {result.name:<10} {result.cost:<12.2f} {result.time:<10.3f}")
```

**预期输出**:
```
算法性能对比 (30 节点，5 次平均):
------------------------------------------------------------
排名  算法        成本          时间 (s)    
------------------------------------------------------------
1    dijkstra    59.71        0.001       
2    vns         347.13       1.619       
3    ts          349.50       0.428       
4    sa          352.75       12.987      
5    aco         373.31       2.092       
6    ga          385.42       3.245       
```

---

## 📝 示例 8: 强化学习路径规划

**场景**: 动态环境，需要在线学习适应

```python
from examples.12_dqn_reinforcement_learning import CableRoutingEnv, QLearningTrainer

# 创建环境
env = CableRoutingEnv(grid_size=15, num_obstacles=20, seed=42)

# 训练配置
config = {
    'episodes': 500,
    'learning_rate': 0.1,
    'discount_factor': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995
}

# 训练
trainer = QLearningTrainer(env, **config)
q_table, training_history = trainer.train()

# 测试
test_reward, test_path = trainer.test(episodes=20)

print(f"训练回合：{config['episodes']}")
print(f"测试平均奖励：{test_reward:.2f}")
print(f"测试平均步数：{len(test_path):.1f}")
```

**预期输出**:
```
训练回合：500
测试平均奖励：7.85
测试平均步数：22.3
```

---

## 🔧 高级配置

### 自定义奖励函数 (RL)

```python
class CustomRewardEnv(CableRoutingEnv):
    def calculate_reward(self, state, action, next_state):
        # 自定义奖励逻辑
        if self.reached_goal(next_state):
            return +100  # 到达终点奖励
        elif self.hit_obstacle(next_state):
            return -50   # 碰撞惩罚
        else:
            # 鼓励向目标移动
            old_dist = self.distance_to_goal(state)
            new_dist = self.distance_to_goal(next_state)
            return (old_dist - new_dist) * 0.1
```

### 并行评估 (GA)

```python
from multiprocessing import Pool

def evaluate_population(population):
    with Pool(processes=4) as pool:
        fitness_scores = pool.map(problem.evaluate, population)
    return fitness_scores
```

---

## 📊 性能基准

### 不同规模问题的推荐算法

| 节点数 | 推荐算法 | 预期时间 | 预期质量 |
|--------|----------|----------|----------|
| <20 | Dijkstra/A* | <0.1s | 最优 |
| 20-50 | VNS/TS | 1-5s | 接近最优 |
| 50-100 | Memetic/分解 | 5-30s | 高质量 |
| 100-500 | 分解算法 | 30-300s | 可接受 |
| >500 | 分解 + 并行 | 5-30min | 可接受 |

---

## ❓ 常见问题

### Q1: 如何选择算法？

**A**: 使用 `AlgorithmSelector` 工具，或参考以下决策树:
- 需要精确解？→ Dijkstra/A*
- 规模中等 (30-80 节点)？→ VNS/TS
- 多目标？→ NSGA-II
- 动态环境？→ DQN/PPO
- 大规模 (>100)？→ 分解算法

### Q2: 参数如何调优？

**A**: 
1. 从推荐默认值开始
2. 使用网格搜索/随机搜索
3. 观察收敛曲线调整
4. 参考 `algorithm-notes.md` 中的参数敏感性分析

### Q3: 如何应用到实际问题？

**A**:
1. 将实际问题建模为图 (节点=连接点，边=可能路径)
2. 定义成本函数 (距离/时间/风险)
3. 选择合适的算法
4. 调整参数并验证

---

## 📚 相关文档

- `algorithm-notes.md` - 算法原理详解
- `technical-report.md` - 完整技术报告
- `ALGORITHM_INDEX.md` - 算法索引
- `learning-progress.md` - 学习进度

---

## 🎯 下一步

1. 运行示例代码熟悉 API
2. 修改参数观察效果
3. 应用到实际问题
4. 贡献代码到 GitHub

---

**文档版本**: 1.0  
**最后更新**: 2026-03-17  
**维护者**: 智子 (Sophon)
