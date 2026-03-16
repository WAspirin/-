# Week 2 Day 11 - 混合算法设计

_结合启发式算法与强化学习的优势_

---

## 📚 一、为什么需要混合算法？

### 单一算法的局限性

| 算法类型 | 优势 | 劣势 |
|---------|------|------|
| **启发式 (GA/PSO/VNS/TS)** | 全局搜索、无需训练、可解释 | 收敛慢、参数敏感、无学习能力 |
| **强化学习 (DQN/PPO)** | 自适应、可迁移、在线学习 | 训练时间长、样本效率低、不稳定 |
| **精确算法 (MILP/A*)** | 全局最优、理论保证 | 计算复杂度高、规模受限 |

### 混合策略的核心思想
**取长补短**：
- 用启发式算法快速找到可行解
- 用 RL 学习优化策略，加速收敛
- 用精确算法保证局部最优

---

## 📚 二、混合算法设计模式

### 模式 1: 启发式初始化 + RL 优化

```
阶段 1: 启发式算法 (GA/VNS) 生成初始种群
        ↓
阶段 2: RL 学习从初始解到最优解的策略
        ↓
阶段 3: 在线微调 + 迁移学习
```

**代码框架**:
```python
class Hybrid_Heuristic_RL:
    def __init__(self):
        self.heuristic = VNS()  # 或 GA/PSO/TS
        self.rl_agent = PPO()
    
    def train(self, problem_instances):
        # 阶段 1: 启发式生成初始解
        initial_solutions = []
        for instance in problem_instances:
            sol = self.heuristic.solve(instance, max_iter=50)
            initial_solutions.append(sol)
        
        # 阶段 2: RL 学习优化策略
        for sol in initial_solutions:
            # 状态：当前解 + 问题特征
            state = self.encode_state(sol)
            # RL 学习如何改进
            self.rl_agent.learn(state, action, reward)
    
    def solve(self, new_instance):
        # 快速生成初始解
        initial = self.heuristic.solve(new_instance, max_iter=20)
        # RL 快速优化
        optimized = self.rl_agent.optimize(initial, steps=100)
        return optimized
```

**适用场景**:
- 问题规模大，纯 RL 训练慢
- 需要快速得到可行解
- 有大量相似问题实例 (可迁移)

---

### 模式 2: RL 指导启发式搜索

```
传统启发式：随机选择邻域操作
        ↓
RL 增强：学习何时使用哪种邻域操作
        ↓
智能搜索：根据状态选择最优邻域
```

**代码框架**:
```python
class RL_Guided_VNS:
    def __init__(self):
        self.vns = VNS()
        self.policy_net = DQN()  # 学习邻域选择策略
        
        # 邻域操作集合
        self.operators = {
            'swap_adjacent': self.swap_adjacent,
            'swap_any': self.swap_any,
            'reverse': self.reverse_path,
            'insert': self.insert_node,
            '2opt': self.two_opt
        }
    
    def select_operator(self, state):
        # 状态：当前解质量、搜索历史、问题特征
        state_features = self.extract_features(state)
        
        # RL 策略选择邻域操作
        action_probs = self.policy_net(state_features)
        selected_op = self.operators[action_probs.argmax()]
        
        return selected_op
    
    def search(self, initial_solution):
        current = initial_solution
        best = current
        
        for iteration in range(max_iter):
            # 状态编码
            state = {
                'current_cost': current.cost,
                'best_cost': best.cost,
                'iteration': iteration,
                'no_improve_count': self.no_improve_count
            }
            
            # RL 选择邻域操作
            operator = self.select_operator(state)
            
            # 应用操作
            neighbor = operator(current)
            
            # 接受准则 (VNS 原有逻辑)
            if neighbor.cost < current.cost:
                current = neighbor
                if neighbor.cost < best.cost:
                    best = neighbor
                    self.no_improve_count = 0
            else:
                self.no_improve_count += 1
            
            # RL 奖励
            reward = current.cost - neighbor.cost
            self.policy_net.update(state, action, reward)
        
        return best
```

**适用场景**:
- 启发式算法有多个邻域操作
- 不同操作在不同阶段效果不同
- 希望自适应调整搜索策略

---

### 模式 3: 并行搜索 + 信息共享

```
多个算法并行运行
        ↓
定期共享最优解/信息素/策略
        ↓
协同进化，避免局部最优
```

**代码框架**:
```python
class Parallel_Hybrid_Search:
    def __init__(self):
        self.algorithms = {
            'ga': GeneticAlgorithm(),
            'pso': ParticleSwarm(),
            'vns': VNS(),
            'dqn': DQN_Agent()
        }
        self.best_global = None
    
    def search(self, problem):
        # 初始化
        solutions = {name: algo.initialize(problem) 
                     for name, algo in self.algorithms.items()}
        
        for generation in range(max_generations):
            # 各算法独立搜索
            for name, algo in self.algorithms.items():
                solutions[name] = algo.step(solutions[name], problem)
            
            # 信息共享 (每 N 代)
            if generation % 10 == 0:
                self.share_information(solutions)
        
        # 返回最优解
        return min(solutions.values(), key=lambda s: s.cost)
    
    def share_information(self, solutions):
        # 找到当前最优解
        self.best_global = min(solutions.values(), key=lambda s: s.cost)
        
        # 共享给所有算法
        for name, algo in self.algorithms.items():
            if name == 'ga':
                algo.inject_individual(self.best_global)
            elif name == 'pso':
                algo.update_global_best(self.best_global)
            elif name == 'dqn':
                algo.add_experience(self.best_global)
```

**适用场景**:
- 计算资源充足 (可并行)
- 问题复杂，单一算法易陷入局部最优
- 需要鲁棒性 (多算法保险)

---

### 模式 4: 分层决策 (宏观 + 微观)

```
高层 (RL): 决策策略选择 (走哪条大路径)
        ↓
低层 (启发式): 局部优化 (具体怎么走)
        ↓
协同完成复杂任务
```

**在线缆布线中的应用**:
```
高层 RL: 选择布线路径的"骨架" (经过哪些关键节点)
        ↓
低层 A*: 在骨架之间进行精确路径规划
        ↓
低层 VNS: 对完整路径进行局部优化
```

**代码框架**:
```python
class Hierarchical_CableRouter:
    def __init__(self):
        self.high_level_rl = PPO()  # 学习宏观策略
        self.low_level_planner = AStar()  # 精确路径规划
        self.local_optimizer = VNS()  # 局部优化
    
    def route(self, start, end, connectors, obstacles):
        # 高层：选择关键节点序列
        key_points = self.high_level_rl.select_key_points(
            start, end, connectors, obstacles
        )
        
        # 低层：逐段路径规划
        full_path = []
        for i in range(len(key_points) - 1):
            segment = self.low_level_planner.plan(
                key_points[i], 
                key_points[i+1], 
                obstacles
            )
            full_path.extend(segment)
        
        # 局部优化：平滑 + 优化
        optimized_path = self.local_optimizer.optimize(
            full_path, 
            constraints={'turning_radius': True}
        )
        
        return optimized_path
```

**适用场景**:
- 任务可分解为多个层次
- 高层决策需要学习，低层可精确计算
- 长序列决策问题 (如布线、路径规划)

---

## 📚 三、混合算法在线缆布线中的具体应用

### 应用 1: GA + DQN 混合

**设计思路**:
- GA 负责全局探索 (生成多样化路径方案)
- DQN 负责局部优化 (学习如何改进给定路径)

**实现步骤**:
```python
class GA_DQN_Hybrid:
    def __init__(self):
        self.ga = GeneticAlgorithm(
            population_size=50,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        self.dqn = DQN()
    
    def train(self, problem_instances):
        for instance in problem_instances:
            # GA 生成初始种群
            population = self.ga.initialize(instance)
            
            # 对每个个体，用 DQN 优化
            for individual in population:
                optimized = self.dqn.optimize(individual, steps=50)
                individual.fitness = -optimized.cost  # 适应度
            
            # GA 进化
            population = self.ga.evolve(population)
            
            # DQN 学习
            best = min(population, key=lambda x: x.cost)
            self.dqn.update_from_trajectory(best)
```

---

### 应用 2: VNS + PPO 混合

**设计思路**:
- VNS 提供多种邻域操作
- PPO 学习何时使用哪种邻域操作

**状态特征设计**:
```python
def extract_state(current_solution, search_history):
    return {
        # 解的质量
        'current_cost': current_solution.cost,
        'best_cost': search_history.best_cost,
        'gap_to_best': (current_solution.cost - search_history.best_cost) / search_history.best_cost,
        
        # 搜索状态
        'iteration': search_history.iteration,
        'no_improve_count': search_history.no_improve_count,
        'operator_used': search_history.last_operator,
        
        # 解的结构特征
        'num_turns': current_solution.count_turns(),
        'path_length': current_solution.length,
        'constraint_violations': current_solution.check_constraints(),
        
        # 问题特征
        'num_connectors': current_solution.num_connectors,
        'grid_density': current_solution.occupied_ratio
    }
```

**动作空间**:
```python
actions = {
    0: 'swap_adjacent',    # 交换相邻节点
    1: 'swap_any',         # 交换任意节点
    2: 'reverse',          # 路径逆转
    3: 'insert',           # 插入节点
    4: '2opt',             # 2-opt 优化
    5: 'perturb',          # 扰动 (跳出局部最优)
}
```

**奖励设计**:
```python
def compute_reward(old_cost, new_cost, action, no_improve_count):
    # 基础奖励：改进量
    improvement = old_cost - new_cost
    
    # 稀疏奖励增强
    if new_cost < old_cost:
        reward = improvement + 1.0  # 额外奖励
    else:
        reward = improvement - 0.1  # 轻微惩罚
    
    # 鼓励探索 (长时间未改进时使用扰动)
    if action == 'perturb' and no_improve_count > 10:
        reward += 0.5
    
    return reward
```

---

### 应用 3: ACO + RL 混合

**设计思路**:
- ACO 的信息素机制适合路径构建
- RL 可以学习信息素更新策略

**混合策略**:
```python
class ACO_RL_Hybrid:
    def __init__(self):
        self.aco = AntColony(
            num_ants=20,
            alpha=1.0,  # 信息素权重
            beta=2.0,   # 启发式权重
            rho=0.1     # 蒸发率
        )
        self.rl = DQN()  # 学习 alpha, beta, rho 参数
    
    def solve(self, graph):
        # RL 动态调整 ACO 参数
        state = self.encode_search_state()
        params = self.rl.select_parameters(state)
        
        self.aco.alpha = params['alpha']
        self.aco.beta = params['beta']
        self.aco.rho = params['rho']
        
        # ACO 搜索
        paths = []
        for ant in range(self.aco.num_ants):
            path = self.aco.build_path(graph)
            paths.append(path)
        
        # 更新信息素
        self.aco.update_pheromones(paths)
        
        # RL 学习
        best_path = min(paths, key=lambda p: p.cost)
        reward = -best_path.cost
        self.rl.update(state, params, reward)
        
        return best_path
```

---

## 🎯 四、今日实践任务

### 任务 1: 实现 RL 指导的 VNS (90 分钟)

```bash
cd cable-optimization/week2/
python 13_rl_guided_vns.py
```

**目标**:
- 实现 DQN 策略网络 (选择邻域操作)
- 集成到 VNS 框架中
- 对比纯 VNS vs RL 指导 VNS 的性能

**预期结果**:
- 收敛速度提升 20-30%
- 最终解质量提升 5-10%

---

### 任务 2: 实现 GA+DQN 混合 (60 分钟)

```bash
python 14_ga_dqn_hybrid.py
```

**目标**:
- GA 生成初始种群
- DQN 优化每个个体
- 协同进化

---

### 任务 3: 设计分层路由架构 (30 分钟)

```bash
python 15_hierarchical_router.py
```

**目标**:
- 高层 PPO 选择关键节点
- 低层 A* 进行路径规划
- 验证分层决策的可行性

---

## 📊 五、性能对比预期

| 算法 | 收敛速度 | 解质量 | 稳定性 | 实现难度 |
|------|---------|--------|--------|---------|
| VNS (基准) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| RL 指导 VNS | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| GA+DQN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 分层路由 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |

---

## 💡 六、设计建议

### 1. 从简单开始
先实现 **RL 指导 VNS** (模式 2)，因为：
- 代码改动最小 (在现有 VNS 基础上增加策略网络)
- 效果提升明显
- 便于调试

### 2. 状态特征很重要
- 不要只用"当前成本"作为状态
- 加入搜索历史、解的结构特征
- 可以考虑用自编码器学习状态表示

### 3. 奖励设计是关键
- 稀疏奖励 (只在找到更优解时给奖励) → 学习慢
- 稠密奖励 (每步都给奖励) → 可能学到次优策略
- 建议：混合奖励 = 改进量 + 稀疏 bonus + 探索鼓励

### 4. 训练技巧
- 先在简单问题上预训练
- 逐步增加问题难度 (课程学习)
- 保存检查点，避免灾难性遗忘

---

## 🔗 七、参考资料

1. **混合元启发式**: Blum et al. (2011) - "Hybrid Metaheuristics"
2. **RL 指导搜索**: Da Silva et al. (2020) - "Reinforcement Learning for Variable Neighborhood Search"
3. **GA+RL**: Zhang et al. (2021) - "Genetic Algorithm enhanced by Deep Reinforcement Learning"
4. **分层 RL**: Bacon et al. (2017) - "The Option-Critic Architecture"

**代码参考**:
- https://github.com/Naixue/Zhang/tree/master/Hybrid-GA-RL
- https://github.com/MaximilianAnzinger/pytorch-dqn-vns

---

_创建时间：2026-03-11 02:11_
_智子主动准备 - Week 2 Day 11 学习材料_
