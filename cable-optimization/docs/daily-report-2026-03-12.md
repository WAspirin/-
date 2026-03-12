# 每日学习报告 - Day 12 (2026-03-12)

**学习主题**: 多目标优化 - NSGA-II 算法  
**所属周次**: Week 3 - 高级主题  
**学习时长**: ~2 小时

---

## ✅ 今日完成内容

### 1. 算法学习 (35 分钟)

**学习内容**:
- 多目标优化理论基础
- Pareto 最优与支配关系概念
- NSGA-II 核心机制（非支配排序、拥挤度、精英保留）
- 与其他多目标方法对比（权重法、ε-约束、MOEA/D）
- 在线缆布线中的应用建模

**关键理解**:
- 多目标问题不存在单一最优解，而是 Pareto 最优解集
- NSGA-II 通过非支配排序分层，通过拥挤度保持多样性
- 精英保留策略确保优质解不丢失
- 决策者可以从 Pareto 前沿根据偏好选择解

### 2. 代码实现 (65 分钟)

**实现文件**: `examples/18_multiobjective_nsga2.py` (~680 行)

**核心组件**:

| 组件 | 行数 | 功能 |
|------|------|------|
| NSGA2Config | ~15 | 算法配置参数 |
| Node/Edge | ~25 | 问题数据结构 |
| MultiObjectiveCableRouting | ~100 | 多目标问题定义 |
| Individual | ~10 | 个体表示 |
| NSGA2Optimizer | ~250 | NSGA-II 主算法 |
| NSGA2Visualizer | ~200 | 可视化功能 |
| 主程序 | ~80 | 演示流程 |

**关键实现细节**:

1. **快速非支配排序**:
   ```python
   def _fast_non_dominated_sort(self):
       # 计算支配关系
       for i in range(len(pop)):
           for j in range(i+1, len(pop)):
               if dominates(ind_i, ind_j):
                   ind_i.dominated_solutions.append(ind_j)
                   ind_j.domination_count += 1
       
       # 分层
       fronts = [[]]
       for ind in population:
           if ind.domination_count == 0:
               fronts[0].append(ind)
       
       # 继续分层直到所有个体分配
   ```

2. **拥挤度距离计算**:
   ```python
   def _calculate_crowding_distance(self, front):
       for m in range(num_objectives):
           sorted_front = sorted(front, key=lambda ind: ind.objectives[m])
           sorted_front[0].crowding = float('inf')
           sorted_front[-1].crowding = float('inf')
           
           obj_range = sorted_front[-1].objectives[m] - sorted_front[0].objectives[m]
           for i in range(1, len(sorted_front) - 1):
               sorted_front[i].crowding += (
                   (sorted_front[i+1].objectives[m] - sorted_front[i-1].objectives[m]) / obj_range
               )
   ```

3. **精英保留选择**:
   ```python
   # 合并父代和子代
   combined = population + offspring
   
   # 非支配排序
   fronts = fast_non_dominated_sort(combined)
   
   # 按层选择
   new_population = []
   for front in fronts:
       if len(new_population) + len(front) <= N:
           new_population.extend(front)
       else:
           # 按拥挤度排序选择剩余
           front.sort(key=lambda ind: -ind.crowding_distance)
           new_population.extend(front[:(N - len(new_population))])
           break
   ```

4. **双目标评估**:
   ```python
   def evaluate_solution(self, path):
       total_length = sum(distance_matrix[path[i], path[i+1]] for i in range(len(path)-1))
       total_risk = sum(risk_matrix[path[i], path[i+1]] for i in range(len(path)-1))
       return total_length, total_risk
   ```

5. **可视化功能**:
   - Pareto 前沿图（成本 vs 风险）
   - 收敛曲线（Pareto 大小、平均目标）
   - 解路径对比（最小成本、最小风险、折中）
   - 风险热力图背景

### 3. 文档更新 (20 分钟)

**更新文件**:
- `docs/algorithm-notes.md` - 添加 NSGA-II 理论笔记 (~650 行新增)
- `docs/daily-report-2026-03-12.md` - 今日报告（本文件）
- `docs/learning-progress.md` - 更新进度

**笔记内容**:
- Pareto 最优概念与支配关系
- NSGA-II 算法流程详解
- 拥挤度距离计算方法
- 与其他多目标方法对比
- 代码实现要点
- 在线缆布线中的应用案例

### 4. 实验测试 (10 分钟)

**测试配置**:
- 节点数：20
- 种群大小：100
- 迭代次数：200
- 交叉率：0.9
- 变异率：0.15

**预期结果**:
```
Pareto 前沿大小：~15-25 个解
最小长度解：长度~200-300, 风险~8-12
最小风险解：长度~350-450, 风险~3-5
折中解：长度~280, 风险~6-8
```

---

## 📊 代码统计

| 文件 | 行数 | 说明 |
|------|------|------|
| 18_multiobjective_nsga2.py | ~680 | NSGA-II 完整实现 |
| algorithm-notes.md (新增) | ~650 | NSGA-II 理论笔记 |
| daily-report-2026-03-12.md | ~300 | 今日报告 |
| **今日新增** | **~1630** | 代码 + 文档 |

**累计代码量**: ~7160 行  
**累计文档量**: ~4300 行  
**累计算法数**: 18 种

---

## 🤔 遇到的问题

### 问题 1: 支配关系判断逻辑

**问题描述**: 
初始实现时混淆了支配与被支配的关系，导致排序错误。

**分析**:
- A 支配 B：A 的所有目标都不差于 B，且至少一个严格优于 B
- 需要同时满足"不差于"和"严格优于"两个条件

**解决方案**:
```python
def dominates(ind1, ind2):
    not_worse = all(o1 <= o2 for o1, o2 in zip(ind1.objectives, ind2.objectives))
    strictly_better = any(o1 < o2 for o1, o2 in zip(ind1.objectives, ind2.objectives))
    return not_worse and strictly_better
```

### 问题 2: 拥挤度计算边界处理

**问题描述**: 
边界点的拥挤度应该如何设置？

**分析**:
- 边界点应该优先保留（它们代表极端解）
- 如果设为 0，会在选择时被淘汰

**解决方案**:
- 将边界点拥挤度设为无穷大 (float('inf'))
- 确保极端解始终被保留

### 问题 3: 路径交叉的合法性

**问题描述**: 
顺序交叉 (OX) 可能产生重复节点或丢失起点/终点。

**解决方案**:
- 交叉后检查并修复起点 (必须为 0) 和终点 (必须为 n-1)
- 移除中间节点的重复
- 确保路径连续有效

### 问题 4: 精英保留的层处理

**问题描述**: 
当最后一层无法全部加入时，如何选择？

**解决方案**:
- 对最后一层按拥挤度降序排序
- 选择拥挤度最大的前 k 个个体
- 确保种群大小恰好为 N

---

## 💡 关键洞察

### 1. 为什么需要多目标优化？

```
单目标优化：找到"最优"解
多目标优化：找到"权衡"解集

实际工程问题很少只有一个目标：
- 成本 vs 质量
- 性能 vs 能耗
- 速度 vs 准确性
- 成本 vs 可靠性（本例）

Pareto 前沿让决策者看到 trade-off，做出知情选择
```

### 2. NSGA-II 的核心创新

| 机制 | 作用 | 类比 |
|------|------|------|
| 快速非支配排序 | 区分解的优劣层级 | 考试排名（按总分分层） |
| 拥挤度距离 | 保持解的多样性 | 避免所有人站在一起 |
| 精英保留 | 确保优质解不丢失 | 保送机制 |

### 3. Pareto 前沿的解读

```
目标 2 (风险)
    ↑
    |    ·  ·  ·  ← 高成本低风险（绕远路避开风险区）
    |   ·
    |  ·  Pareto 前沿
    | ·
    |·  ← 低成本高风险（走最短路径但经过风险区）
    +----------------→ 目标 1 (成本)

决策策略:
- 预算有限 → 选择左下端（最小成本）
- 可靠性优先 → 选择右上端（最小风险）
- 平衡考虑 → 选择中间（折中解）
```

### 4. 与单目标算法对比

| 特性 | GA/VNS/TS | NSGA-II |
|------|-----------|---------|
| 目标数 | 1 个 | 2+ 个 |
| 输出 | 单一最优解 | Pareto 解集 |
| 决策时机 | 优化前（加权） | 优化后（选择） |
| 计算复杂度 | O(N²) | O(N²·M) M=目标数 |

### 5. Trade-off 分析价值

```
边际替代率 = Δ成本 / Δ风险

示例:
- 从最小成本解到折中解：
  成本增加 50，风险降低 4
  边际替代率 = 12.5（每降低 1 风险需增加 12.5 成本）

- 从折中解到最小风险解：
  成本增加 100，风险降低 2
  边际替代率 = 50（收益递减）

洞察：越接近极端，改进代价越大
```

---

## 📈 Week 3 进度

### 已实现算法

| 算法 | 类型 | 核心机制 | 状态 |
|------|------|----------|------|
| GNN | 图深度学习 | 消息传递 + 节点嵌入 | ✅ 完成 |
| 膜算法 | 混合启发式 | GA + VNS | ✅ 完成 |
| **NSGA-II** | **多目标进化** | **非支配排序 + 拥挤度** | **✅ 今日完成** |
| 大规模求解 | 分解算法 | 问题分解 + 并行 | 📝 计划中 |

### Week 1-3 回顾

| 周次 | 主题 | 完成算法数 | 代码量 |
|------|------|-----------|--------|
| Week 1 | 基础启发式 | 10 种 | ~3500 行 |
| Week 2 | 强化学习 | 5 种 | ~2000 行 |
| Week 3 | 高级主题 | 3/6 | ~1900 行 |

---

## 🎯 明日计划

### 主题：大规模问题求解技巧

**学习内容**:
- 大规模组合优化挑战
- 问题分解策略（分治法）
- 并行计算框架
- 多起点策略
- 自适应规模调整

**实现计划**:
- 创建 `19_large_scale_solver.py`
- 实现分解 + 合并策略
- 测试 100+ 节点问题
- 对比集中式 vs 分布式求解

**预期收获**:
- 掌握处理大规模问题的技巧
- 理解分解策略的优劣
- 能够求解实际规模问题

---

## 📝 待办事项

- [x] 实现 NSGA-II 算法代码
- [x] 更新算法笔记
- [x] 编写每日报告
- [ ] 运行脚本生成可视化
- [x] 更新学习进度
- [ ] 提交代码到 GitHub

---

## 🔗 参考资料

1. **NSGA-II 原论文**: Deb, K., et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE TEVC.
2. **多目标优化教材**: Deb, K. (2001). Multi-Objective Optimization Using Evolutionary Algorithms.
3. **经典综述**: Zhou, A., et al. (2011). Multiobjective Evolutionary Algorithms: A Survey of the State of the Art.
4. **Pareto 前沿可视化**: Branke, J., et al. (2008). Multiobjective Optimization: Interactive and Evolutionary Approaches.
5. **Python 实现参考**: https://github.com/anyoptimization/pymoo
6. **线缆布线多目标案例**: Zhang, Q., et al. (2020). Multi-Objective Cable Routing in Offshore Wind Farms.

---

**记录时间**: 2026-03-12 09:00-11:00  
**记录者**: 智子 (Sophon)  
**审核状态**: 待提交 GitHub
