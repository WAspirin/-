# 每日学习报告 - Day 3

**日期**: 2026-03-04  
**学习主题**: 最小生成树 (Minimum Spanning Tree, MST)  
**用时**: ~1.5 小时

---

## ✅ 今日完成内容

### 1. 理论学习 (30 分钟)

- **最小生成树定义与性质**
  - 连通加权无向图的最小权重生成树
  - 包含 V 个顶点和 V-1 条边
  - 连通且无环

- **Prim 算法**
  - 贪心策略：从单顶点逐步扩展
  - 数据结构：优先队列
  - 时间复杂度：O(E log V)
  - 适合稠密图

- **Kruskal 算法**
  - 贪心策略：按边权重排序选择
  - 数据结构：并查集
  - 时间复杂度：O(E log E)
  - 适合稀疏图

- **并查集优化**
  - 路径压缩
  - 按秩合并

### 2. 代码实现 (45 分钟)

**文件**: `examples/07_minimum_spanning_tree.py`

**实现内容**:
- `UnionFind` 类：完整的并查集实现
- `MinimumSpanningTree` 类：
  - `prim()`: Prim 算法实现
  - `kruskal()`: Kruskal 算法实现
  - `visualize()`: 可视化方法
- `generate_cable_network()`: 随机网络生成
- `compare_algorithms()`: 算法对比测试
- `cable_routing_application()`: 园区布线应用案例

**代码特点**:
- 详细的中文注释
- 完整的 docstring
- 可视化输出（原始图 vs MST）
- 实际应用案例（园区网络布线）

### 3. 文档更新 (20 分钟)

- **algorithm-notes.md**: 添加 MST 详细理论笔记
  - Prim 算法流程与复杂度
  - Kruskal 算法流程与复杂度
  - 两种算法对比表
  - 线缆布线应用场景
  - 代码实现要点

- **learning-progress.md**: 更新进度追踪
  - 标记 Week 1 的 MST 任务完成 ✅
  - 添加 Day 3 详细学习记录
  - 更新代码统计

### 4. 代码提交 (10 分钟)

- Git 提交信息：`feat: 实现最小生成树算法 (Prim/Kruskal) - Day 3`
- 提交范围：cable-optimization/examples/, cable-optimization/docs/

---

## 📊 代码统计

| 指标 | 数值 |
|------|------|
| 新增代码文件 | 1 |
| 新增代码行数 | ~450 |
| 实现算法数 | 2 (Prim + Kruskal) |
| 应用案例 | 1 (园区布线) |
| 可视化图表 | 2 (对比图 + 应用图) |

**累计进度**:
| 文件 | 行数 | 说明 |
|------|------|------|
| 01_milp_basic.py | ~180 | MILP 基础 |
| 02_dijkstra.py | ~170 | Dijkstra |
| 03_genetic_algorithm.py | ~240 | 遗传算法 |
| 04_pso.py | ~300 | 粒子群优化 |
| 05_simulated_annealing.py | ~380 | 模拟退火 |
| 06_astar.py | ~450 | A* 搜索 |
| 07_minimum_spanning_tree.py | ~450 | 最小生成树 |
| **总计** | **~2170** | **7 个算法** |

---

## 🤔 遇到的问题

### 问题 1: 理解两种算法的适用场景

**困惑**: Prim 和 Kruskal 都能求 MST，什么时候用哪个？

**解决**:
- **Prim**: 适合稠密图（边多），因为复杂度主要取决于 E
- **Kruskal**: 适合稀疏图（边少），实现更简单
- 实际应用中差别不大，Kruskal 代码更简洁

### 问题 2: 并查集的路径压缩

**困惑**: 路径压缩的具体实现方式？

**解决**:
```python
def find(self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])  # 递归压缩
    return self.parent[x]
```
- 每次 find 操作都将路径上的节点直接连到根
- 摊还时间复杂度接近 O(1)

### 问题 3: 可视化区分 MST 边和所有边

**困惑**: 如何在图中清晰显示 MST 边？

**解决**:
- 所有边用灰色细线（alpha=0.2-0.3）
- MST 边用红色粗线（width=3, alpha=0.8）
- MST 边权重用红色粗体标注

---

## 💡 关键收获

### 理论收获

1. **贪心策略的正确性**
   - MST 问题具有贪心选择性质
   - 局部最优选择能导致全局最优

2. **数据结构的重要性**
   - Prim 需要优先队列高效选边
   - Kruskal 需要并查集高效检测环

3. **算法复杂度分析**
   - 理解 O(E log V) vs O(E log E) 的差异
   - 稀疏图 vs 稠密图的选择

### 实践收获

1. **并查集实现技巧**
   - 路径压缩 + 按秩合并
   - 递归实现简洁高效

2. **图的表示方法**
   - 邻接表适合 Prim
   - 边列表适合 Kruskal

3. **应用建模能力**
   - 将实际问题抽象为图论问题
   - 园区布线→MST 的经典应用

---

## 📈 学习进度

### Week 1 进度 (2026-03-02 ~ 2026-03-09)

| 算法 | 状态 | 代码文件 |
|------|------|----------|
| MILP | ✅ | 01_milp_basic.py |
| Dijkstra | ✅ | 02_dijkstra.py |
| 遗传算法 | ✅ | 03_genetic_algorithm.py |
| PSO | ✅ | 04_pso.py |
| 模拟退火 | ✅ | 05_simulated_annealing.py |
| A* | ✅ | 06_astar.py |
| **最小生成树** | ✅ | **07_minimum_spanning_tree.py** |
| 对比 Notebook | 🔄 | 待创建 |
| VNS | 📝 | 明日计划 |

**Week 1 完成度**: 7/9 (78%)

---

## 🎯 明日计划

### 主要任务

1. **创建算法对比 Notebook**
   - 对比所有已实现算法的性能
   - 在同一问题上测试不同算法
   - 可视化收敛曲线/结果对比

2. **实现变邻域搜索 (VNS)**
   - 学习 VNS 原理
   - 实现完整框架
   - 应用到布线问题

3. **整理 Week 1 学习总结**
   - 汇总 7 种算法
   - 编写对比分析
   - 准备 Week 2 学习

### 时间安排

- 上午 9:00-10:30: 算法对比 Notebook
- 下午灵活时间：VNS 实现 + 周总结

---

## 🔗 相关链接

- **代码**: `cable-optimization/examples/07_minimum_spanning_tree.py`
- **笔记**: `cable-optimization/docs/algorithm-notes.md`
- **进度**: `cable-optimization/docs/learning-progress.md`
- **GitHub**: 待提交

---

**报告人**: 智子 (Sophon)  
**审核**: WonderXi  
**创建时间**: 2026-03-04 09:30
