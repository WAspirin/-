# SPT 电缆布线优化 - v2.0 改进计划

**版本**: v2.0  
**日期**: 2026-03-03  
**重点优化**: MILP 求解加速 + 转弯半径约束

---

## 🎯 优化目标

### 1. MILP 求解加速
**当前问题**:
- 大规模问题求解时间长 (>300 秒)
- 分支定界树过大
- 松弛质量差

**优化方案**:
1. **割平面约束 (Cutting Planes)**
   - Subtour elimination constraints
   - Capacity cuts
   - Cover inequalities

2. **Warm Start 策略**
   - 用 SPT 启发式解初始化
   - 提供可行上界

3. **问题预处理**
   - 变量固定（固定明显不选的边）
   - 约束简化
   - 图简化（删除不可能在最优解中的节点）

4. **分解方法**
   - Benders 分解
   - Lagrangian 松弛

### 2. 转弯半径约束
**当前问题**:
- B-Spline 后处理可能导致碰撞
- 转弯半径约束不严格

**优化方案**:
1. **显式 MILP 约束**
   - 角度约束转化为线性约束
   - 禁止急转弯的边组合

2. **改进的 A*启发式**
   - 在寻路时考虑转弯半径
   - Dubins 路径启发式

3. **混合方法**
   - MILP 保证全局最优
   - 后处理保证平滑性

---

## 📝 实现计划

### Phase 1: MILP 加速 (Week 1)

#### 1.1 Warm Start 实现
```python
def milp_with_warm_start(keypoints, path_segments, demands, router, 
                         initial_solution=None, time_limit=300):
    """
    使用启发式解初始化 MILP 求解器
    
    Args:
        initial_solution: 初始可行解 (来自 SPT/MST 启发式)
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    
    # 创建变量
    x, x_k = create_variables(solver, G, num_demands)
    
    # 添加约束
    add_constraints(solver, x, x_k, demands)
    
    # Warm Start: 设置初始解
    if initial_solution is not None:
        for edge in initial_solution['edges']:
            x[edge].SetSolutionValue(1.0)
    
    # 设置求解器参数
    solver.SetTimeLimit(time_limit * 1000)
    solver.SetHint(x, initial_solution)  # SCIP 支持 solution hint
    
    status = solver.Solve()
    return extract_solution(solver, x, demands)
```

**预期效果**: 求解时间减少 30-50%

#### 1.2 割平面约束
```python
def add_subtour_elimination_constraints(solver, x, node_list):
    """
    添加子环消除约束 (Subtour Elimination Constraints)
    
     lazy constraint 形式：
    对于任意节点子集 S，如果 |S| < |V|，则：
    Σ_{i,j ∈ S} x_{ij} ≤ |S| - 1
    """
    # 使用 lazy constraint callback
    # 只在发现子环时添加约束
    pass

def add_capacity_cuts(solver, x, demands, capacities):
    """
    添加容量割平面
    如果某边的流量超过容量，则添加割平面
    """
    pass
```

**预期效果**: 减少分支定界树大小 40-60%

#### 1.3 图预处理
```python
def preprocess_graph(G, demands, cost_grid):
    """
    预处理：简化问题规模
    
    1. 删除不可能在最优解中的边
       - 成本过高的边
       - 远离所有端点的边
    2. 固定某些变量
       - 必经之路的边
    3. 合并相邻节点
    """
    reduced_G = G.copy()
    
    # 1. 删除高成本边
    for u, v, data in G.edges(data=True):
        if data['weight'] > threshold:
            reduced_G.remove_edge(u, v)
    
    # 2. 删除远离端点的节点
    all_terminals = {pt for demand in demands for pt in demand}
    for node in list(G.nodes()):
        min_dist = min(math.hypot(node[0]-t[0], node[1]-t[1]) 
                      for t in all_terminals)
        if min_dist > max_possible_detour:
            reduced_G.remove_node(node)
    
    return reduced_G
```

**预期效果**: 问题规模减少 20-30%

---

### Phase 2: 转弯半径约束 (Week 2)

#### 2.1 显式角度约束
```python
def add_turning_radius_constraints(solver, x, G, min_turning_radius):
    """
    添加转弯半径约束
    
    思路：
    1. 对于每个节点，计算所有入边和出边的夹角
    2. 如果夹角过小（转弯过急），则禁止这对边同时被选中
    
    数学形式：
    对于节点 v 和边 (u,v), (v,w)，如果角度 < θ_min：
    x_{uv} + x_{vw} ≤ 1
    """
    for v in G.nodes():
        for u in G.predecessors(v):
            for w in G.successors(v):
                if u == w: continue
                
                # 计算夹角
                angle = calculate_angle(u, v, w)
                
                # 如果转弯过急，添加约束
                if angle < min_turning_angle:
                    edge_uv = tuple(sorted((u, v)))
                    edge_vw = tuple(sorted((v, w)))
                    solver.Add(x[edge_uv] + x[edge_vw] <= 1)
```

**预期效果**: 保证路径平滑，避免急转弯

#### 2.2 改进的 A*寻路
```python
def astar_with_turning_radius(start, end, G, min_turning_radius):
    """
    A*寻路时考虑转弯半径约束
    
    状态：(node, incoming_direction)
    转移：只允许满足转弯半径的出边
    """
    # 状态空间搜索
    # 每个状态包含：当前位置 + 进入方向
    start_state = (start, None)
    
    open_set = [(0, start_state)]
    came_from = {}
    g_score = {start_state: 0}
    
    while open_set:
        _, (current, in_dir) = heapq.heappop(open_set)
        
        for next_node, out_dir in get_valid_neighbors(current, in_dir, 
                                                       min_turning_radius):
            # 只考虑满足转弯半径的邻居
            if not satisfies_turning_radius(in_dir, out_dir, min_turning_radius):
                continue
            
            # 标准 A*更新
            ...
    
    return reconstruct_path(came_from)

def satisfies_turning_radius(in_dir, out_dir, min_radius):
    """
    检查转弯是否满足最小半径
    
    使用 Dubins 路径理论
    """
    angle = angle_between(in_dir, out_dir)
    # 对于给定速度 v 和最小半径 R
    # 最大角速度 ω = v / R
    # 转弯角度 θ 需要满足一定条件
    return angle <= max_allowed_angle
```

**预期效果**: A*寻路直接生成平滑路径

#### 2.3 混合优化框架
```python
def hybrid_optimization(router, demands, min_turning_radius):
    """
    混合优化框架
    
    流程:
    1. 用改进 A*生成初始解（考虑转弯半径）
    2. 用 MILP 全局优化（添加转弯约束）
    3. 用 B-Spline 后处理（保证光滑）
    """
    # Step 1: 初始解
    initial_paths = []
    for start, end in demands:
        path = astar_with_turning_radius(start, end, router.grid, 
                                         min_turning_radius)
        initial_paths.append(path)
    
    # Step 2: MILP 优化（带转弯约束）
    optimized_paths = milp_with_turning_constraints(
        router, initial_paths, demands, 
        min_turning_radius,
        time_limit=300
    )
    
    # Step 3: B-Spline 平滑
    final_paths = []
    for path in optimized_paths:
        smoothed = bspline_smooth_with_radius_constraint(
            path, router, min_turning_radius
        )
        final_paths.append(smoothed)
    
    return final_paths
```

---

## 📊 预期性能提升

| 优化项 | 当前 | 目标 | 提升 |
|--------|------|------|------|
| MILP 求解时间 | 300s | 100-150s | 50-67% |
| 转弯半径违规 | 10-15% | <1% | 90%+ |
| 路径平滑度 | 中等 | 优秀 | 显著改善 |
| 总电缆长度 | 基准 | -5-10% | 质量提升 |

---

## 🔧 实现时间表

### Week 1 (2026-03-04 ~ 03-10)
- [x] Warm Start 实现
- [ ] 割平面约束
- [ ] 图预处理
- [ ] 性能测试

### Week 2 (2026-03-11 ~ 03-17)
- [ ] 转弯半径约束（MILP）
- [ ] 改进 A*寻路
- [ ] B-Spline 约束增强
- [ ] 集成测试

### Week 3 (2026-03-18 ~ 03-24)
- [ ] 混合优化框架
- [ ] 基准测试
- [ ] 文档完善
- [ ] 论文撰写

---

## 📚 参考资料

### MILP 加速
1. "Valid Inequalities for Mixed Integer Linear Programs" - Wolsey
2. "Branch-and-Cut Algorithms" - Mitchell
3. SCIP 官方文档：https://www.scipopt.org/

### 转弯半径约束
1. "Dubins Paths" - Dubins (1957)
2. "Path Planning with Curvature Constraints" - 相关文献
3. "B-Spline with Constraints" - 数值分析教材

---

**实现者**: 智子 (Sophon)  
**开始日期**: 2026-03-03  
**预计完成**: 2026-03-24

_持续优化，追求卓越！🚀_
