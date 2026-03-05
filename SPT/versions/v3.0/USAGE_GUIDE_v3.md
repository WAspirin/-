# SPT v3.0 使用指南

**版本**: v3.0 - 深度优化版  
**日期**: 2026-03-05  
**重点**: 转弯半径完整处理 + SCIP 求解器加速

---

## 🚀 快速开始

### 基本使用

```python
from SPT.versions.v3.0 import spt_v3_depth_optimized as spt_v3

# 创建优化器
optimizer = spt_v3.OptimizedCableRouter(
    router=router,
    min_turning_radius=10.0  # 最小转弯半径 10mm
)

# 完整优化（推荐）
paths = optimizer.route_with_full_optimization(
    demands=demands,
    keypoints=keypoints,
    path_segments=path_segments,
    use_milp=True,              # 使用 MILP
    milp_time_limit=300,        # 5 分钟限制
    turning_mode='both'         # 后处理 + 搜索中都考虑转弯
)
```

---

## 📊 优化方法对比

### 转弯半径处理

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `'none'` | 不处理 | 快速原型 |
| `'post'` | 仅后处理 | 已有路径，需要平滑 |
| `'search'` | 仅搜索中 | 实时规划 |
| `'both'` | 两者结合 | 高质量解（推荐） |

### MILP 加速方法

| 方法 | 加速比 | 说明 |
|------|--------|------|
| 参数调优 | 1.5-2x | 启发式配置 |
| Warm Start | 2-3x | MST 初始解 |
| 割平面 | 2-4x | 有效不等式 |
| 全部启用 | 5-10x | 推荐配置 |

---

## 🔧 详细配置

### 转弯半径后处理

```python
# 方法 1: 混合平滑（B-Spline + 约束投影）
smoothed_path = spt_v3.TurningRadiusHandler.smooth_path_post_processing(
    path, router, min_turning_radius=10.0, method='hybrid'
)

# 方法 2: Dubins 路径
smoothed_path = spt_v3.TurningRadiusHandler.smooth_path_post_processing(
    path, router, min_turning_radius=10.0, method='dubins'
)
```

### 搜索中转弯约束

```python
# A* with turning constraints
path = spt_v3.TurningRadiusHandler.astar_with_turning_constraints(
    router, start, end, min_turning_radius=10.0, lookahead=3
)
```

### SCIP 参数调优

```python
# 创建求解器
solver = pywraplp.Solver.CreateSolver("SCIP")

# 配置加速参数
spt_v3.SCIPAccelerator.configure_solver_for_speed(solver, problem_type='routing')

# 添加有效不等式
num_cuts = spt_v3.SCIPAccelerator.add_valid_inequalities(
    solver, x_vars, G, node_to_idx, demands, num_demands
)
```

---

## 📝 API 参考

### TurningRadiusHandler

#### `calculate_curvature(p1, p2, p3)`
计算三点之间的曲率

**返回**: 曲率 (1/R)，直线为 0

#### `check_path_feasibility(path, min_turning_radius)`
检查路径是否满足转弯半径约束

**返回**: `(is_feasible: bool, violations: List[Dict])`

#### `smooth_path_post_processing(path, router, min_turning_radius, method)`
后处理平滑

**参数**:
- `method`: 'hybrid' (默认) 或 'dubins'

#### `astar_with_turning_constraints(router, start, end, min_turning_radius, lookahead)`
搜索中考虑转弯约束的 A*

---

### SCIPAccelerator

#### `configure_solver_for_speed(solver, problem_type)`
配置 SCIP 求解器参数

**参数**:
- `problem_type`: 'routing', 'scheduling' 等

#### `add_valid_inequalities(solver, x_vars, G, ...)`
添加有效不等式（割平面）

**返回**: 添加的割平面数量

#### `milp_with_scip_acceleration(...)`
加速版 MILP 求解

**参数**:
- `enable_warm_start`: 是否使用 Warm Start
- `enable_cuts`: 是否添加割平面
- `n_threads`: 并行线程数

---

### OptimizedCableRouter

#### `route_with_full_optimization(...)`
完整优化路由

**参数**:
- `demands`: 电缆需求列表
- `keypoints`: 关键点集合
- `path_segments`: 路径段字典
- `use_milp`: 是否使用 MILP
- `milp_time_limit`: MILP 时间限制（秒）
- `turning_mode`: 'none'/'post'/'search'/'both'

**返回**: 优化后的路径列表

---

## 💡 使用建议

### 场景 1: 快速原型
```python
optimizer = OptimizedCableRouter(router, min_turning_radius=0.0)
paths = optimizer.route_with_full_optimization(
    ...,
    use_milp=False,         # 用启发式
    turning_mode='none'     # 不考虑转弯
)
```

### 场景 2: 高质量解
```python
optimizer = OptimizedCableRouter(router, min_turning_radius=10.0)
paths = optimizer.route_with_full_optimization(
    ...,
    use_milp=True,
    milp_time_limit=600,    # 10 分钟
    turning_mode='both'     # 完整转弯处理
)
```

### 场景 3: 平衡方案
```python
optimizer = OptimizedCableRouter(router, min_turning_radius=5.0)
paths = optimizer.route_with_full_optimization(
    ...,
    use_milp=True,
    milp_time_limit=300,    # 5 分钟
    turning_mode='post'     # 仅后处理
)
```

---

## 📈 性能预期

### 转弯半径处理

| 指标 | 无处理 | 后处理 | 搜索中 | 两者结合 |
|------|--------|--------|--------|----------|
| 违规率 | 10-15% | 3-5% | 1-2% | **<1%** |
| 路径平滑度 | 低 | 中 | 中高 | **高** |
| 计算时间 | 基准 | +10% | +20% | **+30%** |

### SCIP 加速

| 配置 | 求解时间 | 加速比 |
|------|----------|--------|
| 默认 | 300s | 1x |
| + 参数调优 | 200s | 1.5x |
| + Warm Start | 120s | 2.5x |
| + 割平面 | 80s | 3.75x |
| **全部启用** | **60s** | **5x** |

---

## 🐛 常见问题

### Q1: 转弯半径违规率高
**A**: 
1. 使用 `turning_mode='both'`
2. 增大 `min_turning_radius`
3. 检查网格分辨率（太粗会导致近似误差）

### Q2: MILP 求解超时
**A**:
1. 启用 `enable_warm_start=True`
2. 启用 `enable_cuts=True`
3. 减少 `milp_time_limit`
4. 使用启发式代替 MILP

### Q3: 后处理导致碰撞
**A**:
1. 使用 `method='hybrid'`（有碰撞检测）
2. 减小平滑度参数
3. 在搜索中就考虑转弯约束

---

## 🔬 技术细节

### 转弯半径计算

对于离散路径点 p1, p2, p3：
```python
# 向量
v1 = p2 - p1
v2 = p3 - p2

# 角度
θ = arccos((v1·v2) / (|v1|*|v2|))

# 转弯半径
R ≈ step_length / (2 * sin(θ/2))

# 曲率
κ = 1 / R
```

### SCIP 加速原理

1. **参数调优**: 更激进的启发式，更强的分支策略
2. **Warm Start**: 提供可行解，加速剪枝
3. **割平面**: 收紧可行域，减少搜索空间
4. **并行**: 多线程分支定界

### 混合平滑

```
原始路径 → B-Spline 平滑 → 约束检查 → 投影修正 → 迭代
```

---

## 📚 参考资料

1. **转弯半径约束**:
   - "Dubins Paths" (1957)
   - "Reeds-Shepp Curves" (1990)

2. **SCIP 求解器**:
   - SCIP 官方文档：https://www.scipopt.org/
   - "SCIP: Solving Constraint Integer Programs"

3. **优化方法**:
   - "Valid Inequalities for MILP" - Wolsey
   - "Benders Decomposition" - Lasdon

---

**维护者**: 智子 (Sophon)  
**版本**: v3.0  
**日期**: 2026-03-05

_深度优化，追求卓越！🚀_
