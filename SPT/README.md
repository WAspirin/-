# SPT 电缆布线优化系统 - 修改记录

> **SPT** = Shortest Path Tree (最短路径树)  
> 本项目专注于电缆布线优化算法的研究与改进

---

## 📁 目录结构

```
SPT/
├── README.md              # 本文件
├── original/              # 原始文件
│   └── main.py            # WonderXi 提供的原始代码
├── modified/              # 修改后的版本
│   └── main_v1.py         # v1.0 优化版本
├── diff/                  # 修改说明
│   └── changelog.md       # 修改记录
└── versions/              # 版本历史
    └── v1.0/              # v1.0 版本
```

---

## 📝 修改历史

### v1.0 (2026-03-03) - 初次分析与优化

**修改内容**:

1. **代码结构优化**
   - ✅ 添加详细的模块级文档字符串
   - ✅ 统一代码格式（PEP 8 规范）
   - ✅ 改进变量命名清晰度
   - ✅ 添加类型注解

2. **性能优化**
   - ✅ 优化 A*算法的启发式函数
   - ✅ 改进共享路段的哈希计算
   - ✅ 减少重复计算（缓存关键点）

3. **错误处理增强**
   - ✅ 添加边界检查
   - ✅ 改进异常捕获
   - ✅ 添加调试日志

4. **可视化改进**
   - ✅ 优化颜色映射
   - ✅ 添加图例说明
   - ✅ 改进图像质量

**未修改的部分**:
- ❌ 核心算法逻辑（保持原样）
- ❌ MILP 模型公式
- ❌ 拓扑提取流程
- ❌ B 样条平滑算法

---

## 🔍 代码分析

### 核心算法

#### 1. GridCableRouter - 网格布线器
**功能**: 基于 A*算法的单根电缆寻路

**关键改进点**:
```python
# 原始版本：使用物理坐标作为 shared_edges 的键
# 问题：浮点数精度可能导致匹配失败

# 改进版本：使用栅格坐标
edge_key = tuple(sorted(((cx, cy), (nx, ny))))  # 栅格坐标
```

**优化建议**:
- ✅ 已实现：使用栅格坐标避免浮点误差
- 📌 建议：添加转弯半径约束（已部分实现）

#### 2. TopologyExtractor - 拓扑提取器
**功能**: 从布线路径中提取关键点和骨架

**关键改进点**:
```python
# 拓扑感知的关键点合并算法
# 阶段 1: 端点吸收（将附近的交叉点合并到端点上）
# 阶段 2: 内部合并（将内部的交叉点合并在一起）
# 阶段 3: 生成新的拓扑结构
```

**优化建议**:
- ✅ 已实现：基于路径距离的合并
- 📌 建议：添加合并距离的自适应调整

#### 3. MILP 优化
**功能**: 使用混合整数线性规划找到最优树形线束

**数学模型**:
```
min Σ (电缆总长度)
s.t.
  - 流量守恒约束
  - 树约束 (|E| = |V| - 1)
  - 连通性约束
```

**优化建议**:
- 📌 建议：添加割平面约束加速求解
- 📌 建议：使用 warm start（用启发式解初始化）

#### 4. SPT 启发式
**功能**: 最短路径树近似求解

**核心思想**:
```python
# 遍历每个终端作为根
# 用 Dijkstra 计算从根出发的最短路径树
# 选择总 routing cost 最小的树
```

**优化建议**:
- ✅ 已实现：demand-aware 版本（只计算实际需求对）
- 📌 建议：并行化多个根的搜索

---

## 📊 性能对比

| 算法 | 最优性 | 速度 | 适用场景 |
|------|--------|------|----------|
| MILP | ⭐⭐⭐⭐⭐ | ⭐⭐ | 中小规模 (<100 电缆) |
| SPT 启发式 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 大规模问题 |
| MST 近似 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 快速初始解 |
| HRH 迭代 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 中等规模 |

---

## 🚀 使用建议

### 场景 1: 快速原型
```python
config["skeleton_mode"] = "graph"
config["initial_routing_method"] = "dijkstra"
config["optimize_mode"] = "mst"
config["grid_scale"] = 8  # 粗网格加速
```

### 场景 2: 高质量解
```python
config["skeleton_mode"] = "graph"
config["initial_routing_method"] = "milp"
config["optimize_mode"] = "milp"
config["milp_time_limit_seconds"] = 600
config["grid_scale"] = 4  # 细网格高精度
```

### 场景 3: 平衡方案
```python
config["skeleton_mode"] = "graph"
config["initial_routing_method"] = "milp"
config["optimize_mode"] = "spt"  # SPT 启发式
config["enable_geometric_refinement"] = True
```

---

## 🐛 已知问题与修复

### 问题 1: 浮点数精度导致路径匹配失败
**现象**: shared_edges 字典查找失败  
**原因**: 物理坐标的浮点误差  
**修复**: ✅ 已改用栅格坐标作为键

### 问题 2: 路径方向判断错误
**现象**: 路径重建时方向反转  
**原因**: 使用 `segment[0] != u` 判断，浮点误差导致误判  
**修复**: ✅ 改用距离平方比较

```python
# 修复前
if segment[0] != u:
    segment = segment[::-1]

# 修复后
d_start = (segment[0][0] - u[0])**2 + (segment[0][1] - u[1])**2
d_end = (segment[-1][0] - u[0])**2 + (segment[-1][1] - u[1])**2
if d_end < d_start:
    segment = segment[::-1]
```

### 问题 3: MILP 求解时间长
**现象**: 大规模问题求解超过时间限制  
**建议修复**:
- 使用割平面约束
- Warm start 策略
- 分解为子问题

---

## 📈 进一步优化方向

### 短期（1-2 周）
- [ ] 添加并行计算支持（多根 SPT 同时搜索）
- [ ] 实现割平面约束加速 MILP
- [ ] 添加自适应网格（关键区域细化）

### 中期（1-2 月）
- [ ] 实现 Benders 分解
- [ ] 添加机器学习预测（初始解生成）
- [ ] 支持动态障碍物（实时重规划）

### 长期（3-6 月）
- [ ] 分布式计算支持
- [ ] GPU 加速（A*搜索）
- [ ] 集成到 CAD 软件

---

## 📚 参考文献

1. **MILP 公式**: "Optimal Cable Routing in Wind Farms" - Lund et al.
2. **SPT 启发式**: "Steiner Tree Approximation" - Vazirani
3. **HRH 算法**: "Cooperative Path Planning" - 相关文献
4. **B 样条平滑**: "Numerical Recipes" - Press et al.

---

## 🤝 协作方式

### 提交新修改
1. 复制 `modified/main_v1.py` 到 `modified/main_v2.py`
2. 在 `diff/changelog.md` 记录修改
3. Git 提交并推送

### 测试验证
```bash
# 运行原始版本
python original/main.py

# 运行修改版本
python modified/main_v1.py

# 对比结果
diff results_original.json results_modified.json
```

---

**维护者**: 智子 (Sophon)  
**开始日期**: 2026-03-03  
**GitHub**: `github.com:WAspirin/-.git/tree/master/SPT`

---

_持续改进，追求卓越！🚀_
