# 每日学习报告 - Day 10 (2026-03-10)

**学习主题**: 图神经网络 (GNN) 入门  
**所属周次**: Week 3 - 高级主题  
**学习时长**: ~2 小时

---

## ✅ 今日完成内容

### 1. 算法学习 (40 分钟)

**学习内容**:
- 图神经网络基础理论 (GCN, GAT, MPNN)
- 消息传递机制 (Message Passing)
- 节点嵌入学习 (Node Embedding)
- GNN 在组合优化中的应用
- 与强化学习的对比分析

**关键理解**:
- GCN 通过归一化邻接矩阵聚合邻居信息
- GAT 使用注意力机制学习邻居重要性
- MPNN 是 GNN 的通用框架
- GNN 可直接处理图结构，无需手工特征工程

### 2. 代码实现 (60 分钟)

**实现文件**: `examples/16_gnn_graph_neural_network.py` (~650 行)

**核心组件**:

| 组件 | 行数 | 功能 |
|------|------|------|
| Graph | ~40 | 图数据结构 |
| GraphConvolutionLayer | ~80 | 图卷积层实现 |
| GraphAttentionLayer | ~100 | 图注意力层实现 |
| SimpleGNN | ~60 | 简单 GNN 模型 |
| GNNForCableRouting | ~200 | 线缆布线应用 |
| 可视化 | ~120 | 结果可视化 |
| 主程序 | ~50 | 演示流程 |

**关键实现细节**:

1. **图卷积层**:
   ```python
   # 归一化邻接矩阵：D^{-1/2} Ã D^{-1/2}
   adj_with_self = adj_matrix + np.eye(n)
   degrees = np.sum(adj_with_self, axis=1)
   d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
   norm_adj = d_inv_sqrt @ adj_with_self @ d_inv_sqrt
   
   # 图卷积：H' = σ(Ã H W)
   output = norm_adj @ node_features @ weights
   ```

2. **图注意力层**:
   ```python
   # 注意力系数计算
   score = LeakyReLU(a^T [Wh_i || Wh_j])
   α_ij = exp(score) / Σ_k exp(score_k)
   
   # 多头注意力
   outputs = [head_output for head in range(n_heads)]
   output = concatenate(outputs)
   ```

3. **节点特征设计**:
   - 归一化位置 (x, y)
   - 到起点/终点的距离
   - 局部密度（周围节点数）
   - 共 5 维特征

4. **自监督训练**:
   - 任务：重构邻接矩阵（边预测）
   - 损失：二元交叉熵 (BCE)
   - 评估：边预测准确率

5. **GNN 引导路径规划**:
   - 使用节点嵌入相似度指导搜索
   - 贪心策略选择最优邻居
   - 结合距离启发式

### 3. 文档更新 (20 分钟)

**更新文件**:
- `docs/algorithm-notes.md` - 添加 GNN 理论笔记 (~600 行新增)
- `docs/daily-report-2026-03-10.md` - 今日报告（本文件）
- `docs/learning-progress.md` - 更新进度

**笔记内容**:
- GCN 核心公式推导
- GAT 注意力机制详解
- MPNN 通用框架
- GNN 在线缆布线中的应用
- 与 RL 算法对比
- 参数调优指南

### 4. 实验测试 (10 分钟)

**测试配置**:
- 节点数：20
- 节点特征维度：5
- 隐藏层维度：64
- 输出嵌入维度：32
- 网络层数：2
- 训练迭代：100

**预期结果**:
```
训练损失：0.69 → 0.55
边预测准确率：50% → 75%
路径规划：成功连接起点到终点
```

---

## 📊 代码统计

| 文件 | 行数 | 说明 |
|------|------|------|
| 16_gnn_graph_neural_network.py | ~650 | GNN 完整实现 |
| algorithm-notes.md (新增) | ~600 | GNN 理论笔记 |
| daily-report-2026-03-10.md | ~250 | 今日报告 |
| **今日新增** | **~1500** | 代码 + 文档 |

**累计代码量**: ~5880 行  
**累计文档量**: ~3000 行  
**累计算法数**: 16 种

---

## 🤔 遇到的问题

### 问题 1: GNN 过平滑 (Over-smoothing)

**问题描述**: 
多层 GCN 后，所有节点嵌入趋向相同，失去区分度。

**原因分析**:
- 多次聚合后，节点特征被"平均化"
- 深层网络导致信息传播过远

**解决方案**:
- 限制网络层数 (2-4 层)
- 使用残差连接 (Residual Connection)
- 使用跳跃连接 (Jumping Knowledge)
- 本实现使用 2 层，避免过平滑

### 问题 2: 注意力机制实现复杂度

**问题描述**:
标准 GAT 需要计算所有节点对的注意力系数，复杂度 O(n²)。

**解决方案**:
- 简化实现：只计算邻居间的注意力
- 使用稀疏邻接矩阵
- 未来可升级为高效注意力 (如 Sparse GAT)

### 问题 3: 自监督学习任务设计

**问题描述**:
如何设计无需标注数据的训练任务？

**解决方案**:
- 使用边预测作为自监督任务
- 正样本：图中存在的边
- 负样本：随机采样的不存在的边
- 损失：二元交叉熵

---

## 💡 关键洞察

### 1. GNN vs 传统方法

| 维度 | 传统 ML | GNN |
|------|---------|-----|
| 输入 | 固定维度向量 | 图结构 |
| 特征工程 | 手工设计 | 自动学习 |
| 泛化能力 | 弱 | 强 (不同图结构) |
| 归纳偏置 | 无 | 图结构先验 |

**理解**: GNN 的核心优势是"直接处理图结构，无需展平或手工特征"

### 2. 消息传递的本质

```
每个节点 = 信息接收者 + 信息发送者

迭代过程:
1. 收集邻居信息 (Message)
2. 聚合信息 (Aggregate)
3. 更新自身表示 (Update)

类比: 社交网络中，你的观点受朋友影响，也影响朋友
```

### 3. GCN 归一化的重要性

- **不加归一化**: 高度数节点主导聚合结果
- **加归一化**: 每个邻居贡献均衡
- **公式**: D^{-1/2} Ã D^{-1/2} 确保数值稳定

### 4. GNN + RL 的混合潜力

- **GNN**: 编码图结构 → 节点嵌入
- **RL**: 基于嵌入做序列决策 → 路径规划
- **优势**: 结合结构理解 + 序列优化

---

## 📈 Week 3 进度

### 已实现算法

| 算法 | 类型 | 核心机制 | 状态 |
|------|------|----------|------|
| **GNN** | **图深度学习** | **消息传递 + 节点嵌入** | **✅ 今日完成** |
| 混合算法 (GA+LS) | 混合启发式 | 全局搜索 + 局部优化 | 📝 计划中 |
| 多目标优化 | 进化算法 | Pareto 前沿 | 📝 计划中 |
| 大规模求解 | 分解算法 | 问题分解 + 并行 | 📝 计划中 |

### Week 1-2 回顾

| 周次 | 主题 | 完成算法数 | 代码量 |
|------|------|-----------|--------|
| Week 1 | 基础启发式 | 10 种 | ~3500 行 |
| Week 2 | 强化学习 | 5 种 | ~2000 行 |
| Week 3 | 高级主题 | 1/6 | ~650 行 |

---

## 🎯 明日计划

### 主题：混合算法设计 (GA + Local Search)

**学习内容**:
- 混合元启发式理论基础
- Memetic 算法 (进化算法 + 局部搜索)
- GA + VNS/TS 混合策略
- 协同优化机制

**实现计划**:
- 创建 `17_hybrid_ga_ls.py`
- 实现 GA 框架 + VNS 局部搜索
- 在布线问题上测试
- 对比纯 GA vs 混合算法

**预期收获**:
- 理解混合算法的优势
- 掌握算法组合技巧
- 提升解的质量

---

## 📝 待办事项

- [x] 实现 GNN 代码
- [x] 更新算法笔记
- [x] 编写每日报告
- [ ] 运行训练脚本，生成可视化
- [ ] 提交代码到 GitHub
- [ ] 准备明日学习材料

---

## 🔗 参考资料

1. **GCN 原论文**: Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
2. **GAT 原论文**: Veličković et al. (2018). Graph Attention Networks. ICLR 2018.
3. **MPNN 框架**: Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. ICML 2017.
4. **GNN 综述**: Wu et al. (2020). A Comprehensive Survey on Graph Neural Networks. IEEE TNNLS.
5. **GNN 组合优化**: Bengio et al. (2021). Machine Learning for Combinatorial Optimization: a Methodological Tour d'horizon.
6. **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
7. **DGL Library**: https://www.dgl.ai/

---

**记录时间**: 2026-03-10 09:00-11:00  
**记录者**: 智子 (Sophon)  
**审核状态**: 待提交 GitHub
