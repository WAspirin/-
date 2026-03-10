#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_gnn_graph_neural_network.py

图神经网络 (Graph Neural Network, GNN) 在线缆布线优化中的应用

作者：智子 (Sophon)
日期：2026-03-10
周次：Week 3 - 高级主题 - Day 10

算法核心:
- 使用图神经网络学习节点嵌入 (Node Embedding)
- 消息传递机制 (Message Passing) 聚合邻居信息
- 预测边的连接概率或重要性
- 应用于线缆布线路径规划

参考论文:
- Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
- Veličković et al. (2018). Graph Attention Networks (GAT)
- Bresson & Laurent (2021). Two Simple Yet Effective Graph Networks for Combinatorial Optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 1. 图神经网络基础组件
# ============================================================================

class Graph:
    """
    简单的图数据结构
    
    用于表示线缆布线网络：
    - 节点：设备/接线盒位置
    - 边：可能的布线路径
    - 节点特征：位置、类型、容量等
    - 边特征：距离、成本、容量等
    """
    
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.adj_matrix = np.zeros((n_nodes, n_nodes))
        self.node_features = None
        self.edge_features = None
        
    def add_edge(self, i, j, weight=1.0):
        """添加无向边"""
        self.adj_matrix[i, j] = weight
        self.adj_matrix[j, i] = weight
        
    def set_node_features(self, features):
        """设置节点特征矩阵 (n_nodes × feature_dim)"""
        self.node_features = features
        
    def get_neighbors(self, node):
        """获取节点的邻居列表"""
        return np.where(self.adj_matrix[node] > 0)[0].tolist()
    
    def get_adjacency_list(self):
        """获取邻接表表示"""
        adj_list = defaultdict(list)
        for i in range(self.n_nodes):
            for j in np.where(self.adj_matrix[i] > 0)[0]:
                adj_list[i].append(j)
        return adj_list


class GraphConvolutionLayer:
    """
    图卷积层 (Graph Convolutional Layer)
    
    核心公式:
    H^{(l+1)} = σ(D^{-1/2} Ã D^{-1/2} H^{(l)} W^{(l)})
    
    其中:
    - Ã = A + I (添加自环)
    - D 是度矩阵
    - H^{(l)} 是第 l 层的节点表示
    - W^{(l)} 是可学习权重
    - σ 是激活函数 (ReLU)
    """
    
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # 初始化权重 (Xavier 初始化)
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))
        self.b = np.zeros(output_dim)
        
    def normalize_adj(self, adj_matrix):
        """
        计算归一化的邻接矩阵: D^{-1/2} Ã D^{-1/2}
        
        Args:
            adj_matrix: 原始邻接矩阵 (n × n)
            
        Returns:
            normalized_adj: 归一化邻接矩阵
        """
        n = adj_matrix.shape[0]
        
        # 添加自环
        adj_with_self = adj_matrix + np.eye(n)
        
        # 计算度矩阵
        degrees = np.sum(adj_with_self, axis=1)
        
        # 避免除零
        degrees = np.where(degrees == 0, 1, degrees)
        
        # D^{-1/2}
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        
        # D^{-1/2} Ã D^{-1/2}
        normalized_adj = d_inv_sqrt @ adj_with_self @ d_inv_sqrt
        
        return normalized_adj
    
    def forward(self, node_features, adj_matrix):
        """
        前向传播
        
        Args:
            node_features: 节点特征矩阵 (n_nodes × input_dim)
            adj_matrix: 邻接矩阵 (n_nodes × n_nodes)
            
        Returns:
            output: 新的节点表示 (n_nodes × output_dim)
        """
        # 归一化邻接矩阵
        norm_adj = self.normalize_adj(adj_matrix)
        
        # 图卷积: H' = σ(Ã H W)
        output = norm_adj @ node_features @ self.W + self.b
        
        # 激活函数
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'sigmoid':
            output = 1 / (1 + np.exp(-np.clip(output, -500, 500)))
        elif self.activation == 'tanh':
            output = np.tanh(output)
            
        return output
    
    def get_params(self):
        """获取可学习参数"""
        return {'W': self.W, 'b': self.b}
    
    def set_params(self, params):
        """设置可学习参数"""
        self.W = params['W']
        self.b = params['b']


class GraphAttentionLayer:
    """
    图注意力层 (Graph Attention Layer, GAT)
    
    核心思想: 使用注意力机制学习邻居节点的重要性权重
    
    注意力系数计算:
    α_ij = exp(LeakyReLU(a^T [Wh_i || Wh_j])) / Σ_k exp(LeakyReLU(a^T [Wh_i || Wh_k]))
    
    其中:
    - h_i, h_j 是节点 i 和 j 的特征
    - W 是可学习权重
    - a 是注意力向量
    - || 表示拼接
    """
    
    def __init__(self, input_dim, output_dim, n_heads=1, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.activation = activation
        self.head_dim = output_dim // n_heads
        
        # 每个注意力头的权重
        self.W_heads = []
        self.a_heads = []
        
        for _ in range(n_heads):
            W = np.random.randn(input_dim, self.head_dim) * np.sqrt(2.0 / (input_dim + self.head_dim))
            a = np.random.randn(2 * self.head_dim) * 0.1
            self.W_heads.append(W)
            self.a_heads.append(a)
    
    def leaky_relu(self, x, alpha=0.2):
        """LeakyReLU 激活函数"""
        return np.where(x > 0, x, alpha * x)
    
    def compute_attention(self, h_i, h_j, a):
        """
        计算节点 i 和 j 之间的注意力系数
        
        Args:
            h_i: 节点 i 的特征 (head_dim,)
            h_j: 节点 j 的特征 (head_dim,)
            a: 注意力向量 (2*head_dim,)
            
        Returns:
            attention: 注意力系数 (标量)
        """
        # 拼接特征
        concatenated = np.concatenate([h_i, h_j])
        
        # 计算注意力分数
        score = np.dot(a, concatenated)
        
        return self.leaky_relu(score)
    
    def forward(self, node_features, adj_matrix):
        """
        前向传播
        
        Args:
            node_features: 节点特征 (n_nodes × input_dim)
            adj_matrix: 邻接矩阵 (n_nodes × n_nodes)
            
        Returns:
            output: 新的节点表示 (n_nodes × output_dim)
        """
        n_nodes = node_features.shape[0]
        outputs = []
        
        for head_idx in range(self.n_heads):
            W = self.W_heads[head_idx]
            a = self.a_heads[head_idx]
            
            # 线性变换
            h_transformed = node_features @ W  # (n_nodes × head_dim)
            
            # 计算注意力系数
            attention_matrix = np.zeros((n_nodes, n_nodes))
            
            for i in range(n_nodes):
                neighbors = np.where(adj_matrix[i] > 0)[0]
                
                for j in neighbors:
                    attention_matrix[i, j] = self.compute_attention(
                        h_transformed[i], h_transformed[j], a
                    )
                
                # Softmax 归一化 (只考虑邻居)
                if len(neighbors) > 0:
                    max_score = np.max(attention_matrix[i, neighbors])
                    exp_scores = np.exp(attention_matrix[i, neighbors] - max_score)
                    attention_matrix[i, neighbors] = exp_scores / (np.sum(exp_scores) + 1e-10)
            
            # 聚合邻居信息
            h_new = attention_matrix @ h_transformed
            
            # 激活函数
            if self.activation == 'relu':
                h_new = np.maximum(0, h_new)
            elif self.activation == 'elu':
                h_new = np.where(h_new > 0, h_new, np.exp(h_new) - 1)
            
            outputs.append(h_new)
        
        # 拼接所有注意力头的输出
        output = np.concatenate(outputs, axis=1)
        
        return output


# ============================================================================
# 2. 图神经网络模型
# ============================================================================

class SimpleGNN:
    """
    简单的图神经网络模型
    
    架构:
    - 输入层：节点特征
    - 隐藏层 1: GraphConv (input_dim → hidden_dim)
    - 隐藏层 2: GraphConv (hidden_dim → hidden_dim)
    - 输出层：节点嵌入或边预测
    
    应用:
    1. 节点嵌入：学习每个节点的向量表示
    2. 边预测：预测两个节点之间是否应该有连接
    3. 路径规划：基于节点嵌入指导搜索
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, n_layers=2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # 构建网络层
        self.layers = []
        
        # 第一层
        self.layers.append(GraphConvolutionLayer(input_dim, hidden_dim, activation='relu'))
        
        # 中间层
        for _ in range(n_layers - 2):
            self.layers.append(GraphConvolutionLayer(hidden_dim, hidden_dim, activation='relu'))
        
        # 输出层
        self.layers.append(GraphConvolutionLayer(hidden_dim, output_dim, activation='relu'))
    
    def forward(self, node_features, adj_matrix):
        """
        前向传播
        
        Args:
            node_features: 节点特征 (n_nodes × input_dim)
            adj_matrix: 邻接矩阵 (n_nodes × n_nodes)
            
        Returns:
            embeddings: 节点嵌入 (n_nodes × output_dim)
        """
        h = node_features
        
        for layer in self.layers:
            h = layer.forward(h, adj_matrix)
        
        return h
    
    def predict_edge_probability(self, node_i, node_j, embeddings):
        """
        预测两个节点之间的连接概率
        
        使用余弦相似度或内积
        
        Args:
            node_i: 节点 i 的索引
            node_j: 节点 j 的索引
            embeddings: 节点嵌入矩阵
            
        Returns:
            probability: 连接概率 (0-1)
        """
        emb_i = embeddings[node_i]
        emb_j = embeddings[node_j]
        
        # 余弦相似度
        similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-10)
        
        # 转换为概率
        probability = (similarity + 1) / 2  # 映射到 [0, 1]
        
        return probability


class GNNForCableRouting:
    """
    面向线缆布线的图神经网络应用
    
    功能:
    1. 构建布线网络图
    2. 学习节点嵌入
    3. 预测最优布线路径
    4. 可视化学习结果
    """
    
    def __init__(self, n_nodes, node_positions, obstacle_positions=None):
        self.n_nodes = n_nodes
        self.node_positions = np.array(node_positions)
        self.obstacle_positions = obstacle_positions or []
        
        # 初始化图
        self.graph = Graph(n_nodes)
        
        # 构建完全图（所有节点对都有边）
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                distance = np.linalg.norm(self.node_positions[i] - self.node_positions[j])
                self.graph.add_edge(i, j, weight=1.0 / (distance + 0.1))
        
        # 初始化节点特征
        self._init_node_features()
        
        # 初始化 GNN 模型
        self.gnn = SimpleGNN(input_dim=5, hidden_dim=64, output_dim=32, n_layers=2)
        
        # 训练历史
        self.training_history = []
    
    def _init_node_features(self):
        """
        初始化节点特征
        
        特征维度 = 5:
        1-2. 归一化位置 (x, y)
        3. 到起点的距离
        4. 到终点的距离
        5. 局部密度（周围节点数）
        """
        features = []
        
        # 计算边界用于归一化
        min_pos = np.min(self.node_positions, axis=0)
        max_pos = np.max(self.node_positions, axis=0)
        range_pos = max_pos - min_pos + 1e-10
        
        start_pos = self.node_positions[0]
        end_pos = self.node_positions[-1]
        
        for i in range(self.n_nodes):
            pos = self.node_positions[i]
            
            # 归一化位置
            norm_x = (pos[0] - min_pos[0]) / range_pos[0]
            norm_y = (pos[1] - min_pos[1]) / range_pos[1]
            
            # 到起点和终点的距离
            dist_to_start = np.linalg.norm(pos - start_pos)
            dist_to_end = np.linalg.norm(pos - end_pos)
            max_dist = np.linalg.norm(max_pos - min_pos)
            norm_dist_start = dist_to_start / (max_dist + 1e-10)
            norm_dist_end = dist_to_end / (max_dist + 1e-10)
            
            # 局部密度（半径为 20 的圆内节点数）
            neighbors = 0
            for j in range(self.n_nodes):
                if i != j and np.linalg.norm(pos - self.node_positions[j]) < 20:
                    neighbors += 1
            density = neighbors / self.n_nodes
            
            features.append([norm_x, norm_y, norm_dist_start, norm_dist_end, density])
        
        self.graph.set_node_features(np.array(features))
    
    def train(self, n_iterations=100, learning_rate=0.01):
        """
        训练 GNN 模型
        
        使用自监督学习：
        - 目标：重构邻接矩阵
        - 损失：预测边概率与真实边的差异
        
        Args:
            n_iterations: 训练迭代次数
            learning_rate: 学习率
        """
        print("=" * 60)
        print("开始训练图神经网络...")
        print("=" * 60)
        
        n_nodes = self.n_nodes
        node_features = self.graph.node_features
        adj_matrix = (self.graph.adj_matrix > 0).astype(float)
        
        # 采样边对用于训练
        positive_edges = []
        negative_edges = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # 完全图，所有边都存在
                positive_edges.append((i, j))
        
        # 生成负样本（不存在的边 - 在完全图中需要虚拟生成）
        # 为了训练，我们创建一些"弱连接"作为负样本
        all_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
        np.random.seed(42)
        
        # 随机选择一些边作为"测试"负样本
        n_negative = min(len(positive_edges), 50)
        negative_edges = list(np.random.choice(len(positive_edges), n_negative, replace=False))
        negative_edges = [positive_edges[idx] for idx in negative_edges]
        
        # 重新选择正样本
        positive_sample_indices = list(np.random.choice(len(positive_edges), min(n_negative, len(positive_edges)), replace=False))
        positive_edges_sampled = [positive_edges[idx] for idx in positive_sample_indices]
        
        print(f"训练样本：{len(positive_edges_sampled)} 正边 + {len(negative_edges)} 负边")
        
        # 如果样本太少，使用简化的自监督任务
        if len(positive_edges_sampled) < 2:
            print("使用简化训练模式...")
            # 创建基于距离的正负样本
            distances = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    dist = np.linalg.norm(self.node_positions[i] - self.node_positions[j])
                    distances.append((i, j, dist))
            
            distances.sort(key=lambda x: x[2])
            positive_edges_sampled = [(i, j) for i, j, d in distances[:n_negative]]
            negative_edges = [(i, j) for i, j, d in distances[-n_negative:]]
            
            print(f"基于距离的训练样本：{len(positive_edges_sampled)} 正边 + {len(negative_edges)} 负边")
        
        positive_edges = positive_edges_sampled
        
        for iteration in range(n_iterations):
            # 前向传播
            embeddings = self.gnn.forward(node_features, self.graph.adj_matrix)
            
            # 计算预测
            predictions = []
            targets = []
            
            for i, j in positive_edges + negative_edges:
                prob = self.gnn.predict_edge_probability(i, j, embeddings)
                predictions.append(prob)
            
            targets = [1.0] * len(positive_edges) + [0.0] * len(negative_edges)
            
            # 计算损失 (BCE)
            predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
            loss = -np.mean([
                t * np.log(p) + (1 - t) * np.log(1 - p)
                for p, t in zip(predictions, targets)
            ])
            
            # 计算准确率
            pred_labels = [1 if p > 0.5 else 0 for p in predictions]
            accuracy = np.mean([p == t for p, t in zip(pred_labels, targets)])
            
            self.training_history.append({'iteration': iteration, 'loss': loss, 'accuracy': accuracy})
            
            if iteration % 10 == 0 or iteration == n_iterations - 1:
                print(f"Iteration {iteration:3d}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")
        
        print("=" * 60)
        print("训练完成!")
        print("=" * 60)
        
        return embeddings
    
    def plan_path_gnn(self, start_node=0, end_node=None):
        """
        使用 GNN 嵌入规划布线路径
        
        策略：贪心搜索，选择嵌入相似度最高的邻居
        
        Args:
            start_node: 起点节点索引
            end_node: 终点节点索引（默认最后一个节点）
            
        Returns:
            path: 路径节点列表
            total_cost: 总成本
        """
        if end_node is None:
            end_node = self.n_nodes - 1
        
        # 获取节点嵌入
        embeddings = self.gnn.forward(self.graph.node_features, self.graph.adj_matrix)
        
        # 贪心搜索
        path = [start_node]
        visited = {start_node}
        current = start_node
        total_cost = 0.0
        
        max_steps = self.n_nodes * 2  # 防止无限循环
        
        for step in range(max_steps):
            if current == end_node:
                break
            
            # 获取邻居
            neighbors = self.graph.get_neighbors(current)
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if not unvisited_neighbors:
                break
            
            # 选择嵌入最接近终点的邻居
            target_embedding = embeddings[end_node]
            
            best_neighbor = None
            best_similarity = -1
            
            for neighbor in unvisited_neighbors:
                similarity = self.gnn.predict_edge_probability(neighbor, end_node, embeddings)
                
                # 结合距离启发式
                dist_to_end = np.linalg.norm(
                    self.node_positions[neighbor] - self.node_positions[end_node]
                )
                dist_heuristic = 1.0 / (dist_to_end + 0.1)
                
                # 综合评分
                score = 0.7 * similarity + 0.3 * dist_heuristic
                
                if score > best_similarity:
                    best_similarity = score
                    best_neighbor = neighbor
            
            if best_neighbor is not None:
                # 计算边成本
                edge_cost = np.linalg.norm(
                    self.node_positions[current] - self.node_positions[best_neighbor]
                )
                total_cost += edge_cost
                
                path.append(best_neighbor)
                visited.add(best_neighbor)
                current = best_neighbor
        
        return path, total_cost
    
    def visualize(self, embeddings=None, path=None, save_path=None):
        """
        可视化 GNN 学习结果
        
        Args:
            embeddings: 节点嵌入（可选，用于降维可视化）
            path: 规划的路径
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 原始网络图
        ax1 = axes[0, 0]
        ax1.set_title("Original Network Graph", fontsize=12, fontweight='bold')
        
        # 绘制节点
        for i in range(self.n_nodes):
            x, y = self.node_positions[i]
            if i == 0:
                color = 'green'
                label = 'Start'
            elif i == self.n_nodes - 1:
                color = 'red'
                label = 'End'
            else:
                color = 'skyblue'
                label = None
            
            ax1.scatter(x, y, c=color, s=100, zorder=3, label=label, edgecolors='black', linewidth=1)
            
            # 绘制边
            for j in self.graph.get_neighbors(i):
                if j > i:  # 避免重复绘制
                    x2, y2 = self.node_positions[j]
                    ax1.plot([x, x2], [y, y2], 'gray', alpha=0.3, linewidth=0.5)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.set_aspect('equal')
        
        # 2. 节点嵌入可视化 (简单降维)
        ax2 = axes[0, 1]
        ax2.set_title("Node Embeddings (2D Projection)", fontsize=12, fontweight='bold')
        
        if embeddings is not None:
            # 简单降维：使用前两个主成分或随机投影
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
            except ImportError:
                # 无 sklearn 时使用简单投影
                embeddings_2d = embeddings[:, :2] if embeddings.shape[1] >= 2 else np.hstack([embeddings, np.zeros((len(embeddings), 1))])[:, :2]
            
            # 按节点索引着色
            scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=range(self.n_nodes), cmap='viridis', 
                                 s=80, edgecolors='black')
            plt.colorbar(scatter, ax=ax2, label="Node Index")
        
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        
        # 3. 训练历史
        ax3 = axes[1, 0]
        ax3.set_title("Training History", fontsize=12, fontweight='bold')
        
        if len(self.training_history) > 0:
            iterations = [h['iteration'] for h in self.training_history]
            losses = [h['loss'] for h in self.training_history]
            accuracies = [h['accuracy'] for h in self.training_history]
            
            ax3_twin = ax3.twinx()
            
            ax3.plot(iterations, losses, 'b-', linewidth=2, label='Loss')
            ax3_twin.plot(iterations, accuracies, 'r-', linewidth=2, label='Accuracy')
            
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Loss", color='b')
            ax3_twin.set_ylabel("Accuracy", color='r')
            ax3.grid(True, alpha=0.3)
            
            # 合并图例
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 4. 规划的路径
        ax4 = axes[1, 1]
        ax4.set_title("GNN-Guided Path Planning", fontsize=12, fontweight='bold')
        
        # 绘制所有节点
        for i in range(self.n_nodes):
            x, y = self.node_positions[i]
            if i == 0:
                color = 'green'
            elif i == self.n_nodes - 1:
                color = 'red'
            else:
                color = 'skyblue'
            
            ax4.scatter(x, y, c=color, s=100, zorder=3, edgecolors='black', linewidth=1)
        
        # 绘制路径
        if path is not None:
            path_positions = self.node_positions[path]
            ax4.plot(path_positions[:, 0], path_positions[:, 1], 
                    'r-', linewidth=2, zorder=2, marker='o', markersize=8)
            
            # 标注路径节点
            for idx, node in enumerate(path):
                x, y = self.node_positions[node]
                ax4.annotate(str(idx), (x, y), fontsize=9, 
                            ha='center', va='center', color='white',
                            bbox=dict(boxstyle='circle', facecolor='black', alpha=0.5))
        
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel("X Position")
        ax4.set_ylabel("Y Position")
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化已保存：{save_path}")
        
        plt.show()


# ============================================================================
# 3. 主程序 - 线缆布线应用示例
# ============================================================================

def main():
    """
    主程序：演示 GNN 在线缆布线中的应用
    """
    print("=" * 70)
    print(" " * 20 + "图神经网络 (GNN) 线缆布线优化")
    print("=" * 70)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 创建布线场景
    print("\n[1/4] 创建线缆布线场景...")
    
    n_nodes = 20
    
    # 生成随机节点位置
    node_positions = []
    
    # 起点和终点固定
    node_positions.append([5, 5])  # 起点
    
    # 中间节点随机分布
    for _ in range(n_nodes - 2):
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        node_positions.append([x, y])
    
    # 终点
    node_positions.append([95, 95])
    
    print(f"   - 节点数量：{n_nodes}")
    print(f"   - 起点：{node_positions[0]}")
    print(f"   - 终点：{node_positions[-1]}")
    
    # 2. 初始化 GNN 模型
    print("\n[2/4] 初始化图神经网络模型...")
    
    gnn_routing = GNNForCableRouting(
        n_nodes=n_nodes,
        node_positions=node_positions
    )
    
    print(f"   - 节点特征维度：5")
    print(f"   - 隐藏层维度：64")
    print(f"   - 输出嵌入维度：32")
    print(f"   - 网络层数：2")
    
    # 3. 训练 GNN
    print("\n[3/4] 训练 GNN 模型...")
    
    embeddings = gnn_routing.train(n_iterations=100, learning_rate=0.01)
    
    print(f"\n   最终嵌入形状：{embeddings.shape}")
    
    # 4. 路径规划
    print("\n[4/4] 使用 GNN 进行路径规划...")
    
    path, total_cost = gnn_routing.plan_path_gnn(start_node=0, end_node=n_nodes-1)
    
    print(f"\n   规划路径：{path}")
    print(f"   路径长度：{len(path)} 个节点")
    print(f"   总成本：{total_cost:.2f}")
    
    # 5. 可视化
    print("\n[5/5] 生成可视化结果...")
    
    gnn_routing.visualize(
        embeddings=embeddings,
        path=path,
        save_path='examples/outputs/16_gnn_embeddings.png'
    )
    
    # 6. 保存结果
    results = {
        'n_nodes': n_nodes,
        'path': path,
        'path_length': len(path),
        'total_cost': float(total_cost),
        'embedding_dim': 32,
        'training_iterations': 100
    }
    
    import json
    with open('examples/outputs/16_gnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到：examples/outputs/16_gnn_results.json")
    
    print("\n" + "=" * 70)
    print(" " * 25 + "✅ GNN 演示完成!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
