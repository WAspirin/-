#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模线缆布线优化 - 分解算法

作者：智子 (Sophon)
日期：2026-03-13
主题：Week 3 Day 13 - 大规模问题求解技巧

核心思想：
1. 分治法 (Divide and Conquer)：将大规模问题分解为小规模子问题
2. 聚类分解：使用 K-Means 将节点分组
3. 层次优化：先优化簇内，再优化簇间连接
4. 并行计算：独立子问题可并行求解

适用场景：
- 节点数 > 50 的大规模布线问题
- 需要快速获得可接受解
- 计算资源有限

算法流程：
1. 节点聚类（K-Means）
2. 构建簇间连接图
3. 对每个簇内求解 TSP
4. 连接簇形成完整路径
5. 全局优化改进
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Cluster:
    """簇数据结构"""
    cluster_id: int
    nodes: List[int]  # 节点索引列表
    center: np.ndarray  # 簇中心坐标
    entry_node: int = None  # 进入该簇的节点
    exit_node: int = None  # 离开该簇的节点
    internal_path: List[int] = None  # 簇内路径


@dataclass
class DecompositionConfig:
    """分解算法配置"""
    n_clusters: int = 10  # 簇数量
    n_iterations: int = 100  # 簇内优化迭代次数
    population_size: int = 50  # 簇内 GA 种群大小
    seed: int = 42  # 随机种子


class LargeScaleCableRouting:
    """大规模线缆布线问题"""
    
    def __init__(self, n_nodes: int = 100, seed: int = 42):
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        
        # 生成节点坐标
        self.coordinates = np.random.rand(n_nodes, 2) * 100
        
        # 固定起点和终点
        self.start_node = 0
        self.end_node = n_nodes - 1
        
        # 距离矩阵
        self.dist_matrix = self._compute_distance_matrix()
        
        # 风险地图（某些区域风险高）
        self.risk_map = self._generate_risk_map()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """计算欧氏距离矩阵"""
        diff = self.coordinates[:, np.newaxis, :] - self.coordinates[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    def _generate_risk_map(self) -> np.ndarray:
        """生成风险地图"""
        # 随机风险区域
        risk = np.random.rand(self.n_nodes) * 0.5
        
        # 中心区域高风险
        center = np.mean(self.coordinates, axis=0)
        dist_to_center = np.linalg.norm(self.coordinates - center, axis=1)
        risk += np.exp(-dist_to_center / 20) * 0.3
        
        return np.clip(risk, 0, 1)
    
    def compute_path_cost(self, path: List[int]) -> Tuple[float, float]:
        """计算路径的总成本和风险"""
        total_length = 0.0
        total_risk = 0.0
        
        for i in range(len(path) - 1):
            total_length += self.dist_matrix[path[i], path[i + 1]]
            total_risk += self.risk_map[path[i]]
        
        # 加上终点风险
        total_risk += self.risk_map[path[-1]]
        
        return total_length, total_risk


class DecompositionOptimizer:
    """分解优化器"""
    
    def __init__(self, problem: LargeScaleCableRouting, config: DecompositionConfig):
        self.problem = problem
        self.config = config
        self.clusters: List[Cluster] = []
        self.cluster_graph = None  # 簇间连接图
        self.best_solution = None
        self.best_cost = float('inf')
    
    def optimize(self) -> Tuple[List[int], Dict]:
        """执行分解优化"""
        print("=" * 60)
        print("大规模线缆布线优化 - 分解算法")
        print("=" * 60)
        print(f"节点数：{self.problem.n_nodes}")
        print(f"簇数量：{self.config.n_clusters}")
        print()
        
        start_time = time.time()
        
        # Step 1: 节点聚类
        print("Step 1: 节点聚类 (K-Means)...")
        self._cluster_nodes()
        print(f"  ✓ 完成聚类，{len(self.clusters)} 个簇")
        
        # Step 2: 构建簇间连接图
        print("\nStep 2: 构建簇间连接图...")
        self._build_cluster_graph()
        print("  ✓ 簇间图构建完成")
        
        # Step 3: 簇间路径优化
        print("\nStep 3: 簇间路径优化...")
        cluster_order = self._optimize_cluster_order()
        print(f"  ✓ 簇间路径：{' -> '.join(map(str, cluster_order))}")
        
        # Step 4: 簇内路径优化
        print("\nStep 4: 簇内路径优化...")
        self._optimize_internal_paths(cluster_order)
        print("  ✓ 所有簇内路径优化完成")
        
        # Step 5: 合并路径
        print("\nStep 5: 合并完整路径...")
        full_path = self._merge_paths(cluster_order)
        print(f"  ✓ 完整路径长度：{len(full_path)} 个节点")
        
        # Step 6: 全局优化
        print("\nStep 6: 全局优化改进...")
        full_path, improvement = self._global_optimization(full_path)
        print(f"  ✓ 全局优化改进：{improvement:.2f}%")
        
        # 计算最终成本
        total_time = time.time() - start_time
        length, risk = self.problem.compute_path_cost(full_path)
        
        print("\n" + "=" * 60)
        print("优化结果")
        print("=" * 60)
        print(f"总路径长度：{length:.2f}")
        print(f"总风险：{risk:.2f}")
        print(f"总时间：{total_time:.2f} 秒")
        print("=" * 60)
        
        stats = {
            'n_clusters': len(self.clusters),
            'path_length': length,
            'path_risk': risk,
            'computation_time': total_time,
            'improvement': improvement
        }
        
        return full_path, stats
    
    def _cluster_nodes(self):
        """使用 K-Means 聚类节点"""
        # 确保起点和终点在不同簇或特殊处理
        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.seed,
            n_init=10
        )
        
        labels = kmeans.fit_predict(self.problem.coordinates)
        
        # 创建簇对象
        for cluster_id in range(self.config.n_clusters):
            nodes = np.where(labels == cluster_id)[0].tolist()
            center = kmeans.cluster_centers_[cluster_id]
            
            cluster = Cluster(
                cluster_id=cluster_id,
                nodes=nodes,
                center=center
            )
            self.clusters.append(cluster)
    
    def _build_cluster_graph(self):
        """构建簇间连接图"""
        n_clusters = len(self.clusters)
        
        # 计算簇间距离（簇中心距离）
        centers = np.array([c.center for c in self.clusters])
        self.cluster_graph = cdist(centers, centers, metric='euclidean')
        
        # 找出每个簇的边界节点（最靠近其他簇的节点）
        for i, cluster in enumerate(self.clusters):
            cluster.boundary_nodes = {}
            for j in range(n_clusters):
                if i != j:
                    # 找到簇 i 中离簇 j 最近的节点
                    min_dist = float('inf')
                    closest_node = None
                    
                    for node in cluster.nodes:
                        dist = np.linalg.norm(
                            self.problem.coordinates[node] - centers[j]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            closest_node = node
                    
                    cluster.boundary_nodes[j] = closest_node
    
    def _optimize_cluster_order(self) -> List[int]:
        """优化簇的访问顺序（TSP）"""
        n_clusters = len(self.clusters)
        
        # 找到起点和终点所在的簇
        start_cluster = None
        end_cluster = None
        
        for cluster in self.clusters:
            if self.problem.start_node in cluster.nodes:
                start_cluster = cluster.cluster_id
            if self.problem.end_node in cluster.nodes:
                end_cluster = cluster.cluster_id
        
        # 使用最近邻启发式确定簇顺序
        order = [start_cluster]
        visited = {start_cluster}
        
        while len(order) < n_clusters:
            current = order[-1]
            best_next = None
            best_dist = float('inf')
            
            for next_cluster in range(n_clusters):
                if next_cluster not in visited:
                    dist = self.cluster_graph[current, next_cluster]
                    if dist < best_dist:
                        best_dist = dist
                        best_next = next_cluster
            
            order.append(best_next)
            visited.add(best_next)
        
        # 确保终点簇在最后
        if order[-1] != end_cluster:
            order.remove(end_cluster)
            order.append(end_cluster)
        
        return order
    
    def _optimize_internal_paths(self, cluster_order: List[int]):
        """优化每个簇内的路径"""
        for cluster_id in cluster_order:
            cluster = self.clusters[cluster_id]
            
            # 确定入口和出口节点
            if cluster_id == cluster_order[0]:
                # 第一个簇：从起点开始
                cluster.entry_node = self.problem.start_node
                # 出口是离下一个簇最近的节点
                next_cluster_id = cluster_order[1]
                cluster.exit_node = cluster.boundary_nodes[next_cluster_id]
            
            elif cluster_id == cluster_order[-1]:
                # 最后一个簇：到终点结束
                cluster.entry_node = cluster.boundary_nodes[cluster_order[-2]]
                cluster.exit_node = self.problem.end_node
            
            else:
                # 中间簇
                cluster.entry_node = cluster.boundary_nodes[cluster_order[cluster_order.index(cluster_id) - 1]]
                cluster.exit_node = cluster.boundary_nodes[cluster_order[cluster_order.index(cluster_id) + 1]]
            
            # 簇内节点（排除入口和出口）
            internal_nodes = [n for n in cluster.nodes 
                            if n not in [cluster.entry_node, cluster.exit_node]]
            
            # 使用贪心 + 2-opt 优化簇内路径
            if len(internal_nodes) > 0:
                cluster.internal_path = self._optimize_cluster_path(
                    cluster.entry_node,
                    internal_nodes,
                    cluster.exit_node
                )
            else:
                cluster.internal_path = [cluster.entry_node, cluster.exit_node]
    
    def _optimize_cluster_path(self, entry: int, internal: List[int], exit_node: int) -> List[int]:
        """优化单个簇内的路径（使用 GA + 2-opt）"""
        if len(internal) == 0:
            return [entry, exit_node]
        
        # 贪心初始化
        path = [entry] + internal.copy() + [exit_node]
        
        # 2-opt 优化
        improved = True
        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    # 计算当前成本
                    old_cost = (self.problem.dist_matrix[path[i-1], path[i]] +
                               self.problem.dist_matrix[path[j], path[j+1]])
                    
                    # 计算 2-opt 后成本
                    new_cost = (self.problem.dist_matrix[path[i-1], path[j]] +
                               self.problem.dist_matrix[path[i], path[j+1]])
                    
                    if new_cost < old_cost:
                        # 执行 2-opt
                        path[i:j+1] = reversed(path[i:j+1])
                        improved = True
        
        return path
    
    def _merge_paths(self, cluster_order: List[int]) -> List[int]:
        """合并所有簇的路径"""
        full_path = []
        
        for i, cluster_id in enumerate(cluster_order):
            cluster = self.clusters[cluster_id]
            
            if i == 0:
                # 第一个簇：从 entry_node 开始
                full_path.extend(cluster.internal_path)
            else:
                # 后续簇：跳过 entry_node（已在上一个簇的 exit 中）
                full_path.extend(cluster.internal_path[1:])
        
        return full_path
    
    def _global_optimization(self, path: List[int], n_iterations: int = 50) -> Tuple[List[int], float]:
        """全局优化改进"""
        best_path = path.copy()
        best_cost, _ = self.problem.compute_path_cost(best_path)
        initial_cost = best_cost
        
        current_path = path.copy()
        
        for _ in range(n_iterations):
            # 随机选择一个 2-opt 操作
            i = np.random.randint(1, len(current_path) - 2)
            j = np.random.randint(i + 1, len(current_path) - 1)
            
            # 生成邻居
            neighbor = current_path.copy()
            neighbor[i:j+1] = reversed(neighbor[i:j+1])
            
            # 评估
            neighbor_cost, _ = self.problem.compute_path_cost(neighbor)
            
            if neighbor_cost < best_cost:
                best_path = neighbor
                best_cost = neighbor_cost
                current_path = neighbor
        
        improvement = (initial_cost - best_cost) / initial_cost * 100
        return best_path, improvement


class DecompositionVisualizer:
    """分解算法可视化"""
    
    def __init__(self, problem: LargeScaleCableRouting, optimizer: DecompositionOptimizer):
        self.problem = problem
        self.optimizer = optimizer
    
    def plot_clustering(self, save_path: str = None):
        """绘制聚类结果"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左图：节点聚类
        ax1 = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.optimizer.clusters)))
        
        for i, cluster in enumerate(self.optimizer.clusters):
            coords = self.problem.coordinates[cluster.nodes]
            ax1.scatter(coords[:, 0], coords[:, 1], c=[colors[i]], 
                       s=50, label=f'簇 {i}', alpha=0.7, edgecolors='black')
            
            # 绘制簇中心
            ax1.scatter(cluster.center[0], cluster.center[1], 
                       c='red', s=200, marker='X', edgecolors='black', linewidths=2)
        
        # 标记起点和终点
        ax1.scatter(self.problem.coordinates[self.problem.start_node, 0],
                   self.problem.coordinates[self.problem.start_node, 1],
                   c='green', s=300, marker='s', label='起点', edgecolors='black', linewidths=2)
        
        ax1.scatter(self.problem.coordinates[self.problem.end_node, 0],
                   self.problem.coordinates[self.problem.end_node, 1],
                   c='red', s=300, marker='s', label='终点', edgecolors='black', linewidths=2)
        
        ax1.set_xlabel('X 坐标', fontsize=12)
        ax1.set_ylabel('Y 坐标', fontsize=12)
        ax1.set_title('节点聚类结果', fontsize=14)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 右图：簇间连接图
        ax2 = axes[1]
        centers = np.array([c.center for c in self.optimizer.clusters])
        
        # 绘制簇中心
        ax2.scatter(centers[:, 0], centers[:, 1], c='blue', s=200, 
                   edgecolors='black', linewidths=2)
        
        # 绘制簇间边
        for i in range(len(self.optimizer.clusters)):
            for j in range(i + 1, len(self.optimizer.clusters)):
                weight = self.optimizer.cluster_graph[i, j]
                alpha = 1 - min(weight / 100, 0.8)  # 距离越近越不透明
                
                ax2.plot([centers[i, 0], centers[j, 0]],
                        [centers[i, 1], centers[j, 1]],
                        'gray', alpha=alpha, linewidth=1)
        
        ax2.set_xlabel('X 坐标', fontsize=12)
        ax2.set_ylabel('Y 坐标', fontsize=12)
        ax2.set_title('簇间连接图', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"聚类图已保存：{save_path}")
        
        plt.show()
    
    def plot_solution(self, path: List[int], save_path: str = None):
        """绘制最终解"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左图：完整路径
        ax1 = axes[0]
        
        # 绘制所有节点
        ax1.scatter(self.problem.coordinates[:, 0], self.problem.coordinates[:, 1],
                   c='lightgray', s=30, alpha=0.5)
        
        # 绘制路径
        path_coords = self.problem.coordinates[path]
        ax1.plot(path_coords[:, 0], path_coords[:, 1], 'b-', linewidth=1.5, 
                label='布线路径', alpha=0.7)
        
        # 绘制簇边界
        for cluster in self.optimizer.clusters:
            cluster_coords = self.problem.coordinates[cluster.nodes]
            if len(cluster_coords) > 2:
                hull = self._compute_convex_hull(cluster_coords)
                polygon = Polygon(hull, fill=True, alpha=0.1, 
                                 edgecolor='blue', linewidth=1)
                ax1.add_patch(polygon)
        
        # 标记起点和终点
        ax1.scatter(self.problem.coordinates[self.problem.start_node, 0],
                   self.problem.coordinates[self.problem.start_node, 1],
                   c='green', s=200, marker='s', label='起点', 
                   edgecolors='black', linewidths=2, zorder=5)
        
        ax1.scatter(self.problem.coordinates[self.problem.end_node, 0],
                   self.problem.coordinates[self.problem.end_node, 1],
                   c='red', s=200, marker='s', label='终点', 
                   edgecolors='black', linewidths=2, zorder=5)
        
        ax1.set_xlabel('X 坐标', fontsize=12)
        ax1.set_ylabel('Y 坐标', fontsize=12)
        ax1.set_title('大规模布线路径 (分解算法)', fontsize=14)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 右图：路径成本分析
        ax2 = axes[1]
        
        # 计算累积成本
        cumulative_length = [0]
        cumulative_risk = [0]
        
        for i in range(len(path) - 1):
            cumulative_length.append(cumulative_length[-1] + 
                                    self.problem.dist_matrix[path[i], path[i+1]])
            cumulative_risk.append(cumulative_risk[-1] + 
                                  self.problem.risk_map[path[i]])
        
        ax2.plot(range(len(path)), cumulative_length, 'b-', linewidth=2, 
                label='累积长度')
        ax2.set_xlabel('路径节点索引', fontsize=12)
        ax2.set_ylabel('累积长度', fontsize=12, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(True, alpha=0.3)
        
        # 双 Y 轴显示风险
        ax2_risk = ax2.twinx()
        ax2_risk.plot(range(len(path)), cumulative_risk, 'r-', linewidth=2, 
                     label='累积风险')
        ax2_risk.set_ylabel('累积风险', fontsize=12, color='red')
        ax2_risk.tick_params(axis='y', labelcolor='red')
        
        ax2.set_title('路径成本分析', fontsize=14)
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_risk.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"路径图已保存：{save_path}")
        
        plt.show()
    
    def _compute_convex_hull(self, points: np.ndarray) -> np.ndarray:
        """计算点集的凸包"""
        from scipy.spatial import ConvexHull
        if len(points) < 3:
            return points
        
        hull = ConvexHull(points)
        return points[hull.vertices]


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("大规模线缆布线优化 - 分解算法演示")
    print("=" * 60 + "\n")
    
    # 创建大规模问题（100 个节点）
    problem = LargeScaleCableRouting(n_nodes=100, seed=42)
    
    # 配置分解算法
    config = DecompositionConfig(
        n_clusters=10,
        n_iterations=100,
        population_size=50,
        seed=42
    )
    
    # 创建优化器
    optimizer = DecompositionOptimizer(problem, config)
    
    # 执行优化
    path, stats = optimizer.optimize()
    
    # 可视化
    visualizer = DecompositionVisualizer(problem, optimizer)
    
    print("\n生成可视化图表...")
    visualizer.plot_clustering(save_path='outputs/large_scale_clustering.png')
    visualizer.plot_solution(path, save_path='outputs/large_scale_solution.png')
    
    print("\n" + "=" * 60)
    print("大规模优化完成！")
    print("=" * 60)
    print(f"\n关键指标:")
    print(f"  • 节点数：{problem.n_nodes}")
    print(f"  • 簇数量：{stats['n_clusters']}")
    print(f"  • 路径长度：{stats['path_length']:.2f}")
    print(f"  • 路径风险：{stats['path_risk']:.2f}")
    print(f"  • 计算时间：{stats['computation_time']:.2f} 秒")
    print(f"  • 全局优化改进：{stats['improvement']:.2f}%")
    print()


if __name__ == "__main__":
    main()
