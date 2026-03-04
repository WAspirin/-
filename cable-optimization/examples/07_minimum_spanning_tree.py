"""
图论算法 - 最小生成树 (Minimum Spanning Tree, MST)

问题描述:
在连通加权无向图中找到连接所有顶点的最小权重树
应用于线缆布线：以最小总长度连接所有节点

算法原理:
1. Prim 算法 (贪心策略):
   - 从任意顶点开始
   - 每次选择连接已选顶点集和未选顶点集的最小权重边
   - 重复直到所有顶点都被选中
   - 时间复杂度: O(E log V) 使用优先队列

2. Kruskal 算法 (贪心策略):
   - 将所有边按权重排序
   - 依次选择最小权重边，如果该边不形成环
   - 使用并查集 (Union-Find) 检测环
   - 重复直到选择 V-1 条边
   - 时间复杂度: O(E log E)

应用场景:
- 网络设计（电缆、管道、道路）
- 聚类分析
- 近似算法（如 TSP 的 2-approximation）
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq


@dataclass
class Edge:
    """边类"""
    u: int  # 起点
    v: int  # 终点
    weight: float  # 权重
    
    def __lt__(self, other):
        return self.weight < other.weight


class UnionFind:
    """并查集 - 用于 Kruskal 算法"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """查找根节点（带路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        合并两个集合
        返回 True 如果合并成功（不在同一集合），False 如果已在同一集合
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        # 按秩合并
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True


class MinimumSpanningTree:
    """最小生成树求解器"""
    
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.adj_list: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]
        self.edges: List[Edge] = []
    
    def add_edge(self, u: int, v: int, weight: float):
        """添加无向边"""
        self.adj_list[u].append((v, weight))
        self.adj_list[v].append((u, weight))
        self.edges.append(Edge(u, v, weight))
    
    def prim(self, start: int = 0) -> Tuple[List[Edge], float]:
        """
        Prim 算法求最小生成树
        
        Args:
            start: 起始顶点
            
        Returns:
            mst_edges: MST 的边列表
            total_weight: 总权重
        """
        if start < 0 or start >= self.n_nodes:
            raise ValueError(f"Invalid start node: {start}")
        
        visited: Set[int] = {start}
        mst_edges: List[Edge] = []
        total_weight = 0.0
        
        # 优先队列：(weight, u, v)
        pq = []
        for v, weight in self.adj_list[start]:
            heapq.heappush(pq, (weight, start, v))
        
        while pq and len(visited) < self.n_nodes:
            weight, u, v = heapq.heappop(pq)
            
            if v in visited:
                continue
            
            # 选择这条边
            visited.add(v)
            mst_edges.append(Edge(u, v, weight))
            total_weight += weight
            
            # 添加新顶点的边到队列
            for next_v, next_weight in self.adj_list[v]:
                if next_v not in visited:
                    heapq.heappush(pq, (next_weight, v, next_v))
        
        if len(mst_edges) != self.n_nodes - 1:
            raise ValueError("Graph is not connected")
        
        return mst_edges, total_weight
    
    def kruskal(self) -> Tuple[List[Edge], float]:
        """
        Kruskal 算法求最小生成树
        
        Returns:
            mst_edges: MST 的边列表
            total_weight: 总权重
        """
        # 按权重排序所有边
        sorted_edges = sorted(self.edges)
        
        uf = UnionFind(self.n_nodes)
        mst_edges: List[Edge] = []
        total_weight = 0.0
        
        for edge in sorted_edges:
            if uf.union(edge.u, edge.v):
                mst_edges.append(edge)
                total_weight += edge.weight
                
                if len(mst_edges) == self.n_nodes - 1:
                    break
        
        if len(mst_edges) != self.n_nodes - 1:
            raise ValueError("Graph is not connected")
        
        return mst_edges, total_weight
    
    def visualize(self, mst_edges: List[Edge], title: str = "Minimum Spanning Tree"):
        """可视化最小生成树"""
        # 创建网络图
        G = nx.Graph()
        
        # 添加所有节点
        for i in range(self.n_nodes):
            G.add_node(i)
        
        # 添加 MST 边
        for edge in mst_edges:
            G.add_edge(edge.u, edge.v, weight=edge.weight)
        
        # 生成节点位置（使用 spring_layout 使布局美观）
        np.random.seed(42)
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左图：原始图（所有边）
        ax1.set_title("Original Graph (All Edges)", fontsize=14, fontweight='bold')
        
        # 绘制所有边（灰色）
        for edge in self.edges:
            x0, y0 = pos[edge.u]
            x1, y1 = pos[edge.v]
            ax1.plot([x0, x1], [y0, y1], 'gray', alpha=0.3, linewidth=1)
        
        # 绘制节点
        node_colors = ['#3498db'] * self.n_nodes
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                               node_size=500, edgecolors='black', linewidths=2)
        
        # 绘制边权重
        edge_labels = {(e.u, e.v): f'{e.weight:.1f}' for e in self.edges}
        nx.draw_networkx_edge_labels(G, pos, ax=ax1, edge_labels=edge_labels, 
                                     font_size=8, alpha=0.7)
        
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10, font_weight='bold')
        ax1.axis('off')
        
        # 右图：MST（高亮显示）
        ax2.set_title(f"{title}\nTotal Weight: {sum(e.weight for e in mst_edges):.2f}", 
                     fontsize=14, fontweight='bold')
        
        # 绘制所有边（背景，灰色）
        for edge in self.edges:
            x0, y0 = pos[edge.u]
            x1, y1 = pos[edge.v]
            ax2.plot([x0, x1], [y0, y1], 'gray', alpha=0.2, linewidth=1)
        
        # 高亮 MST 边
        mst_G = nx.Graph()
        for edge in mst_edges:
            mst_G.add_edge(edge.u, edge.v, weight=edge.weight)
        
        # 绘制 MST 边（彩色）
        mst_edge_list = [(e.u, e.v) for e in mst_edges]
        nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=mst_edge_list,
                               edge_color='#e74c3c', width=3, alpha=0.8)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='#3498db',
                               node_size=500, edgecolors='black', linewidths=2)
        
        # 绘制 MST 边权重
        mst_edge_labels = {(e.u, e.v): f'{e.weight:.1f}' for e in mst_edges}
        nx.draw_networkx_edge_labels(G, pos, ax=ax2, edge_labels=mst_edge_labels,
                                     font_size=9, font_weight='bold', font_color='#e74c3c')
        
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10, font_weight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('mst_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return pos


def generate_cable_network(n_nodes: int, seed: int = 42) -> MinimumSpanningTree:
    """
    生成线缆布线网络示例
    
    Args:
        n_nodes: 节点数量
        seed: 随机种子
        
    Returns:
        图对象
    """
    np.random.seed(seed)
    
    mst = MinimumSpanningTree(n_nodes)
    
    # 生成随机节点位置
    positions = np.random.rand(n_nodes, 2) * 100
    
    # 添加边（完全图，权重为欧几里得距离）
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            distance = np.linalg.norm(positions[i] - positions[j])
            # 添加一些随机扰动模拟实际布线成本
            cost = distance * (1 + 0.1 * np.random.rand())
            mst.add_edge(i, j, cost)
    
    return mst


def compare_algorithms():
    """对比 Prim 和 Kruskal 算法"""
    print("=" * 60)
    print("最小生成树算法对比")
    print("=" * 60)
    
    # 生成测试图
    n_nodes = 15
    mst_solver = generate_cable_network(n_nodes, seed=42)
    
    # 运行 Prim 算法
    print(f"\n节点数：{n_nodes}")
    print(f"边数：{len(mst_solver.edges)}")
    
    prim_edges, prim_weight = mst_solver.prim(start=0)
    print(f"\n【Prim 算法】")
    print(f"  总权重：{prim_weight:.2f}")
    print(f"  边数：{len(prim_edges)}")
    print(f"  MST 边:")
    for edge in prim_edges[:5]:
        print(f"    {edge.u} -- {edge.v}: {edge.weight:.2f}")
    if len(prim_edges) > 5:
        print(f"    ... (共 {len(prim_edges)} 条边)")
    
    # 运行 Kruskal 算法
    kruskal_edges, kruskal_weight = mst_solver.kruskal()
    print(f"\n【Kruskal 算法】")
    print(f"  总权重：{kruskal_weight:.2f}")
    print(f"  边数：{len(kruskal_edges)}")
    print(f"  MST 边:")
    for edge in kruskal_edges[:5]:
        print(f"    {edge.u} -- {edge.v}: {edge.weight:.2f}")
    if len(kruskal_edges) > 5:
        print(f"    ... (共 {len(kruskal_edges)} 条边)")
    
    # 验证结果
    print(f"\n【验证】")
    print(f"  两种算法结果一致：{abs(prim_weight - kruskal_weight) < 0.001}")
    print(f"  边数正确 (V-1): {len(prim_edges) == n_nodes - 1}")
    
    # 可视化
    mst_solver.visualize(prim_edges, "MST - Prim Algorithm")
    
    return prim_weight, kruskal_weight


def cable_routing_application():
    """
    应用案例：园区网络布线
    
    场景：需要在园区内连接 10 栋建筑，求最小总长度的布线方案
    """
    print("\n" + "=" * 60)
    print("应用案例：园区网络布线")
    print("=" * 60)
    
    # 创建园区建筑位置（模拟坐标）
    buildings = {
        0: "主楼",
        1: "图书馆",
        2: "实验楼 A",
        3: "实验楼 B",
        4: "宿舍楼 1",
        5: "宿舍楼 2",
        6: "食堂",
        7: "体育馆",
        8: "行政楼",
        9: "校门"
    }
    
    n_buildings = len(buildings)
    
    # 生成建筑间距离（基于实际坐标）
    np.random.seed(123)
    positions = {
        0: (50, 50),   # 主楼 - 中心
        1: (30, 70),   # 图书馆
        2: (70, 70),   # 实验楼 A
        3: (80, 50),   # 实验楼 B
        4: (20, 30),   # 宿舍楼 1
        5: (30, 20),   # 宿舍楼 2
        6: (60, 30),   # 食堂
        7: (80, 20),   # 体育馆
        8: (50, 80),   # 行政楼
        9: (50, 10)    # 校门
    }
    
    mst_solver = MinimumSpanningTree(n_buildings)
    
    # 添加所有可能的连接（完全图）
    for i in range(n_buildings):
        for j in range(i + 1, n_buildings):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            # 布线成本 = 距离 * 单位成本（假设 100 元/米）
            cost = dist * 100
            mst_solver.add_edge(i, j, cost)
    
    # 求解 MST
    mst_edges, total_cost = mst_solver.prim(start=0)
    
    print(f"\n园区建筑数量：{n_buildings}")
    print(f"可能的连接数：{len(mst_solver.edges)}")
    print(f"MST 连接数：{len(mst_edges)}")
    print(f"\n【最优布线方案】")
    print(f"  总成本：¥{total_cost:,.2f}")
    print(f"  布线路线:")
    
    for edge in mst_edges:
        building_u = buildings[edge.u]
        building_v = buildings[edge.v]
        print(f"    {building_u} ←→ {building_v}: ¥{edge.weight:,.2f}")
    
    print(f"\n【节省分析】")
    # 计算如果所有边都布线的成本
    total_all_edges = sum(e.weight for e in mst_solver.edges)
    savings = total_all_edges - total_cost
    print(f"  全连接成本：¥{total_all_edges:,.2f}")
    print(f"  MST 成本：¥{total_cost:,.2f}")
    print(f"  节省：¥{savings:,.2f} ({savings/total_all_edges*100:.1f}%)")
    
    # 可视化
    mst_solver.visualize(mst_edges, "Campus Network - MST")


if __name__ == "__main__":
    # 运行算法对比
    compare_algorithms()
    
    # 运行应用案例
    cable_routing_application()
    
    print("\n" + "=" * 60)
    print("✅ 最小生成树算法学习完成!")
    print("=" * 60)
