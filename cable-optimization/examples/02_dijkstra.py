"""
最短路径树方法 - Dijkstra 算法

问题描述:
给定带权有向图 G=(V,E)，从源点 s 到所有其他节点的最短路径

算法原理:
1. 维护两个集合：已确定最短路径的节点集合 S，未确定的集合 V-S
2. 每次从 V-S 中选择距离最小的节点 u 加入 S
3. 松弛 u 的所有出边：如果经过 u 到 v 更短，则更新 v 的距离

时间复杂度：O((V+E)logV) 使用优先队列
"""

import heapq
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class DijkstraSolver:
    """Dijkstra 算法求解器"""
    
    def __init__(self, G):
        self.G = G
        self.distances = {}
        self.predecessors = {}
    
    def solve(self, source):
        """
        求解从 source 到所有节点的最短路径
        
        返回:
            distances: 源点到各节点的最短距离
            predecessors: 前驱节点（用于重构路径）
        """
        # 初始化
        self.distances = {node: float('inf') for node in self.G.nodes()}
        self.distances[source] = 0
        self.predecessors = {node: None for node in self.G.nodes()}
        
        # 优先队列：(距离，节点)
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # 松弛操作
            for neighbor in self.G.successors(current_node):
                edge_weight = self.G[current_node][neighbor].get('weight', 1)
                distance = current_dist + edge_weight
                
                if distance < self.distances[neighbor]:
                    self.distances[neighbor] = distance
                    self.predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        return self.distances, self.predecessors
    
    def get_path(self, source, target):
        """重构从 source 到 target 的路径"""
        if self.distances.get(target, float('inf')) == float('inf'):
            return None, float('inf')
        
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = self.predecessors[current]
        
        path.reverse()
        return path, self.distances[target]


def create_cable_network():
    """创建一个电缆布线网络示例"""
    G = nx.DiGraph()
    
    # 节点表示接线盒/节点
    nodes = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
    G.add_nodes_from(nodes)
    
    # 边表示可能的布线路径，权重为距离/成本
    edges = [
        ('J0', 'J1', 4), ('J0', 'J2', 2),
        ('J1', 'J2', 1), ('J1', 'J3', 5),
        ('J2', 'J3', 8), ('J2', 'J4', 10),
        ('J3', 'J4', 2), ('J3', 'J5', 6),
        ('J4', 'J5', 3), ('J4', 'J6', 7),
        ('J5', 'J6', 1), ('J5', 'J7', 4),
        ('J6', 'J7', 2),
    ]
    
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)
    
    return G


def visualize_dijkstra(G, distances, predecessors, source, path=None):
    """可视化 Dijkstra 算法结果"""
    pos = nx.spring_layout(G, seed=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：原始网络
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10)
    
    # 绘制所有边
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', width=1.5, arrows=True)
    
    # 标注边权重
    edge_labels = {(u, v): str(G[u][v]['weight']) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1, 
                                font_size=9, label_pos=0.5)
    
    # 标记源点
    nx.draw_networkx_nodes(G, pos, nodelist=[source], ax=ax1,
                          node_color='red', node_size=600, label='Source')
    
    ax1.set_title("电缆布线网络", fontsize=14)
    ax1.axis('off')
    ax1.legend()
    
    # 右图：最短路径树
    node_colors = []
    for node in G.nodes():
        if node == source:
            node_colors.append('red')
        elif distances.get(node, float('inf')) == float('inf'):
            node_colors.append('gray')
        else:
            node_colors.append('lightgreen')
    
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10)
    
    # 绘制最短路径树边
    tree_edges = []
    for node in G.nodes():
        if node != source and predecessors.get(node) is not None:
            tree_edges.append((predecessors[node], node))
    
    if tree_edges:
        nx.draw_networkx_edges(G, pos, edgelist=tree_edges, ax=ax2,
                              edge_color='green', width=2, arrows=True)
    
    # 如果指定了路径，高亮显示
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax2,
                              edge_color='red', width=3, arrows=True)
    
    # 标注距离
    distance_labels = {node: f"d={d:.1f}" if d < float('inf') else "∞" 
                      for node, d in distances.items()}
    nx.draw_networkx_labels(G, pos, labels=distance_labels, ax=ax2, 
                           font_size=8, verticalalignment='bottom')
    
    ax2.set_title(f"最短路径树 (从 {source} 出发)", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("最短路径树方法 - Dijkstra 算法")
    print("=" * 60)
    
    # 创建网络
    G = create_cable_network()
    print(f"\n✓ 创建电缆网络：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")
    
    # 设置源点
    source = 'J0'
    print(f"✓ 源点：{source}")
    
    # 求解
    print("\n⏳ 运行 Dijkstra 算法...")
    solver = DijkstraSolver(G)
    distances, predecessors = solver.solve(source)
    
    # 输出结果
    print(f"\n📊 最短路径结果:")
    print(f"\n  从 {source} 到各节点的最短距离:")
    for node in sorted(G.nodes()):
        dist = distances.get(node, float('inf'))
        if dist < float('inf'):
            path, _ = solver.get_path(source, node)
            path_str = " → ".join(path)
            print(f"    {node}: 距离 = {dist:.1f}, 路径：{path_str}")
        else:
            print(f"    {node}: 不可达")
    
    # 计算特定路径
    target = 'J7'
    path, path_cost = solver.get_path(source, target)
    if path:
        print(f"\n🎯 到 {target} 的最优路径:")
        print(f"    路径：{' → '.join(path)}")
        print(f"    总成本：{path_cost}")
    
    # 可视化
    fig = visualize_dijkstra(G, distances, predecessors, source, path)
    
    # 保存图像
    output_path = "/root/.openclaw/workspace/cable-optimization/examples/dijkstra_result.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 结果图已保存到：{output_path}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
