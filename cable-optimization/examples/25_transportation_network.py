#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交通网络布线优化案例
Transportation Network Cable Routing Optimization

场景：城市交通监控与通信网络布线
- 沿道路网络部署摄像头和通信节点
- 最小化布线成本
- 满足覆盖和连通性约束
- 考虑交通流量和施工难度

作者：智子 (Sophon)
日期：2026-03-23
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import random


# ============================================================================
# 数据类定义
# ============================================================================

class RoadType(Enum):
    """道路类型"""
    HIGHWAY = "高速公路"  # 施工难度大，成本高
    MAIN_ROAD = "主干道"  # 中等难度
    SECONDARY = "次干道"  # 较低难度
    BRANCH = "支路"  # 施工容易


@dataclass
class Intersection:
    """交叉路口节点"""
    id: int
    x: float
    y: float
    type: str = "normal"  # normal/hub/terminal
    traffic_flow: float = 1.0  # 交通流量系数 (0.5-2.0)
    
    def distance_to(self, other: 'Intersection') -> float:
        """计算到另一个路口的距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class RoadSegment:
    """道路段"""
    id: int
    from_node: int
    to_node: int
    road_type: RoadType
    length: float
    construction_difficulty: float = 1.0  # 施工难度系数
    
    def get_cable_cost(self, cable_type: str = "standard") -> float:
        """计算电缆铺设成本"""
        base_costs = {
            "standard": 50.0,   # 标准电缆 (元/米)
            "fiber": 120.0,     # 光纤
            "heavy": 80.0       # 重型电缆
        }
        base_cost = base_costs.get(cable_type, 50.0)
        # 成本 = 基础成本 × 长度 × 施工难度 × 交通流量影响
        return base_cost * self.length * self.construction_difficulty


@dataclass
class CameraNode:
    """监控摄像头节点"""
    id: int
    intersection_id: int
    camera_type: str = "HD"  # HD/4K/PTZ
    priority: int = 1  # 优先级 1-5
    coverage_radius: float = 50.0  # 覆盖半径 (米)


@dataclass
class CommunicationHub:
    """通信汇聚节点"""
    id: int
    intersection_id: int
    capacity: int = 10  # 最大连接摄像头数
    connected_cameras: List[int] = field(default_factory=list)


# ============================================================================
# 交通网络建模
# ============================================================================

class TransportationNetwork:
    """交通网络模型"""
    
    def __init__(self, grid_size: int = 10, seed: int = 42):
        self.grid_size = grid_size
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        self.intersections: Dict[int, Intersection] = {}
        self.road_segments: Dict[int, RoadSegment] = {}
        self.cameras: Dict[int, CameraNode] = {}
        self.hubs: Dict[int, CommunicationHub] = {}
        
        self._create_grid_network()
        self._assign_road_types()
        self._place_cameras()
        self._place_hubs()
    
    def _create_grid_network(self):
        """创建网格状道路网络"""
        node_id = 0
        spacing = 100.0  # 路口间距 (米)
        
        # 创建路口节点
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = i * spacing
                y = j * spacing
                # 边缘节点可能是终端节点
                if i == 0 or i == self.grid_size - 1 or \
                   j == 0 or j == self.grid_size - 1:
                    node_type = "terminal"
                elif (i + j) % 3 == 0:
                    node_type = "hub"  # 潜在的汇聚点
                else:
                    node_type = "normal"
                
                # 交通流量：中心区域更高
                center_x = self.grid_size / 2
                center_y = self.grid_size / 2
                dist_from_center = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                max_dist = np.sqrt(2) * self.grid_size / 2
                traffic_flow = 2.0 - 1.5 * (dist_from_center / max_dist)
                traffic_flow = max(0.5, min(2.0, traffic_flow))
                
                self.intersections[node_id] = Intersection(
                    id=node_id,
                    x=x,
                    y=y,
                    type=node_type,
                    traffic_flow=traffic_flow
                )
                node_id += 1
        
        # 创建道路段 (水平和垂直连接)
        segment_id = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current_id = i * self.grid_size + j
                
                # 水平连接
                if j < self.grid_size - 1:
                    next_id = i * self.grid_size + (j + 1)
                    length = spacing
                    self.road_segments[segment_id] = RoadSegment(
                        id=segment_id,
                        from_node=current_id,
                        to_node=next_id,
                        road_type=RoadType.MAIN_ROAD,  # 临时赋值
                        length=length,
                        construction_difficulty=1.0
                    )
                    segment_id += 1
                
                # 垂直连接
                if i < self.grid_size - 1:
                    next_id = (i + 1) * self.grid_size + j
                    length = spacing
                    self.road_segments[segment_id] = RoadSegment(
                        id=segment_id,
                        from_node=current_id,
                        to_node=next_id,
                        road_type=RoadType.MAIN_ROAD,  # 临时赋值
                        length=length,
                        construction_difficulty=1.0
                    )
                    segment_id += 1
    
    def _assign_road_types(self):
        """分配道路类型"""
        for segment in self.road_segments.values():
            # 根据位置分配道路类型
            from_node = self.intersections[segment.from_node]
            to_node = self.intersections[segment.to_node]
            
            # 中心区域更可能是主干道
            center_x = self.grid_size / 2
            center_y = self.grid_size / 2
            mid_x = (from_node.x + to_node.x) / 2 / 100.0
            mid_y = (from_node.y + to_node.y) / 2 / 100.0
            dist_from_center = np.sqrt((mid_x - center_x)**2 + (mid_y - center_y)**2)
            
            rand = random.random()
            if dist_from_center < 2:
                # 中心区域
                if rand < 0.3:
                    segment.road_type = RoadType.HIGHWAY
                    segment.construction_difficulty = 2.5
                elif rand < 0.7:
                    segment.road_type = RoadType.MAIN_ROAD
                    segment.construction_difficulty = 1.5
                else:
                    segment.road_type = RoadType.SECONDARY
                    segment.construction_difficulty = 1.0
            else:
                # 边缘区域
                if rand < 0.1:
                    segment.road_type = RoadType.MAIN_ROAD
                    segment.construction_difficulty = 1.5
                elif rand < 0.5:
                    segment.road_type = RoadType.SECONDARY
                    segment.construction_difficulty = 1.0
                else:
                    segment.road_type = RoadType.BRANCH
                    segment.construction_difficulty = 0.8
    
    def _place_cameras(self):
        """放置监控摄像头"""
        camera_id = 0
        for node_id, node in self.intersections.items():
            # 根据交通流量和节点类型决定是否放置摄像头
            probability = node.traffic_flow / 2.0
            if node.type == "hub":
                probability = min(1.0, probability + 0.3)
            
            if random.random() < probability:
                # 确定摄像头类型
                rand = random.random()
                if rand < 0.6:
                    cam_type = "HD"
                    priority = 2
                elif rand < 0.9:
                    cam_type = "4K"
                    priority = 3
                else:
                    cam_type = "PTZ"
                    priority = 4
                
                self.cameras[camera_id] = CameraNode(
                    id=camera_id,
                    intersection_id=node_id,
                    camera_type=cam_type,
                    priority=priority
                )
                camera_id += 1
    
    def _place_hubs(self):
        """放置通信汇聚节点"""
        hub_id = 0
        for node_id, node in self.intersections.items():
            if node.type == "hub":
                self.hubs[hub_id] = CommunicationHub(
                    id=hub_id,
                    intersection_id=node_id,
                    capacity=random.randint(8, 15)
                )
                hub_id += 1
        
        # 确保至少有一个汇聚节点
        if len(self.hubs) == 0:
            center_id = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)
            self.hubs[0] = CommunicationHub(
                id=0,
                intersection_id=center_id,
                capacity=20
            )
    
    def get_network_graph(self) -> nx.Graph:
        """获取 NetworkX 图"""
        G = nx.Graph()
        
        # 添加节点
        for node_id, node in self.intersections.items():
            G.add_node(node_id, pos=(node.x, node.y), type=node.type)
        
        # 添加边
        for seg_id, seg in self.road_segments.items():
            cost = seg.get_cable_cost()
            G.add_edge(
                seg.from_node,
                seg.to_node,
                weight=cost,
                length=seg.length,
                road_type=seg.road_type.value,
                segment_id=seg_id
            )
        
        return G
    
    def get_camera_positions(self) -> Dict[int, Tuple[float, float]]:
        """获取所有摄像头位置"""
        positions = {}
        for cam in self.cameras.values():
            node = self.intersections[cam.intersection_id]
            positions[cam.id] = (node.x, node.y)
        return positions
    
    def get_hub_positions(self) -> Dict[int, Tuple[float, float]]:
        """获取所有汇聚节点位置"""
        positions = {}
        for hub in self.hubs.values():
            node = self.intersections[hub.intersection_id]
            positions[hub.id] = (node.x, node.y)
        return positions


# ============================================================================
# 优化算法
# ============================================================================

class CableRoutingOptimizer:
    """电缆布线优化器"""
    
    def __init__(self, network: TransportationNetwork):
        self.network = network
        self.G = network.get_network_graph()
        self.results = {}
    
    def optimize_with_mst(self, cable_type: str = "standard") -> Dict:
        """使用最小生成树优化主干网络"""
        # 计算边的权重
        for u, v, data in self.G.edges(data=True):
            segment = self.network.road_segments[data['segment_id']]
            data['weight'] = segment.get_cable_cost(cable_type)
        
        # 计算 MST
        mst = nx.minimum_spanning_tree(self.G, weight='weight')
        
        # 计算总成本
        total_cost = sum(data['weight'] for u, v, data in mst.edges(data=True))
        total_length = sum(data['length'] for u, v, data in mst.edges(data=True))
        
        self.results['mst'] = {
            'edges': list(mst.edges()),
            'total_cost': total_cost,
            'total_length': total_length,
            'num_edges': mst.number_of_edges()
        }
        
        return self.results['mst']
    
    def optimize_camera_to_hub(self, assignment_strategy: str = "nearest") -> Dict:
        """优化摄像头到汇聚节点的连接"""
        camera_positions = self.network.get_camera_positions()
        hub_positions = self.network.get_hub_positions()
        
        assignments = {}
        total_cost = 0
        
        for cam_id, cam_pos in camera_positions.items():
            cam_node_id = self.network.cameras[cam_id].intersection_id
            
            if assignment_strategy == "nearest":
                # 分配到最近的汇聚节点
                best_hub = None
                best_dist = float('inf')
                
                for hub_id, hub_pos in hub_positions.items():
                    hub_node_id = self.network.hubs[hub_id].intersection_id
                    # 使用图上的最短路径距离
                    try:
                        path_length = nx.shortest_path_length(
                            self.G, cam_node_id, hub_node_id, weight='length'
                        )
                        if path_length < best_dist:
                            best_dist = path_length
                            best_hub = hub_id
                    except nx.NetworkXNoPath:
                        continue
                
                if best_hub is not None:
                    assignments[cam_id] = best_hub
                    # 计算连接成本 (假设直连，沿道路)
                    cost = best_dist * 50.0  # 50 元/米
                    total_cost += cost
            
            elif assignment_strategy == "balanced":
                # 负载均衡分配
                hub_loads = {hub_id: 0 for hub_id in hub_positions}
                
                # 按优先级排序摄像头
                sorted_cams = sorted(
                    self.network.cameras.values(),
                    key=lambda c: c.priority,
                    reverse=True
                )
                
                for cam in sorted_cams:
                    cam_node_id = cam.intersection_id
                    
                    # 找到容量允许且最近的汇聚节点
                    best_hub = None
                    best_score = float('inf')
                    
                    for hub_id, hub in self.network.hubs.items():
                        if hub_loads[hub_id] >= hub.capacity:
                            continue
                        
                        hub_node_id = hub.intersection_id
                        try:
                            path_length = nx.shortest_path_length(
                                self.G, cam_node_id, hub_node_id, weight='length'
                            )
                            # 评分 = 距离 + 负载惩罚
                            score = path_length + hub_loads[hub_id] * 10
                            if score < best_score:
                                best_score = score
                                best_hub = hub_id
                        except nx.NetworkXNoPath:
                            continue
                    
                    if best_hub is not None:
                        assignments[cam.id] = best_hub
                        hub_loads[best_hub] += 1
        
        self.results['camera_assignment'] = {
            'assignments': assignments,
            'total_cost': total_cost,
            'num_cameras': len(assignments),
            'strategy': assignment_strategy
        }
        
        return self.results['camera_assignment']
    
    def optimize_with_vns(self, max_iterations: int = 100) -> Dict:
        """使用变邻域搜索优化路径"""
        # 从 MST 解开始
        if 'mst' not in self.results:
            self.optimize_with_mst()
        
        current_edges = set(self.results['mst']['edges'])
        current_cost = self.results['mst']['total_cost']
        
        best_edges = current_edges.copy()
        best_cost = current_cost
        
        history = [(0, current_cost)]
        
        for iteration in range(max_iterations):
            # 邻域操作：尝试替换一条边
            if len(current_edges) > 0:
                # 随机选择一条边移除
                edge_to_remove = random.choice(list(current_edges))
                current_edges.remove(edge_to_remove)
                
                # 检查连通性，如果不连通则找到替代边
                temp_G = nx.Graph()
                temp_G.add_edges_from(current_edges)
                
                if not nx.is_connected(temp_G):
                    # 找到连接两个连通分量的最佳边
                    components = list(nx.connected_components(temp_G))
                    best_new_edge = None
                    best_new_cost = float('inf')
                    
                    for u, v, data in self.G.edges(data=True):
                        if (u, v) in current_edges or (v, u) in current_edges:
                            continue
                        
                        # 检查这条边是否连接两个不同连通分量
                        for i, comp1 in enumerate(components):
                            for j, comp2 in enumerate(components):
                                if i >= j:
                                    continue
                                if u in comp1 and v in comp2 or u in comp2 and v in comp1:
                                    if data['weight'] < best_new_cost:
                                        best_new_cost = data['weight']
                                        best_new_edge = (u, v)
                    
                    if best_new_edge:
                        current_edges.add(best_new_edge)
                    else:
                        # 恢复原边
                        current_edges.add(edge_to_remove)
                else:
                    # 连通，接受这个变化
                    new_cost = sum(
                        self.G[u][v]['weight'] for u, v in current_edges
                    )
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_edges = current_edges.copy()
                    else:
                        # 以一定概率接受劣解
                        if random.random() < 0.1:
                            current_cost = new_cost
                        else:
                            # 恢复
                            current_edges.add(edge_to_remove)
                            current_cost = self.results['mst']['total_cost']
            
            history.append((iteration + 1, current_cost))
        
        self.results['vns'] = {
            'edges': list(best_edges),
            'total_cost': best_cost,
            'improvement': (current_cost - best_cost) / current_cost * 100,
            'iterations': max_iterations,
            'history': history
        }
        
        return self.results['vns']


# ============================================================================
# 可视化
# ============================================================================

class TransportationVisualizer:
    """交通网络可视化"""
    
    def __init__(self, network: TransportationNetwork, optimizer: CableRoutingOptimizer):
        self.network = network
        self.optimizer = optimizer
        self.fig = None
        
    def plot_network(self, show_cables: bool = True, save_path: str = None):
        """绘制交通网络和电缆布线"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('交通网络布线优化 - Transportation Network Cable Routing', 
                    fontsize=16, fontweight='bold')
        
        # 子图 1: 原始道路网络
        ax1 = axes[0, 0]
        self._plot_road_network(ax1)
        ax1.set_title('1. 原始道路网络\nOriginal Road Network', fontsize=12)
        
        # 子图 2: MST 优化结果
        ax2 = axes[0, 1]
        self._plot_road_network(ax2)
        if 'mst' in self.optimizer.results:
            self._plot_cables(ax2, self.optimizer.results['mst']['edges'], 'blue')
        ax2.set_title('2. MST 主干网络\nMST Backbone Network', fontsize=12)
        
        # 子图 3: 摄像头分配
        ax3 = axes[1, 0]
        self._plot_camera_assignments(ax3)
        ax3.set_title('3. 摄像头 - 汇聚节点分配\nCamera-Hub Assignment', fontsize=12)
        
        # 子图 4: 成本分析
        ax4 = axes[1, 1]
        self._plot_cost_analysis(ax4)
        ax4.set_title('4. 成本分析\nCost Analysis', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图表已保存：{save_path}")
        
        plt.show()
    
    def _plot_road_network(self, ax):
        """绘制道路网络"""
        # 绘制道路段
        for seg in self.network.road_segments.values():
            from_node = self.network.intersections[seg.from_node]
            to_node = self.network.intersections[seg.to_node]
            
            color_map = {
                RoadType.HIGHWAY: 'red',
                RoadType.MAIN_ROAD: 'orange',
                RoadType.SECONDARY: 'yellow',
                RoadType.BRANCH: 'gray'
            }
            width_map = {
                RoadType.HIGHWAY: 3,
                RoadType.MAIN_ROAD: 2.5,
                RoadType.SECONDARY: 2,
                RoadType.BRANCH: 1.5
            }
            
            ax.plot(
                [from_node.x, to_node.x],
                [from_node.y, to_node.y],
                color=color_map[seg.road_type],
                linewidth=width_map[seg.road_type],
                alpha=0.6,
                label=f'{seg.road_type.value}' if seg.id == 0 else ""
            )
        
        # 绘制路口节点
        for node in self.network.intersections.values():
            color = 'green' if node.type == 'hub' else ('blue' if node.type == 'terminal' else 'gray')
            size = 80 if node.type == 'hub' else 40
            ax.scatter(node.x, node.y, c=color, s=size, zorder=5, edgecolors='white')
        
        # 图例
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_cables(self, ax, edges, color='blue'):
        """绘制电缆路径"""
        for u, v in edges:
            from_node = self.network.intersections[u]
            to_node = self.network.intersections[v]
            ax.plot(
                [from_node.x, to_node.x],
                [from_node.y, to_node.y],
                color=color,
                linewidth=2,
                alpha=0.8,
                linestyle='--'
            )
    
    def _plot_camera_assignments(self, ax):
        """绘制摄像头分配"""
        # 绘制道路网络背景
        self._plot_road_network(ax)
        
        # 绘制汇聚节点
        hub_positions = self.network.get_hub_positions()
        for hub_id, (x, y) in hub_positions.items():
            hub = self.network.hubs[hub_id]
            ax.scatter(x, y, c='red', s=200, marker='s', zorder=10, 
                      label=f'汇聚节点 {hub_id}' if hub_id == 0 else "")
            ax.annotate(f'H{hub_id}', (x, y+10), fontsize=9, ha='center')
        
        # 绘制摄像头和连接线
        if 'camera_assignment' in self.optimizer.results:
            assignments = self.optimizer.results['camera_assignment']['assignments']
            for cam_id, hub_id in assignments.items():
                cam = self.network.cameras[cam_id]
                cam_node = self.network.intersections[cam.intersection_id]
                hub_node = self.network.intersections[self.network.hubs[hub_id].intersection_id]
                
                # 绘制摄像头
                color_map = {'HD': 'cyan', '4K': 'magenta', 'PTZ': 'purple'}
                ax.scatter(cam_node.x, cam_node.y, c=color_map.get(cam.camera_type, 'cyan'),
                          s=60, marker='o', zorder=8)
                
                # 绘制连接线
                ax.plot(
                    [cam_node.x, hub_node.x],
                    [cam_node.y, hub_node.y],
                    color='cyan',
                    linewidth=1,
                    alpha=0.4,
                    linestyle=':'
                )
        
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_cost_analysis(self, ax):
        """绘制成本分析"""
        categories = []
        costs = []
        
        if 'mst' in self.optimizer.results:
            categories.append('MST 主干')
            costs.append(self.optimizer.results['mst']['total_cost'])
        
        if 'camera_assignment' in self.optimizer.results:
            categories.append('摄像头连接')
            costs.append(self.optimizer.results['camera_assignment']['total_cost'])
        
        if 'vns' in self.optimizer.results:
            categories.append('VNS 优化后')
            costs.append(self.optimizer.results['vns']['total_cost'])
        
        if len(categories) > 0:
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(categories)))
            bars = ax.bar(categories, costs, color=colors)
            
            # 添加数值标签
            for bar, cost in zip(bars, costs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'¥{cost:.0f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('成本 (元)')
            ax.set_title('布线成本分解')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加总成本
            total = sum(costs)
            ax.text(0.5, -0.15, f'总成本：¥{total:.0f}', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', ha='center')


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("交通网络布线优化系统")
    print("Transportation Network Cable Routing Optimization")
    print("=" * 60)
    
    # 创建交通网络
    print("\n1. 创建交通网络模型...")
    network = TransportationNetwork(grid_size=8, seed=42)
    print(f"   - 路口节点：{len(network.intersections)} 个")
    print(f"   - 道路段：{len(network.road_segments)} 条")
    print(f"   - 摄像头：{len(network.cameras)} 个")
    print(f"   - 汇聚节点：{len(network.hubs)} 个")
    
    # 统计道路类型
    road_type_counts = {}
    for seg in network.road_segments.values():
        road_type_counts[seg.road_type.value] = road_type_counts.get(seg.road_type.value, 0) + 1
    print(f"   - 道路类型分布：{road_type_counts}")
    
    # 创建优化器
    print("\n2. 执行优化算法...")
    optimizer = CableRoutingOptimizer(network)
    
    # MST 优化
    print("   - 运行 MST 算法...")
    mst_result = optimizer.optimize_with_mst()
    print(f"     • MST 总成本：¥{mst_result['total_cost']:.2f}")
    print(f"     • 总长度：{mst_result['total_length']:.1f} 米")
    
    # 摄像头分配
    print("   - 分配摄像头到汇聚节点...")
    camera_result = optimizer.optimize_camera_to_hub(assignment_strategy="balanced")
    print(f"     • 已分配摄像头：{camera_result['num_cameras']} 个")
    print(f"     • 连接成本：¥{camera_result['total_cost']:.2f}")
    
    # VNS 优化
    print("   - 运行 VNS 局部优化...")
    vns_result = optimizer.optimize_with_vns(max_iterations=50)
    print(f"     • VNS 优化后成本：¥{vns_result['total_cost']:.2f}")
    print(f"     • 改进幅度：{vns_result['improvement']:.2f}%")
    
    # 可视化
    print("\n3. 生成可视化结果...")
    visualizer = TransportationVisualizer(network, optimizer)
    save_path = "/root/.openclaw/workspace/cable-optimization/outputs/transportation_network.png"
    visualizer.plot_network(save_path=save_path)
    
    # 输出总结
    print("\n" + "=" * 60)
    print("优化结果总结")
    print("=" * 60)
    total_cost = mst_result['total_cost'] + camera_result['total_cost']
    print(f"主干网络成本：¥{mst_result['total_cost']:.2f}")
    print(f"摄像头连接成本：¥{camera_result['total_cost']:.2f}")
    print(f"总成本：¥{total_cost:.2f}")
    print(f"VNS 优化改进：{vns_result['improvement']:.2f}%")
    print(f"\n输出文件：{save_path}")
    print("=" * 60)
    
    return optimizer.results


if __name__ == "__main__":
    results = main()
