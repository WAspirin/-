#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铁路网络信号与通信电缆布线优化
Railway Network Signal and Communication Cable Routing Optimization

Day 31: 新案例研究 - 铁路网络

问题描述:
- 铁路沿线信号系统电缆布线
- 车站 - 信号机 - 道岔的通信连接
- 考虑铁路安全约束 (冗余设计、防火要求)
- 优化电缆长度和铺设成本

作者：智子 (Sophon)
日期：2026-03-31
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import heapq


# ==================== 数据结构定义 ====================

class CableType(Enum):
    """电缆类型枚举"""
    SIGNAL_NORMAL = "信号电缆 - 普通"  # 普通信号电缆
    SIGNAL_FIRE = "信号电缆 - 阻燃"   # 阻燃信号电缆 (隧道用)
    COMMUNICATION = "通信电缆"        # 通信电缆
    POWER = "电源电缆"               # 电源电缆
    
    @property
    def cost_per_meter(self) -> float:
        """每米成本 (元)"""
        costs = {
            CableType.SIGNAL_NORMAL: 150,
            CableType.SIGNAL_FIRE: 280,
            CableType.COMMUNICATION: 200,
            CableType.POWER: 180
        }
        return costs[self]
    
    @property
    def max_distance(self) -> float:
        """最大传输距离 (米)"""
        distances = {
            CableType.SIGNAL_NORMAL: 2000,
            CableType.SIGNAL_FIRE: 2000,
            CableType.COMMUNICATION: 5000,
            CableType.POWER: 1000
        }
        return distances[self]


@dataclass
class RailwayNode:
    """铁路节点 (车站/信号机/道岔)"""
    id: int
    name: str
    x: float  # 位置 (km)
    y: float  # 位置 (km)
    node_type: str  # 'station', 'signal', 'switch', 'crossing'
    priority: int = 1  # 优先级 (1-5, 5 最高)
    has_redundancy: bool = False  # 是否需要冗余连接
    
    def __post_init__(self):
        if self.node_type == 'station':
            self.priority = 5
            self.has_redundancy = True
        elif self.node_type == 'signal':
            self.priority = 4
            self.has_redundancy = True
        elif self.node_type == 'switch':
            self.priority = 3
            self.has_redundancy = False


@dataclass
class TrackSection:
    """轨道区段"""
    id: int
    start_node: int
    end_node: int
    length: float  # 长度 (km)
    track_type: str = 'main'  # 'main', 'branch', 'tunnel', 'bridge'
    has_cable_tray: bool = False  # 是否有电缆槽
    
    @property
    def construction_difficulty(self) -> float:
        """施工难度系数"""
        difficulties = {
            'main': 1.0,
            'branch': 1.2,
            'tunnel': 2.5,
            'bridge': 2.0
        }
        return difficulties.get(self.track_type, 1.0)


@dataclass
class CableRoute:
    """电缆路由"""
    id: int
    start_node: int
    end_node: int
    cable_type: CableType
    length: float  # 长度 (米)
    cost: float  # 总成本
    is_redundant: bool = False  # 是否为冗余路由
    path: List[int] = field(default_factory=list)  # 路径节点


# ==================== 铁路网络建模 ====================

class RailwayNetwork:
    """铁路网络模型"""
    
    def __init__(self, name: str = "铁路网络"):
        self.name = name
        self.nodes: Dict[int, RailwayNode] = {}
        self.track_sections: List[TrackSection] = []
        self.cable_routes: List[CableRoute] = []
        self.adjacency: Dict[int, List[Tuple[int, float]]] = {}
        
    def add_node(self, node: RailwayNode):
        """添加节点"""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
    
    def add_track(self, section: TrackSection):
        """添加轨道区段"""
        self.track_sections.append(section)
        # 构建邻接表
        distance = section.length * 1000  # km → m
        self.adjacency[section.start_node].append((section.end_node, distance))
        self.adjacency[section.end_node].append((section.start_node, distance))
    
    def get_distance(self, node1: int, node2: int) -> float:
        """获取两节点间的欧氏距离 (米)"""
        n1, n2 = self.nodes[node1], self.nodes[node2]
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2) * 1000
    
    def get_track_type(self, node1: int, node2: int) -> str:
        """获取轨道类型"""
        for section in self.track_sections:
            if (section.start_node == node1 and section.end_node == node2) or \
               (section.start_node == node2 and section.end_node == node1):
                return section.track_type
        return 'main'


# ==================== 优化算法 ====================

class CableRoutingOptimizer:
    """电缆布线优化器"""
    
    def __init__(self, network: RailwayNetwork):
        self.network = network
        self.optimized_routes: List[CableRoute] = []
    
    def dijkstra(self, start: int, end: int) -> Tuple[List[int], float]:
        """Dijkstra 最短路径算法"""
        dist = {node_id: float('inf') for node_id in self.network.nodes}
        dist[start] = 0
        prev = {node_id: None for node_id in self.network.nodes}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            
            if u == end:
                break
            
            for v, weight in self.network.adjacency.get(u, []):
                if v not in visited:
                    new_dist = dist[u] + weight
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        prev[v] = u
                        heapq.heappush(pq, (new_dist, v))
        
        # 重构路径
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        
        return path, dist[end]
    
    def select_cable_type(self, node1: int, node2: int, track_type: str) -> CableType:
        """根据场景选择电缆类型"""
        # 隧道内必须用阻燃电缆
        if track_type == 'tunnel':
            return CableType.SIGNAL_FIRE
        
        # 根据节点类型选择
        n1, n2 = self.network.nodes[node1], self.network.nodes[node2]
        
        # 车站连接用通信电缆
        if n1.node_type == 'station' or n2.node_type == 'station':
            return CableType.COMMUNICATION
        
        # 信号机连接用信号电缆
        if n1.node_type == 'signal' or n2.node_type == 'signal':
            return CableType.SIGNAL_NORMAL
        
        return CableType.SIGNAL_NORMAL
    
    def calculate_cost(self, length: float, cable_type: CableType, 
                       track_type: str) -> float:
        """计算电缆成本"""
        base_cost = length * cable_type.cost_per_meter
        difficulty_factor = {
            'main': 1.0,
            'branch': 1.2,
            'tunnel': 2.5,
            'bridge': 2.0
        }.get(track_type, 1.0)
        
        return base_cost * difficulty_factor
    
    def optimize_primary_routes(self):
        """优化主用路由 (MST 思想)"""
        route_id = 0
        stations = [n.id for n in self.network.nodes.values() 
                   if n.node_type == 'station']
        
        if not stations:
            print("⚠️  未找到车站节点")
            return
        
        # 以第一个车站为根，构建到所有其他节点的连接
        root = stations[0]
        connected = {root}
        
        while len(connected) < len(self.network.nodes):
            best_edge = None
            best_cost = float('inf')
            
            # 找到连接已连接集合和未连接集合的最优边
            for node_id in connected:
                for neighbor, distance in self.network.adjacency.get(node_id, []):
                    if neighbor not in connected:
                        track_type = self.network.get_track_type(node_id, neighbor)
                        cable_type = self.select_cable_type(node_id, neighbor, track_type)
                        cost = self.calculate_cost(distance, cable_type, track_type)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_edge = (node_id, neighbor, distance, cable_type, track_type)
            
            if best_edge:
                node1, node2, distance, cable_type, track_type = best_edge
                path, _ = self.dijkstra(node1, node2)
                
                route = CableRoute(
                    id=route_id,
                    start_node=node1,
                    end_node=node2,
                    cable_type=cable_type,
                    length=distance,
                    cost=best_cost,
                    is_redundant=False,
                    path=path
                )
                self.optimized_routes.append(route)
                connected.add(node2)
                route_id += 1
    
    def add_redundant_routes(self):
        """添加冗余路由 (提高可靠性)"""
        route_id = len(self.optimized_routes)
        
        for node in self.network.nodes.values():
            if node.has_redundancy:
                # 找到该节点的主要连接
                primary_routes = [r for r in self.optimized_routes 
                                 if node.id in [r.start_node, r.end_node]]
                
                if len(primary_routes) >= 1:
                    # 找到次优连接作为冗余
                    primary_neighbors = set()
                    for route in primary_routes:
                        if route.start_node == node.id:
                            primary_neighbors.add(route.end_node)
                        else:
                            primary_neighbors.add(route.start_node)
                    
                    # 找另一个邻居作为冗余路径
                    for neighbor, distance in self.network.adjacency.get(node.id, []):
                        if neighbor not in primary_neighbors:
                            track_type = self.network.get_track_type(node.id, neighbor)
                            cable_type = self.select_cable_type(node.id, neighbor, track_type)
                            cost = self.calculate_cost(distance, cable_type, track_type)
                            
                            path, _ = self.dijkstra(node.id, neighbor)
                            
                            route = CableRoute(
                                id=route_id,
                                start_node=node.id,
                                end_node=neighbor,
                                cable_type=cable_type,
                                length=distance,
                                cost=cost,
                                is_redundant=True,
                                path=path
                            )
                            self.optimized_routes.append(route)
                            route_id += 1
                            break
    
    def optimize(self):
        """执行完整优化"""
        print("🚂 开始铁路网络电缆布线优化...")
        print(f"   节点数：{len(self.network.nodes)}")
        print(f"   轨道区段数：{len(self.network.track_sections)}")
        
        self.optimize_primary_routes()
        self.add_redundant_routes()
        
        print(f"   优化完成：生成 {len(self.optimized_routes)} 条电缆路由")
        
        return self.optimized_routes


# ==================== 可视化 ====================

class RailwayVisualizer:
    """铁路网络可视化"""
    
    def __init__(self, network: RailwayNetwork, routes: List[CableRoute]):
        self.network = network
        self.routes = routes
        self.fig = None
        
    def plot_network(self, save_path: str = "outputs/railway_network_topology.png"):
        """绘制网络拓扑图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('铁路网络信号与通信电缆布线优化', fontsize=16, fontweight='bold')
        
        # 子图 1: 物理拓扑
        ax1 = axes[0, 0]
        self._plot_physical_topology(ax1)
        
        # 子图 2: 电缆类型分布
        ax2 = axes[0, 1]
        self._plot_cable_distribution(ax2)
        
        # 子图 3: 成本分析
        ax3 = axes[1, 0]
        self._plot_cost_analysis(ax3)
        
        # 子图 4: 可靠性分析
        ax4 = axes[1, 1]
        self._plot_reliability_analysis(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 可视化已保存：{save_path}")
    
    def _plot_physical_topology(self, ax):
        """绘制物理拓扑"""
        # 绘制轨道
        for section in self.network.track_sections:
            n1 = self.network.nodes[section.start_node]
            n2 = self.network.nodes[section.end_node]
            
            color = {'main': '#333333', 'branch': '#666666', 
                    'tunnel': '#8B4513', 'bridge': '#4682B4'}
            ax.plot([n1.x, n2.x], [n1.y, n2.y], 
                   color=color.get(section.track_type, 'gray'), 
                   linewidth=2, alpha=0.6)
        
        # 绘制节点
        node_colors = {'station': '#FF4444', 'signal': '#44AA44', 
                      'switch': '#4444FF', 'crossing': '#FFAA00'}
        node_sizes = {'station': 500, 'signal': 200, 
                     'switch': 150, 'crossing': 180}
        
        for node in self.network.nodes.values():
            ax.scatter(node.x, node.y, 
                      c=node_colors.get(node.node_type, 'gray'),
                      s=node_sizes.get(node.node_type, 100),
                      marker='o', edgecolors='white', linewidth=1.5,
                      label=f"{node.node_type}" if node.id == 0 else "")
            ax.annotate(node.name, (node.x, node.y), 
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8)
        
        # 绘制电缆路由
        for route in self.routes:
            if not route.is_redundant:
                n1 = self.network.nodes[route.start_node]
                n2 = self.network.nodes[route.end_node]
                ax.plot([n1.x, n2.x], [n1.y, n2.y], 
                       'r-', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('位置 (km)')
        ax.set_ylabel('位置 (km)')
        ax.set_title('物理拓扑与电缆路由')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_cable_distribution(self, ax):
        """绘制电缆类型分布"""
        cable_lengths = {}
        for route in self.routes:
            cable_name = route.cable_type.value
            if cable_name not in cable_lengths:
                cable_lengths[cable_name] = 0
            cable_lengths[cable_name] += route.length
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        ax.pie(cable_lengths.values(), labels=cable_lengths.keys(),
              autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('电缆类型长度分布')
    
    def _plot_cost_analysis(self, ax):
        """绘制成本分析"""
        primary_cost = sum(r.cost for r in self.routes if not r.is_redundant)
        redundant_cost = sum(r.cost for r in self.routes if r.is_redundant)
        
        categories = ['主用路由', '冗余路由']
        costs = [primary_cost, redundant_cost]
        colors = ['#4CAF50', '#FF9800']
        
        bars = ax.bar(categories, costs, color=colors)
        ax.set_ylabel('成本 (元)')
        ax.set_title('电缆布线成本分析')
        
        # 添加数值标签
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                   f'¥{cost:,.0f}', ha='center', va='bottom', fontsize=10)
        
        total = primary_cost + redundant_cost
        ax.text(0.5, -0.3, f'总成本：¥{total:,.0f}\n冗余比：{redundant_cost/primary_cost*100:.1f}%',
               transform=ax.transAxes, ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_reliability_analysis(self, ax):
        """绘制可靠性分析"""
        # 统计各节点的连接数
        node_connections = {node_id: 0 for node_id in self.network.nodes}
        for route in self.routes:
            node_connections[route.start_node] += 1
            node_connections[route.end_node] += 1
        
        # 按节点类型分组
        type_avg = {}
        for node in self.network.nodes.values():
            if node.node_type not in type_avg:
                type_avg[node.node_type] = []
            type_avg[node.node_type].append(node_connections[node.id])
        
        categories = list(type_avg.keys())
        averages = [np.mean(v) for v in type_avg.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        bars = ax.bar(categories, averages, color=colors)
        ax.set_ylabel('平均连接数')
        ax.set_title('节点连接可靠性分析')
        ax.set_ylim(0, max(averages) * 1.5)
        
        # 添加数值标签
        for bar, avg in zip(bars, averages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{avg:.2f}', ha='center', va='bottom', fontsize=9)


# ==================== 测试用例生成 ====================

def create_test_network() -> RailwayNetwork:
    """创建测试铁路网络"""
    network = RailwayNetwork("测试铁路网络")
    
    # 创建车站 (5 个)
    stations = [
        RailwayNode(0, "北京站", 0, 0, 'station'),
        RailwayNode(1, "天津站", 120, 30, 'station'),
        RailwayNode(2, "济南站", 280, 20, 'station'),
        RailwayNode(3, "徐州站", 420, -30, 'station'),
        RailwayNode(4, "南京站", 580, 10, 'station'),
    ]
    
    # 创建信号机 (10 个)
    signals = [
        RailwayNode(5, "信号 1 号", 60, 15, 'signal'),
        RailwayNode(6, "信号 2 号", 180, 25, 'signal'),
        RailwayNode(7, "信号 3 号", 350, -10, 'signal'),
        RailwayNode(8, "信号 4 号", 500, 0, 'signal'),
        RailwayNode(9, "信号 5 号", 80, -20, 'signal'),
        RailwayNode(10, "信号 6 号", 220, 40, 'signal'),
        RailwayNode(11, "信号 7 号", 400, 30, 'signal'),
        RailwayNode(12, "信号 8 号", 520, -20, 'signal'),
        RailwayNode(13, "信号 9 号", 150, -30, 'signal'),
        RailwayNode(14, "信号 10 号", 320, 35, 'signal'),
    ]
    
    # 创建道岔 (5 个)
    switches = [
        RailwayNode(15, "道岔 1 号", 100, 10, 'switch'),
        RailwayNode(16, "道岔 2 号", 250, 15, 'switch'),
        RailwayNode(17, "道岔 3 号", 380, -20, 'switch'),
        RailwayNode(18, "道岔 4 号", 480, 15, 'switch'),
        RailwayNode(19, "道岔 5 号", 200, -15, 'switch'),
    ]
    
    # 添加所有节点
    for node in stations + signals + switches:
        network.add_node(node)
    
    # 创建轨道区段 (主干线)
    main_tracks = [
        TrackSection(0, 0, 1, 120, 'main'),
        TrackSection(1, 1, 2, 160, 'main'),
        TrackSection(2, 2, 3, 180, 'main'),
        TrackSection(3, 3, 4, 170, 'main'),
    ]
    
    # 支线
    branch_tracks = [
        TrackSection(4, 1, 6, 50, 'branch'),
        TrackSection(5, 2, 10, 40, 'branch'),
        TrackSection(6, 3, 11, 60, 'branch'),
    ]
    
    # 隧道段
    tunnel_tracks = [
        TrackSection(7, 5, 7, 80, 'tunnel'),
        TrackSection(8, 14, 11, 60, 'tunnel'),
    ]
    
    # 桥梁段
    bridge_tracks = [
        TrackSection(9, 7, 8, 90, 'bridge'),
        TrackSection(10, 16, 17, 70, 'bridge'),
    ]
    
    # 添加所有轨道
    for track in main_tracks + branch_tracks + tunnel_tracks + bridge_tracks:
        network.add_track(track)
    
    return network


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("铁路网络信号与通信电缆布线优化系统")
    print("Railway Network Cable Routing Optimization System")
    print("=" * 60)
    print()
    
    # 创建网络
    network = create_test_network()
    
    # 优化
    optimizer = CableRoutingOptimizer(network)
    routes = optimizer.optimize()
    
    print()
    print("📊 优化结果统计:")
    print("-" * 40)
    
    total_length = sum(r.length for r in routes)
    total_cost = sum(r.cost for r in routes)
    primary_cost = sum(r.cost for r in routes if not r.is_redundant)
    redundant_cost = sum(r.cost for r in routes if r.is_redundant)
    
    print(f"   电缆路由总数：{len(routes)} 条")
    print(f"   主用路由：{len([r for r in routes if not r.is_redundant])} 条")
    print(f"   冗余路由：{len([r for r in routes if r.is_redundant])} 条")
    print(f"   总长度：{total_length/1000:.2f} km")
    print(f"   总成本：¥{total_cost:,.0f} 元")
    print(f"   主用成本：¥{primary_cost:,.0f} 元")
    print(f"   冗余成本：¥{redundant_cost:,.0f} 元")
    print(f"   冗余比：{redundant_cost/primary_cost*100:.1f}%")
    
    print()
    print("📋 电缆类型统计:")
    print("-" * 40)
    cable_stats = {}
    for route in routes:
        cable_name = route.cable_type.value
        if cable_name not in cable_stats:
            cable_stats[cable_name] = {'length': 0, 'cost': 0, 'count': 0}
        cable_stats[cable_name]['length'] += route.length
        cable_stats[cable_name]['cost'] += route.cost
        cable_stats[cable_name]['count'] += 1
    
    for cable_name, stats in cable_stats.items():
        print(f"   {cable_name}:")
        print(f"      路由数：{stats['count']} 条")
        print(f"      总长度：{stats['length']/1000:.2f} km")
        print(f"      总成本：¥{stats['cost']:,.0f} 元")
    
    # 可视化
    visualizer = RailwayVisualizer(network, routes)
    visualizer.plot_network()
    
    print()
    print("=" * 60)
    print("✅ 优化完成！")
    print("=" * 60)
    
    return network, routes


if __name__ == "__main__":
    main()
