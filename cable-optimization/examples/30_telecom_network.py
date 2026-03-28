#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电信网络布线优化 - 光纤网络设计
Telecommunications Network Optimization - Fiber Optic Network Design

Day 28: 新增案例研究 - 电信网络优化

特性:
- 环形拓扑设计 (自愈网络)
- 多 commodity 流优化
- 延迟与带宽约束
- 冗余路径设计
- 成本 - 可靠性权衡分析

作者：智子 (Sophon)
日期：2026-03-28
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import heapq


# ============================================================================
# 数据结构定义
# ============================================================================

class CableType(Enum):
    """光纤电缆类型"""
    SINGLE_MODE = "单模光纤"  # 长距离，高带宽
    MULTI_MODE = "多模光纤"   # 短距离，成本低
    COAXIAL = "同轴电缆"      # 短距离，传统网络


@dataclass
class TelecomNode:
    """电信节点"""
    id: int
    x: float
    y: float
    node_type: str  # "core", "distribution", "access"
    demand: float = 0.0  # Gbps 带宽需求
    priority: int = 1  # 1-5，5 最高
    
    def distance_to(self, other: 'TelecomNode') -> float:
        """计算到另一节点的欧氏距离"""
        return np.sqrt((self.x - other.x)**2 + **(self.y - other.y)2)


@dataclass
class FiberCable:
    """光纤电缆"""
    cable_type: CableType
    capacity: float  # Gbps
    cost_per_km: float
    latency_per_km: float  # ms/km
    max_distance: float  # km
    
    def __post_init__(self):
        self.available_capacity = self.capacity


@dataclass
class NetworkLink:
    """网络链路"""
    source: int
    target: int
    cable: FiberCable
    length: float  # km
    used_capacity: float = 0.0
    is_primary: bool = True
    is_backup: bool = False
    
    @property
    def latency(self) -> float:
        """链路延迟 (ms)"""
        return self.length * self.cable.latency_per_km
    
    @property
    def cost(self) -> float:
        """链路成本"""
        return self.length * self.cable.cost_per_km


@dataclass
class TrafficDemand:
    """流量需求"""
    source: int
    destination: int
    bandwidth: float  # Gbps
    max_latency: float  # ms，最大可接受延迟
    priority: int = 3  # 1-5


# ============================================================================
# 电信网络建模
# ============================================================================

class TelecommunicationsNetwork:
    """
    电信网络模型
    
    支持:
    - 多层网络架构 (核心层/汇聚层/接入层)
    - 环形拓扑 (自愈)
    - 冗余路径设计
    - 流量工程
    """
    
    def __init__(self, name: str = "Telecom Network"):
        self.name = name
        self.nodes: Dict[int, TelecomNode] = {}
        self.links: List[NetworkLink] = []
        self.demands: List[TrafficDemand] = []
        self.graph = nx.Graph()
        
        # 光纤电缆类型定义
        self.cable_types = {
            CableType.SINGLE_MODE: FiberCable(
                cable_type=CableType.SINGLE_MODE,
                capacity=100.0,  # 100 Gbps
                cost_per_km=5000,  # ¥5000/km
                latency_per_km=0.005,  # 0.005 ms/km
                max_distance=80.0  # 80 km
            ),
            CableType.MULTI_MODE: FiberCable(
                cable_type=CableType.MULTI_MODE,
                capacity=10.0,  # 10 Gbps
                cost_per_km=2000,
                latency_per_km=0.005,
                max_distance=2.0
            ),
            CableType.COAXIAL: FiberCable(
                cable_type=CableType.COAXIAL,
                capacity=1.0,  # 1 Gbps
                cost_per_km=1000,
                latency_per_km=0.004,
                max_distance=0.5
            )
        }
    
    def add_node(self, node: TelecomNode):
        """添加网络节点"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, pos=(node.x, node.y), **{node_type': node.node_type})
    
    def add_demand(self, demand: TrafficDemand):
        """添加流量需求"""
        self.demands.append(demand)
    
    def generate_city_network(self, size: str = "medium") -> 'TelecommunicationsNetwork':
        """
        生成城市电信网络
        
        Args:
            size: "small" (10 节点), "medium" (20 节点), "large" (50 节点)
        """
        node_counts = {
            "small": {"core": 1, "distribution": 3, "access": 6},
            "medium": {"core": 2, "distribution": 6, "access": 12},
            "large": {"core": 3, "distribution": 10, "access": 37}
        }
        
        counts = node_counts.get(size, node_counts["medium"])
        node_id = 0
        
        # 生成核心节点 (中心区域)
        for i in range(counts["core"]):
            angle = 2 * np.pi * i / counts["core"]
            r = 2.0  # 核心节点靠近中心
            self.add_node(TelecomNode(
                id=node_id,
                x=50 + r * np.cos(angle),
                y=50 + r * np.sin(angle),
                node_type="core",
                demand=0,
                priority=5
            ))
            node_id += 1
        
        # 生成汇聚节点 (中间环)
        for i in range(counts["distribution"]):
            angle = 2 * np.pi * i / counts["distribution"]
            r = 15.0
            self.add_node(TelecomNode(
                id=node_id,
                x=50 + r * np.cos(angle),
                y=50 + r * np.sin(angle),
                node_type="distribution",
                demand=np.random.uniform(5, 15),
                priority=3
            ))
            node_id += 1
        
        # 生成接入节点 (外围环)
        for i in range(counts["access"]):
            angle = 2 * np.pi * i / counts["access"] + np.random.uniform(-0.2, 0.2)
            r = 30 + np.random.uniform(-5, 5)
            self.add_node(TelecomNode(
                id=node_id,
                x=50 + r * np.cos(angle),
                y=50 + r * np.sin(angle),
                node_type="access",
                demand=np.random.uniform(0.5, 3),
                priority=1
            ))
            node_id += 1
        
        # 生成流量需求 (接入节点到核心节点)
        core_nodes = [n.id for n in self.nodes.values() if n.node_type == "core"]
        access_nodes = [n.id for n in self.nodes.values() if n.node_type == "access"]
        
        for access_id in access_nodes:
            core_id = np.random.choice(core_nodes)
            self.add_demand(TrafficDemand(
                source=access_id,
                destination=core_id,
                bandwidth=self.nodes[access_id].demand,
                max_latency=50.0,  # 50ms 最大延迟
                priority=3
            ))
        
        return self
    
    def build_ring_topology(self) -> List[NetworkLink]:
        """
        构建环形拓扑 (自愈网络)
        
        核心层：双环结构
        汇聚层：连接到最近的核心节点
        接入层：连接到最近的汇聚节点
        """
        links = []
        core_nodes = [n for n in self.nodes.values() if n.node_type == "core"]
        dist_nodes = [n for n in self.nodes.values() if n.node_type == "distribution"]
        access_nodes = [n for n in self.nodes.values() if n.node_type == "access"]
        
        # 1. 核心层双环 (高可靠性)
        if len(core_nodes) >= 2:
            # 主环
            for i in range(len(core_nodes)):
                src = core_nodes[i]
                tgt = core_nodes[(i + 1) % len(core_nodes)]
                distance = src.distance_to(tgt) / 1000  # 转换为 km
                
                link = NetworkLink(
                    source=src.id,
                    target=tgt.id,
                    cable=self.cable_types[CableType.SINGLE_MODE],
                    length=distance,
                    is_primary=True
                )
                links.append(link)
            
            # 备用环 (反向)
            for i in range(len(core_nodes)):
                src = core_nodes[i]
                tgt = core_nodes[(i - 1) % len(core_nodes)]
                distance = src.distance_to(tgt) / 1000
                
                link = NetworkLink(
                    source=src.id,
                    target=tgt.id,
                    cable=self.cable_types[CableType.SINGLE_MODE],
                    length=distance,
                    is_backup=True
                )
                links.append(link)
        
        # 2. 汇聚层连接到核心层 (双归属)
        for dist_node in dist_nodes:
            # 找到最近的两个核心节点
            core_distances = [(c, dist_node.distance_to(c)) for c in core_nodes]
            core_distances.sort(key=lambda x: x[1])
            
            for i, (core_node, distance) in enumerate(core_distances[:2]):
                link = NetworkLink(
                    source=dist_node.id,
                    target=core_node.id,
                    cable=self.cable_types[CableType.SINGLE_MODE],
                    length=distance / 1000,
                    is_primary=(i == 0),
                    is_backup=(i == 1)
                )
                links.append(link)
        
        # 3. 接入层连接到汇聚层
        for access_node in access_nodes:
            # 找到最近的两个汇聚节点
            dist_distances = [(d, access_node.distance_to(d)) for d in dist_nodes]
            dist_distances.sort(key=lambda x: x[1])
            
            for i, (dist_node, distance) in enumerate(dist_distances[:2]):
                # 根据距离选择电缆类型
                if distance / 1000 <= 2.0:
                    cable_type = CableType.MULTI_MODE
                else:
                    cable_type = CableType.SINGLE_MODE
                
                link = NetworkLink(
                    source=access_node.id,
                    target=dist_node.id,
                    cable=self.cable_types[cable_type],
                    length=distance / 1000,
                    is_primary=(i == 0),
                    is_backup=(i == 1)
                )
                links.append(link)
        
        self.links = links
        return links
    
    def allocate_traffic(self) -> Dict[str, float]:
        """
        分配流量到链路
        
        使用最短路径算法，考虑带宽约束
        """
        # 构建带权图
        G = nx.Graph()
        for link in self.links:
            if link.is_primary:  # 只考虑主用链路
                G.add_edge(
                    link.source,
                    link.target,
                    weight=link.latency,
                    capacity=link.cable.capacity,
                    link=link
                )
        
        stats = {
            "total_bandwidth": 0.0,
            "allocated_bandwidth": 0.0,
            "blocked_demands": 0,
            "avg_latency": 0.0,
            "total_latency": 0.0
        }
        
        latencies = []
        
        for demand in self.demands:
            try:
                # 最短路径 (延迟最小)
                path = nx.shortest_path(G, demand.source, demand.destination, weight='weight')
                
                # 检查带宽约束
                min_capacity = min(
                    G[path[i]][path[i+1]]['capacity'] - G[path[i]][path[i+1]]['link'].used_capacity
                    for i in range(len(path) - 1)
                )
                
                if min_capacity >= demand.bandwidth:
                    # 分配带宽
                    for i in range(len(path) - 1):
                        link = G[path[i]][path[i+1]]['link']
                        link.used_capacity += demand.bandwidth
                    
                    # 计算路径延迟
                    path_latency = sum(
                        G[path[i]][path[i+1]]['weight']
                        for i in range(len(path) - 1)
                    )
                    latencies.append(path_latency)
                    
                    stats["allocated_bandwidth"] += demand.bandwidth
                    stats["total_latency"] += path_latency
                else:
                    stats["blocked_demands"] += 1
                
                stats["total_bandwidth"] += demand.bandwidth
                
            except nx.NetworkXNoPath:
                stats["blocked_demands"] += 1
                stats["total_bandwidth"] += demand.bandwidth
        
        if latencies:
            stats["avg_latency"] = stats["total_latency"] / len(latencies)
        
        return stats
    
    def calculate_metrics(self) -> Dict[str, float]:
        """计算网络性能指标"""
        total_cost = sum(link.cost for link in self.links)
        total_length = sum(link.length for link in self.links)
        
        # 冗余度
        primary_links = [l for l in self.links if l.is_primary]
        backup_links = [l for l in self.links if l.is_backup]
        redundancy_ratio = len(backup_links) / len(primary_links) if primary_links else 0
        
        # 连通性
        G = nx.Graph()
        for link in self.links:
            G.add_edge(link.source, link.target)
        
        is_connected = nx.is_connected(G)
        
        # 平均节点度
        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if G.nodes() else 0
        
        return {
            "total_cost": total_cost,
            "total_length_km": total_length,
            "num_nodes": len(self.nodes),
            "num_links": len(self.links),
            "num_primary_links": len(primary_links),
            "num_backup_links": len(backup_links),
            "redundancy_ratio": redundancy_ratio,
            "is_connected": is_connected,
            "avg_degree": avg_degree
        }


# ============================================================================
# 可视化
# ============================================================================

class TelecomVisualizer:
    """电信网络可视化"""
    
    def __init__(self, network: TelecommunicationsNetwork):
        self.network = network
        self.fig = None
        self.axes = None
    
    def plot_network_topology(self, save_path: Optional[str] = None):
        """绘制网络拓扑图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.network.name} - 电信网络拓扑', fontsize=16, fontweight='bold')
        
        # 1. 物理拓扑
        ax1 = axes[0, 0]
        self._plot_physical_topology(ax1)
        
        # 2. 流量分布
        ax2 = axes[0, 1]
        self._plot_traffic_distribution(ax2)
        
        # 3. 成本分析
        ax3 = axes[1, 0]
        self._plot_cost_analysis(ax3)
        
        # 4. 可靠性分析
        ax4 = axes[1, 1]
        self._plot_reliability_analysis(ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 拓扑图已保存：{save_path}")
        
        plt.show()
    
    def _plot_physical_topology(self, ax):
        """绘制物理拓扑"""
        # 节点位置
        pos = {node.id: (node.x, node.y) for node in self.network.nodes.values()}
        
        # 绘制链路
        for link in self.network.links:
            src_pos = (self.network.nodes[link.source].x,
                      self.network.nodes[link.source].y)
            tgt_pos = (self.network.nodes[link.target].x,
                      self.network.nodes[link.target].y)
            
            if link.is_backup:
                ax.plot([src_pos[0], tgt_pos[0]],
                       [src_pos[1], tgt_pos[1]],
                       'r--', alpha=0.5, linewidth=1, label='备用链路' if link == self.network.links[0] else '')
            else:
                ax.plot([src_pos[0], tgt_pos[0]],
                       [src_pos[1], tgt_pos[1]],
                       'b-', alpha=0.7, linewidth=2, label='主用链路' if link == self.network.links[0] else '')
        
        # 绘制节点
        for node in self.network.nodes.values():
            if node.node_type == "core":
                ax.scatter(node.x, node.y, s=300, c='red', marker='s', 
                          label='核心节点' if node == list(self.network.nodes.values())[0] else '',
                          zorder=5, edgecolors='white', linewidths=2)
            elif node.node_type == "distribution":
                ax.scatter(node.x, node.y, s=200, c='orange', marker='o',
                          label='汇聚节点' if node == list(self.network.nodes.values())[1] else '',
                          zorder=5, edgecolors='white', linewidths=2)
            else:
                ax.scatter(node.x, node.y, s=100, c='green', marker='o',
                          label='接入节点' if node == list(self.network.nodes.values())[-1] else '',
                          zorder=5, edgecolors='white', linewidths=1)
        
        ax.set_xlabel('X 坐标 (km)')
        ax.set_ylabel('Y 坐标 (km)')
        ax.set_title('物理网络拓扑')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_traffic_distribution(self, ax):
        """绘制流量分布"""
        # 按节点类型统计带宽
        type_bandwidth = {"core": 0, "distribution": 0, "access": 0}
        type_count = {"core": 0, "distribution": 0, "access": 0}
        
        for node in self.network.nodes.values():
            type_count[node.node_type] += 1
            type_bandwidth[node.node_type] += node.demand
        
        types = list(type_bandwidth.keys())
        bandwidths = [type_bandwidth[t] for t in types]
        counts = [type_count[t] for t in types]
        avg_bandwidth = [b/c if c > 0 else 0 for b, c in zip(bandwidths, counts)]
        
        x = np.arange(len(types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, bandwidths, width, label='总带宽 (Gbps)', color='steelblue')
        bars2 = ax.bar(x + width/2, avg_bandwidth, width, label='平均带宽/节点 (Gbps)', color='coral')
        
        ax.set_xlabel('节点类型')
        ax.set_ylabel('带宽 (Gbps)')
        ax.set_title('流量分布统计')
        ax.set_xticks(x)
        ax.set_xticklabels(['核心层', '汇聚层', '接入层'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    def _plot_cost_analysis(self, ax):
        """成本分析"""
        # 按电缆类型统计成本
        cable_costs = {}
        cable_lengths = {}
        
        for link in self.network.links:
            cable_name = link.cable.cable_type.value
            if cable_name not in cable_costs:
                cable_costs[cable_name] = 0
                cable_lengths[cable_name] = 0
            cable_costs[cable_name] += link.cost
            cable_lengths[cable_name] += link.length
        
        cables = list(cable_costs.keys())
        costs = [cable_costs[c] for c in cables]
        lengths = [cable_lengths[c] for c in cables]
        
        x = np.arange(len(cables))
        
        # 堆叠柱状图
        ax.bar(x, costs, label='成本 (¥)', color='steelblue')
        
        ax.set_xlabel('电缆类型')
        ax.set_ylabel('成本 (¥)')
        ax.set_title('成本分析')
        ax.set_xticks(x)
        ax.set_xticklabels(cables, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (cost, length) in enumerate(zip(costs, lengths)):
            ax.annotate(f'¥{cost:,.0f}\n{length:.1f}km',
                       xy=(i, cost),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    def _plot_reliability_analysis(self, ax):
        """可靠性分析"""
        metrics = self.network.calculate_metrics()
        
        # 绘制指标
        categories = ['连通性', '冗余度', '平均节点度']
        values = [
            100 if metrics['is_connected'] else 0,
            metrics['redundancy_ratio'] * 100,
            min(metrics['avg_degree'] * 20, 100)  # 归一化
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合雷达图
        angles += angles[:1]
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('网络可靠性分析', pad=20)
        ax.grid(True)


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序 - 电信网络优化示例"""
    print("=" * 70)
    print("电信网络布线优化 - 光纤网络设计")
    print("=" * 70)
    
    # 1. 创建网络
    print("\n[1/4] 生成城市电信网络...")
    network = TelecommunicationsNetwork("示例城市电信网络")
    network.generate_city_network(size="medium")
    print(f"✓ 网络规模：{len(network.nodes)} 个节点")
    print(f"  - 核心节点：{sum(1 for n in network.nodes.values() if n.node_type == 'core')}")
    print(f"  - 汇聚节点：{sum(1 for n in network.nodes.values() if n.node_type == 'distribution')}")
    print(f"  - 接入节点：{sum(1 for n in network.nodes.values() if n.node_type == 'access')}")
    print(f"✓ 流量需求：{len(network.demands)} 条")
    
    # 2. 构建环形拓扑
    print("\n[2/4] 构建环形拓扑 (自愈网络)...")
    links = network.build_ring_topology()
    print(f"✓ 创建链路：{len(links)} 条")
    print(f"  - 主用链路：{sum(1 for l in links if l.is_primary)}")
    print(f"  - 备用链路：{sum(1 for l in links if l.is_backup)}")
    
    # 3. 分配流量
    print("\n[3/4] 分配流量...")
    stats = network.allocate_traffic()
    print(f"✓ 总带宽需求：{stats['total_bandwidth']:.2f} Gbps")
    print(f"✓ 已分配带宽：{stats['allocated_bandwidth']:.2f} Gbps")
    print(f"✓ 阻塞需求：{stats['blocked_demands']} 条")
    print(f"✓ 平均延迟：{stats['avg_latency']:.2f} ms")
    
    # 4. 计算指标
    print("\n[4/4] 计算网络性能指标...")
    metrics = network.calculate_metrics()
    print(f"✓ 总成本：¥{metrics['total_cost']:,.0f}")
    print(f"✓ 总长度：{metrics['total_length_km']:.2f} km")
    print(f"✓ 冗余比：{metrics['redundancy_ratio']:.2f}")
    print(f"✓ 网络连通：{'是' if metrics['is_connected'] else '否'}")
    print(f"✓ 平均节点度：{metrics['avg_degree']:.2f}")
    
    # 5. 可视化
    print("\n[5/5] 生成可视化...")
    visualizer = TelecomVisualizer(network)
    visualizer.plot_network_topology(save_path="outputs/telecom_network_topology.png")
    
    print("\n" + "=" * 70)
    print("✅ 电信网络优化完成!")
    print("=" * 70)
    
    return network, stats, metrics


if __name__ == "__main__":
    network, stats, metrics = main()
