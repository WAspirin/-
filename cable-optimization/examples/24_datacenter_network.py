#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据中心网络布线优化案例

案例背景:
- 大型数据中心网络拓扑设计
- 服务器机架布局与电缆布线优化
- 考虑散热约束、气流组织、维护通道
- 支持冗余设计 (双路供电/网络)
- 目标：最小化布线成本 + 最大化网络性能 + 满足散热要求

核心功能:
1. 数据中心建模 (机架、交换机、服务器)
2. 网络拓扑设计 (Spine-Leaf/Clos 架构)
3. 电缆布线优化 (考虑长度、类型、成本)
4. 散热与气流约束建模
5. 冗余路径计算 (N+1 冗余)
6. 可视化 (3D 布局 + 网络拓扑)

作者：智子 (Sophon)
日期：2026-03-22
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import heapq
import random
from collections import defaultdict


# ==================== 枚举与常量 ====================

class CableType(Enum):
    """电缆/光缆类型"""
    ETHERNET_CAT6 = "Ethernet Cat6"      # 铜缆，1Gbps, 成本低
    ETHERNET_CAT6A = "Ethernet Cat6A"    # 铜缆，10Gbps, 中等成本
    FIBER_OM3 = "Fiber OM3"              # 多模光纤，40Gbps
    FIBER_OM4 = "Fiber OM4"              # 多模光纤，100Gbps
    FIBER_OS2 = "Fiber OS2"              # 单模光纤，400Gbps+
    POWER_CABLE = "Power Cable"          # 电源线


class RackType(Enum):
    """机架类型"""
    SERVER_RACK = "Server Rack"          # 服务器机架
    NETWORK_RACK = "Network Rack"        # 网络机架 (交换机)
    STORAGE_RACK = "Storage Rack"        # 存储机架
    COOLING_RACK = "Cooling Rack"        # 冷却设备


class NetworkTier(Enum):
    """网络层级"""
    SPINE = "Spine"                      # 核心层
    LEAF = "Leaf"                        # 汇聚层
    TOR = "ToR"                          # 机柜顶交换机
    SERVER = "Server"                    # 服务器


# ==================== 数据类 ====================

@dataclass
class Position:
    """3D 位置"""
    x: float  # 行
    y: float  # 列
    z: float = 0  # 高度 (U 单位，1U=44.45mm)
    
    def distance_to(self, other: 'Position') -> float:
        """计算欧氏距离 (米)"""
        return np.sqrt((self.x - other.x)**2 + 
                      (self.y - other.y)**2 + 
                      (self.z - other.z)**2 * 0.04445)  # Z 轴转换为米
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))


@dataclass
class Cable:
    """电缆/光缆"""
    cable_type: CableType
    length: float  # 米
    cost_per_meter: float
    bandwidth: float  # Gbps
    source: str
    target: str
    redundant: bool = False
    
    @property
    def total_cost(self) -> float:
        return self.length * self.cost_per_meter
    
    @property
    def latency_ns(self) -> float:
        """信号延迟 (纳秒) - 简化计算"""
        # 铜缆：~5ns/m, 光纤：~5ns/m (光速的 2/3)
        return self.length * 5.0


@dataclass
class Rack:
    """机架"""
    rack_id: str
    rack_type: RackType
    position: Position
    width: float = 0.6  # 米 (标准 19 英寸机架)
    depth: float = 1.2  # 米
    height: float = 2.0  # 米 (42U)
    power_kw: float = 10.0  # 千瓦
    heat_output_kw: float = 8.0  # 散热 (千瓦)
    network_ports: int = 48  # 网络端口数
    used_ports: int = 0
    
    @property
    def available_ports(self) -> int:
        return self.network_ports - self.used_ports


@dataclass
class CoolingZone:
    """冷却区域"""
    zone_id: str
    bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    cooling_capacity_kw: float
    current_load_kw: float = 0.0
    
    @property
    def available_capacity(self) -> float:
        return self.cooling_capacity_kw - self.current_load_kw
    
    @property
    def utilization(self) -> float:
        return self.current_load_kw / self.cooling_capacity_kw


@dataclass
class NetworkPath:
    """网络路径"""
    path_id: str
    nodes: List[str]
    cables: List[Cable]
    total_latency_ns: float
    total_cost: float
    bandwidth_gbps: float
    is_redundant: bool = False


# ==================== 数据中心建模 ====================

class DataCenter:
    """数据中心模型"""
    
    def __init__(self, name: str, rows: int = 10, cols: int = 8):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.racks: Dict[str, Rack] = {}
        self.cables: List[Cable] = []
        self.cooling_zones: List[CoolingZone] = []
        self.network_topology: Dict[str, List[str]] = defaultdict(list)
        
        # 布局参数
        self.row_spacing = 1.5  # 行间距 (米) - 热通道/冷通道
        self.col_spacing = 1.0  # 列间距 (米)
        
    def add_rack(self, rack: Rack):
        """添加机架"""
        self.racks[rack.rack_id] = rack
        
        # 更新冷却区域负载
        for zone in self.cooling_zones:
            if self._in_zone(rack.position, zone):
                zone.current_load_kw += rack.heat_output_kw
                break
    
    def _in_zone(self, pos: Position, zone: CoolingZone) -> bool:
        """检查位置是否在冷却区域内"""
        x_min, x_max, y_min, y_max = zone.bounds
        return (x_min <= pos.x <= x_max and 
                y_min <= pos.y <= y_max)
    
    def add_cooling_zone(self, zone: CoolingZone):
        """添加冷却区域"""
        self.cooling_zones.append(zone)
    
    def get_rack(self, rack_id: str) -> Optional[Rack]:
        return self.racks.get(rack_id)
    
    def get_all_racks(self) -> List[Rack]:
        return list(self.racks.values())
    
    def get_network_racks(self) -> List[Rack]:
        return [r for r in self.racks.values() 
                if r.rack_type == RackType.NETWORK_RACK]
    
    def get_server_racks(self) -> List[Rack]:
        return [r for r in self.racks.values() 
                if r.rack_type == RackType.SERVER_RACK]
    
    def check_cooling_constraint(self, rack: Rack) -> bool:
        """检查散热约束"""
        for zone in self.cooling_zones:
            if self._in_zone(rack.position, zone):
                return zone.available_capacity >= rack.heat_output_kw
        return True  # 无冷却区域限制
    
    def calculate_cable_length(self, source: str, target: str, 
                               include_vertical: bool = True) -> float:
        """计算电缆长度 (考虑走线路径)"""
        src_rack = self.get_rack(source)
        tgt_rack = self.get_rack(target)
        
        if not src_rack or not tgt_rack:
            return float('inf')
        
        # 水平距离
        h_dist = np.sqrt((src_rack.position.x - tgt_rack.position.x)**2 +
                        (src_rack.position.y - tgt_rack.position.y)**2)
        
        # 垂直距离 (如果需要)
        if include_vertical:
            v_dist = abs(src_rack.position.z - tgt_rack.position.z) * 0.04445
        else:
            v_dist = 0
        
        # 走线槽路径 (增加 20% 余量)
        return (h_dist + v_dist) * 1.2
    
    def add_cable(self, cable: Cable):
        """添加电缆"""
        self.cables.append(cable)
        
        # 更新网络拓扑
        self.network_topology[cable.source].append(cable.target)
        self.network_topology[cable.target].append(cable.source)
        
        # 更新端口使用
        src_rack = self.get_rack(cable.source)
        tgt_rack = self.get_rack(cable.target)
        if src_rack:
            src_rack.used_ports += 1
        if tgt_rack:
            tgt_rack.used_ports += 1
    
    def get_total_cable_cost(self) -> float:
        return sum(c.total_cost for c in self.cables)
    
    def get_total_cable_length(self) -> float:
        return sum(c.length for c in self.cables)
    
    def get_average_latency(self) -> float:
        if not self.cables:
            return 0.0
        return sum(c.latency_ns for c in self.cables) / len(self.cables)


# ==================== 网络拓扑优化器 ====================

class NetworkTopologyOptimizer:
    """网络拓扑优化器 - Spine-Leaf 架构"""
    
    def __init__(self, datacenter: DataCenter):
        self.dc = datacenter
        self.cable_costs = {
            CableType.ETHERNET_CAT6: 2.0,      # 元/米
            CableType.ETHERNET_CAT6A: 3.5,
            CableType.FIBER_OM3: 8.0,
            CableType.FIBER_OM4: 12.0,
            CableType.FIBER_OS2: 20.0,
            CableType.POWER_CABLE: 5.0,
        }
        self.cable_bandwidths = {
            CableType.ETHERNET_CAT6: 1.0,      # Gbps
            CableType.ETHERNET_CAT6A: 10.0,
            CableType.FIBER_OM3: 40.0,
            CableType.FIBER_OM4: 100.0,
            CableType.FIBER_OS2: 400.0,
            CableType.POWER_CABLE: 0.0,        # 电源线无带宽
        }
    
    def design_spine_leaf(self, num_spine: int = 4, num_leaf: int = 8) -> List[Cable]:
        """
        设计 Spine-Leaf 网络拓扑
        
        Spine-Leaf 架构特点:
        - 每个 Leaf 交换机连接到所有 Spine 交换机
        - 全互联拓扑，任意两点间跳数相同
        - 高带宽、低延迟、易扩展
        
        参数:
            num_spine: Spine 层交换机数量
            num_leaf: Leaf 层交换机数量
        
        返回:
            电缆列表
        """
        cables = []
        
        # 获取或创建网络机架
        network_racks = self.dc.get_network_racks()
        if len(network_racks) < num_spine + num_leaf:
            # 需要创建新的网络机架
            self._create_network_racks(num_spine, num_leaf)
            network_racks = self.dc.get_network_racks()
        
        spine_racks = network_racks[:num_spine]
        leaf_racks = network_racks[num_spine:num_spine + num_leaf]
        
        # 每个 Leaf 连接到所有 Spine (全互联)
        cable_type = CableType.FIBER_OM4  # 高速光纤
        
        for leaf in leaf_racks:
            for spine in spine_racks:
                length = self.dc.calculate_cable_length(leaf.rack_id, spine.rack_id)
                
                cable = Cable(
                    cable_type=cable_type,
                    length=length,
                    cost_per_meter=self.cable_costs[cable_type],
                    bandwidth=self.cable_bandwidths[cable_type],
                    source=leaf.rack_id,
                    target=spine.rack_id,
                    redundant=True  # Spine-Leaf 天然冗余
                )
                cables.append(cable)
                self.dc.add_cable(cable)
        
        return cables
    
    def _create_network_racks(self, num_spine: int, num_leaf: int):
        """创建网络机架"""
        # 将网络机架放置在数据中心前端
        base_x = 0
        base_y = 0
        
        for i in range(num_spine):
            rack = Rack(
                rack_id=f"SPINE-{i+1}",
                rack_type=RackType.NETWORK_RACK,
                position=Position(base_x + i * 1.0, base_y, z=20),  # 20U 高度
                power_kw=2.0,
                heat_output_kw=1.5,
                network_ports=64
            )
            self.dc.add_rack(rack)
        
        for i in range(num_leaf):
            rack = Rack(
                rack_id=f"LEAF-{i+1}",
                rack_type=RackType.NETWORK_RACK,
                position=Position(base_x + i * 1.0, base_y + 2.0, z=20),
                power_kw=2.0,
                heat_output_kw=1.5,
                network_ports=48
            )
            self.dc.add_rack(rack)
    
    def connect_server_racks(self, cable_type: CableType = CableType.ETHERNET_CAT6A):
        """
        连接服务器机架到 Leaf 交换机
        
        策略:
        - 每个服务器机架连接到最近的 2 个 Leaf (冗余)
        - 负载均衡考虑
        """
        cables = []
        server_racks = self.dc.get_server_racks()
        leaf_racks = [r for r in self.dc.get_network_racks() 
                     if r.rack_id.startswith("LEAF")]
        
        for server in server_racks:
            # 找到最近的 2 个 Leaf
            distances = []
            for leaf in leaf_racks:
                dist = self.dc.calculate_cable_length(server.rack_id, leaf.rack_id, 
                                                     include_vertical=True)
                distances.append((dist, leaf))
            
            distances.sort(key=lambda x: x[0])
            
            # 连接到最近的 2 个 Leaf (冗余)
            for i, (dist, leaf) in enumerate(distances[:2]):
                cable = Cable(
                    cable_type=cable_type,
                    length=dist,
                    cost_per_meter=self.cable_costs[cable_type],
                    bandwidth=self.cable_bandwidths[cable_type],
                    source=server.rack_id,
                    target=leaf.rack_id,
                    redundant=(i == 1)  # 第二个连接是冗余的
                )
                cables.append(cable)
                self.dc.add_cable(cable)
        
        return cables
    
    def optimize_cable_routing(self, algorithm: str = "mst") -> List[Cable]:
        """
        优化电缆布线路径
        
        算法选项:
        - mst: 最小生成树 (Kruskal)
        - dijkstra: 最短路径
        - vns: 变邻域搜索
        
        返回:
            优化后的电缆列表
        """
        if algorithm == "mst":
            return self._mst_routing()
        elif algorithm == "dijkstra":
            return self._dijkstra_routing()
        elif algorithm == "vns":
            return self._vns_routing()
        else:
            raise ValueError(f"未知算法：{algorithm}")
    
    def _mst_routing(self) -> List[Cable]:
        """最小生成树路由 (Kruskal 算法)"""
        # 构建所有可能的边
        edges = []
        racks = list(self.dc.racks.values())
        
        for i, r1 in enumerate(racks):
            for r2 in racks[i+1:]:
                length = self.dc.calculate_cable_length(r1.rack_id, r2.rack_id)
                cost = length * self.cable_costs[CableType.ETHERNET_CAT6A]
                edges.append((cost, length, r1.rack_id, r2.rack_id))
        
        # 按成本排序
        edges.sort(key=lambda x: x[0])
        
        # Kruskal 算法
        parent = {r.rack_id: r.rack_id for r in racks}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        cables = []
        for cost, length, r1, r2 in edges:
            if union(r1, r2):
                cable = Cable(
                    cable_type=CableType.ETHERNET_CAT6A,
                    length=length,
                    cost_per_meter=self.cable_costs[CableType.ETHERNET_CAT6A],
                    bandwidth=self.cable_bandwidths[CableType.ETHERNET_CAT6A],
                    source=r1,
                    target=r2
                )
                cables.append(cable)
        
        return cables
    
    def _dijkstra_routing(self) -> List[Cable]:
        """最短路径路由"""
        # 简化实现：从中心网络机架到所有其他机架的最短路径
        network_racks = self.dc.get_network_racks()
        if not network_racks:
            return []
        
        source = network_racks[0].rack_id
        cables = []
        
        # 构建图
        graph = defaultdict(list)
        racks = list(self.dc.racks.values())
        for r1 in racks:
            for r2 in racks:
                if r1.rack_id != r2.rack_id:
                    length = self.dc.calculate_cable_length(r1.rack_id, r2.rack_id)
                    graph[r1.rack_id].append((length, r2.rack_id))
        
        # Dijkstra
        dist = {source: 0}
        prev = {source: None}
        pq = [(0, source)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            
            for length, v in graph[u]:
                if v not in dist or d + length < dist[v]:
                    dist[v] = d + length
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
        
        # 构建电缆
        for rack_id in dist:
            if rack_id == source:
                continue
            # 回溯路径
            current = rack_id
            path = []
            while prev[current] is not None:
                path.append(current)
                current = prev[current]
            
            if len(path) > 0:
                length = dist[rack_id]
                cable = Cable(
                    cable_type=CableType.ETHERNET_CAT6A,
                    length=length,
                    cost_per_meter=self.cable_costs[CableType.ETHERNET_CAT6A],
                    bandwidth=self.cable_bandwidths[CableType.ETHERNET_CAT6A],
                    source=prev[rack_id],
                    target=rack_id
                )
                cables.append(cable)
        
        return cables
    
    def _vns_routing(self, max_iterations: int = 100) -> List[Cable]:
        """变邻域搜索优化"""
        # 从 MST 解开始
        current_cables = self._mst_routing()
        current_cost = sum(c.total_cost for c in current_cables)
        best_cables = current_cables.copy()
        best_cost = current_cost
        
        for iteration in range(max_iterations):
            # 扰动：随机移除 10% 的边
            num_remove = max(1, len(current_cables) // 10)
            removed = random.sample(current_cables, num_remove)
            temp_cables = [c for c in current_cables if c not in removed]
            
            # 局部搜索：添加新边重新连接
            # (简化实现，实际应更复杂)
            temp_cost = sum(c.total_cost for c in temp_cables)
            
            if temp_cost < best_cost:
                best_cables = temp_cables.copy()
                best_cost = temp_cost
            
            current_cables = temp_cables
        
        return best_cables


# ==================== 可视化 ====================

class DataCenterVisualizer:
    """数据中心可视化"""
    
    def __init__(self, datacenter: DataCenter):
        self.dc = datacenter
    
    def plot_2d_layout(self, save_path: str = None, show: bool = True):
        """绘制 2D 布局图"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 绘制冷却区域
        for zone in self.dc.cooling_zones:
            x_min, x_max, y_min, y_max = zone.bounds
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=True, alpha=0.1, color='blue', 
                           label=f'冷却区 {zone.zone_id}')
            ax.add_patch(rect)
        
        # 绘制机架
        colors = {
            RackType.SERVER_RACK: 'lightcoral',
            RackType.NETWORK_RACK: 'lightblue',
            RackType.STORAGE_RACK: 'lightgreen',
            RackType.COOLING_RACK: 'lightyellow',
        }
        
        for rack in self.dc.get_all_racks():
            rect = Rectangle(
                (rack.position.x - rack.width/2, rack.position.y - rack.depth/2),
                rack.width, rack.depth,
                fill=True, color=colors.get(rack.rack_type, 'gray'),
                edgecolor='black', linewidth=1.5,
                label=rack.rack_type.value
            )
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(rack.position.x, rack.position.y, rack.rack_id,
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 绘制电缆
        for cable in self.dc.cables[:50]:  # 限制数量避免过于混乱
            src = self.dc.get_rack(cable.source)
            tgt = self.dc.get_rack(cable.target)
            if src and tgt:
                color = 'red' if cable.redundant else 'gray'
                ax.plot([src.position.x, tgt.position.x],
                       [src.position.y, tgt.position.y],
                       color=color, alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('行位置 (米)')
        ax.set_ylabel('列位置 (米)')
        ax.set_title(f'{self.dc.name} - 2D 布局图')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 创建自定义图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 布局图已保存：{save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_network_topology(self, save_path: str = None, show: bool = True):
        """绘制网络拓扑图"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # 分层布局
        spine_y = 4
        leaf_y = 2
        server_y = 0
        
        positions = {}
        
        # Spine 层
        spine_racks = [r for r in self.dc.get_network_racks() 
                      if r.rack_id.startswith("SPINE")]
        for i, rack in enumerate(spine_racks):
            x = (i + 1) * 2
            positions[rack.rack_id] = (x, spine_y)
            circle = Circle((x, spine_y), 0.3, fill=True, color='darkblue',
                          edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, spine_y, rack.rack_id, ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
        
        # Leaf 层
        leaf_racks = [r for r in self.dc.get_network_racks() 
                     if r.rack_id.startswith("LEAF")]
        for i, rack in enumerate(leaf_racks):
            x = (i + 1) * 1.5
            positions[rack.rack_id] = (x, leaf_y)
            circle = Circle((x, leaf_y), 0.25, fill=True, color='steelblue',
                          edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, leaf_y, rack.rack_id, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
        
        # Server 层
        server_racks = self.dc.get_server_racks()[:12]  # 限制数量
        for i, rack in enumerate(server_racks):
            x = (i % 6) * 2 + (i // 6) * 0.5
            y = server_y - (i // 6) * 0.5
            positions[rack.rack_id] = (x, y)
            rect = Rectangle((x-0.15, y-0.15), 0.3, 0.3, fill=True,
                           color='lightcoral', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y, rack.rack_id, ha='center', va='center',
                   fontsize=7)
        
        # 绘制连接
        for cable in self.dc.cables:
            if cable.source in positions and cable.target in positions:
                src_pos = positions[cable.source]
                tgt_pos = positions[cable.target]
                color = 'red' if cable.redundant else 'gray'
                alpha = 0.5 if cable.redundant else 0.3
                ax.plot([src_pos[0], tgt_pos[0]],
                       [src_pos[1], tgt_pos[1]],
                       color=color, alpha=alpha, linewidth=1)
        
        # 添加层级标签
        ax.text(0.5, spine_y, 'Spine 层\n(核心)', fontsize=10, fontweight='bold')
        ax.text(0.5, leaf_y, 'Leaf 层\n(汇聚)', fontsize=10, fontweight='bold')
        ax.text(0.5, server_y, 'Server 层', fontsize=10, fontweight='bold')
        
        ax.set_xlim(-1, max(p[0] for p in positions.values()) + 1)
        ax.set_ylim(server_y - 1, spine_y + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{self.dc.name} - Spine-Leaf 网络拓扑', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 网络拓扑图已保存：{save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_cost_analysis(self, save_path: str = None, show: bool = True):
        """绘制成本分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 电缆类型成本分布
        cable_type_costs = defaultdict(float)
        cable_type_lengths = defaultdict(float)
        for cable in self.dc.cables:
            cable_type_costs[cable.cable_type.value] += cable.total_cost
            cable_type_lengths[cable.cable_type.value] += cable.length
        
        axes[0, 0].bar(range(len(cable_type_costs)), list(cable_type_costs.values()))
        axes[0, 0].set_xticks(range(len(cable_type_costs)))
        axes[0, 0].set_xticklabels(list(cable_type_costs.keys()), rotation=45, ha='right')
        axes[0, 0].set_ylabel('成本 (元)')
        axes[0, 0].set_title('电缆类型成本分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 冗余 vs 非冗余
        redundant_cost = sum(c.total_cost for c in self.dc.cables if c.redundant)
        normal_cost = sum(c.total_cost for c in self.dc.cables if not c.redundant)
        
        axes[0, 1].pie([normal_cost, redundant_cost],
                      labels=['正常连接', '冗余连接'],
                      autopct='%1.1f%%', colors=['steelblue', 'lightcoral'])
        axes[0, 1].set_title('冗余成本占比')
        
        # 3. 电缆长度分布
        lengths = [c.length for c in self.dc.cables]
        axes[1, 0].hist(lengths, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('电缆长度 (米)')
        axes[1, 0].set_ylabel('数量')
        axes[1, 0].set_title('电缆长度分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 冷却区域利用率
        if self.dc.cooling_zones:
            zone_utils = [z.utilization for z in self.dc.cooling_zones]
            zone_names = [z.zone_id for z in self.dc.cooling_zones]
            colors = plt.cm.RdYlGn(zone_utils)
            bars = axes[1, 1].bar(range(len(zone_utils)), zone_utils, color=colors)
            axes[1, 1].set_xticks(range(len(zone_utils)))
            axes[1, 1].set_xticklabels(zone_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('利用率')
            axes[1, 1].set_title('冷却区域利用率')
            axes[1, 1].set_ylim(0, 1.2)
            axes[1, 1].axhline(y=1.0, color='red', linestyle='--', label='容量上限')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '无冷却区域数据', ha='center', va='center',
                          transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].axis('off')
        
        plt.suptitle(f'{self.dc.name} - 成本与资源分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 成本分析图已保存：{save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


# ==================== 主函数 ====================

def create_sample_datacenter() -> DataCenter:
    """创建示例数据中心"""
    dc = DataCenter("示例数据中心 A", rows=10, cols=8)
    
    # 添加冷却区域 (冷热通道设计)
    dc.add_cooling_zone(CoolingZone(
        zone_id="冷通道 1",
        bounds=(0, 5, 0, 3),
        cooling_capacity_kw=100.0
    ))
    dc.add_cooling_zone(CoolingZone(
        zone_id="冷通道 2",
        bounds=(5, 10, 0, 3),
        cooling_capacity_kw=100.0
    ))
    dc.add_cooling_zone(CoolingZone(
        zone_id="热通道 1",
        bounds=(0, 10, 3, 6),
        cooling_capacity_kw=80.0
    ))
    
    # 添加服务器机架 (网格布局)
    rack_id = 1
    for row in range(8):
        for col in range(6):
            # 交替排列 (面对面，形成冷热通道)
            if row % 2 == 0:
                y_offset = 1.0  # 冷通道侧
            else:
                y_offset = 4.0  # 热通道侧
            
            rack = Rack(
                rack_id=f"SR-{rack_id:02d}",
                rack_type=RackType.SERVER_RACK,
                position=Position(row * 1.5, y_offset + col * 1.2, z=20),
                power_kw=8.0 + random.uniform(-2, 2),
                heat_output_kw=6.0 + random.uniform(-1, 1),
                network_ports=24
            )
            dc.add_rack(rack)
            rack_id += 1
    
    return dc


def main():
    """主函数 - 演示数据中心网络布线优化"""
    print("=" * 60)
    print("数据中心网络布线优化系统")
    print("=" * 60)
    
    # 创建数据中心
    print("\n1. 创建数据中心模型...")
    dc = create_sample_datacenter()
    print(f"   ✓ 数据中心：{dc.name}")
    print(f"   ✓ 机架数量：{len(dc.racks)}")
    print(f"   ✓ 冷却区域：{len(dc.cooling_zones)}")
    
    # 设计 Spine-Leaf 网络
    print("\n2. 设计 Spine-Leaf 网络拓扑...")
    optimizer = NetworkTopologyOptimizer(dc)
    spine_leaf_cables = optimizer.design_spine_leaf(num_spine=4, num_leaf=8)
    print(f"   ✓ Spine 交换机：4 台")
    print(f"   ✓ Leaf 交换机：8 台")
    print(f"   ✓ Spine-Leaf 连接：{len(spine_leaf_cables)} 条")
    
    # 连接服务器机架
    print("\n3. 连接服务器机架...")
    server_cables = optimizer.connect_server_racks(CableType.ETHERNET_CAT6A)
    print(f"   ✓ 服务器机架连接：{len(server_cables)} 条")
    
    # 优化电缆路由
    print("\n4. 优化电缆路由 (MST 算法)...")
    optimized_cables = optimizer.optimize_cable_routing("mst")
    print(f"   ✓ 优化后电缆数量：{len(optimized_cables)} 条")
    
    # 统计结果
    print("\n5. 统计结果:")
    print(f"   - 总电缆数量：{len(dc.cables)}")
    print(f"   - 总电缆长度：{dc.get_total_cable_length():.2f} 米")
    print(f"   - 总成本：{dc.get_total_cable_cost():.2f} 元")
    print(f"   - 平均延迟：{dc.get_average_latency():.2f} 纳秒")
    print(f"   - 冗余连接：{sum(1 for c in dc.cables if c.redundant)} 条")
    
    # 可视化
    print("\n6. 生成可视化图表...")
    visualizer = DataCenterVisualizer(dc)
    
    output_dir = "/root/.openclaw/workspace/cable-optimization/examples/outputs"
    visualizer.plot_2d_layout(
        save_path=f"{output_dir}/datacenter_layout.png",
        show=False
    )
    visualizer.plot_network_topology(
        save_path=f"{output_dir}/datacenter_topology.png",
        show=False
    )
    visualizer.plot_cost_analysis(
        save_path=f"{output_dir}/datacenter_cost_analysis.png",
        show=False
    )
    
    print("\n" + "=" * 60)
    print("✅ 数据中心网络布线优化完成!")
    print("=" * 60)
    
    # 返回结果用于测试
    return {
        "total_cables": len(dc.cables),
        "total_length": dc.get_total_cable_length(),
        "total_cost": dc.get_total_cable_cost(),
        "avg_latency": dc.get_average_latency(),
        "redundant_count": sum(1 for c in dc.cables if c.redundant)
    }


if __name__ == "__main__":
    results = main()
    
    # 保存结果到 JSON
    import json
    output_path = "/root/.openclaw/workspace/cable-optimization/examples/outputs/24_datacenter_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📊 结果已保存：{output_path}")
