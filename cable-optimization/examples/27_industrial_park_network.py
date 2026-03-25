#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业园区网络布线优化案例

场景描述:
- 大型工业园区包含多个功能区域（生产区、仓储区、办公区、动力区）
- 需要设计电力、通信、给排水等多管网系统
- 考虑地下管廊约束、交叉避让、维护通道等实际工程因素
- 多目标优化：成本最小化 + 可靠性最大化 + 施工难度最小化

作者：智子 (Sophon)
日期：2026-03-25
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.collections import PatchCollection
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import heapq
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 数据结构定义
# ============================================================================

class ZoneType(Enum):
    """区域类型"""
    PRODUCTION = "生产区"
    WAREHOUSE = "仓储区"
    OFFICE = "办公区"
    POWER = "动力区"
    GREEN = "绿化带"
    ROAD = "道路"


class UtilityType(Enum):
    """管网类型"""
    ELECTRICITY = "电力"
    COMMUNICATION = "通信"
    WATER_SUPPLY = "给水"
    DRAINAGE = "排水"
    GAS = "燃气"


@dataclass
class Zone:
    """功能区域"""
    id: int
    name: str
    zone_type: ZoneType
    center: Tuple[float, float]
    width: float
    height: float
    power_demand: float = 0.0  # kW
    data_demand: float = 0.0  # Mbps
    water_demand: float = 0.0  # m³/h
    
    def contains_point(self, x: float, y: float) -> bool:
        """检查点是否在区域内"""
        cx, cy = self.center
        return (abs(x - cx) <= self.width / 2 and 
                abs(y - cy) <= self.height / 2)


@dataclass
class Node:
    """管网节点"""
    id: int
    x: float
    y: float
    node_type: str  # 'source', 'sink', 'junction'
    zone_id: Optional[int] = None
    utility_types: List[UtilityType] = field(default_factory=list)


@dataclass
class Edge:
    """管网边"""
    id: int
    start_node: int
    end_node: int
    utility_type: UtilityType
    length: float
    cost: float
    capacity: float
    installed: bool = False


@dataclass
class Obstacle:
    """障碍物（地下设施、建筑基础等）"""
    id: int
    obstacle_type: str  # 'building', 'underground', 'protected'
    vertices: List[Tuple[float, float]]
    
    def contains_segment(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """检查线段是否与障碍物相交"""
        # 简化：检查线段中点是否在障碍物内
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        return self._point_in_polygon(mid)
    
    def _point_in_polygon(self, point: Tuple[float, float]) -> bool:
        """射线法判断点是否在多边形内"""
        x, y = point
        inside = False
        n = len(self.vertices)
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside


@dataclass
class UtilityCorridor:
    """综合管廊"""
    id: int
    start: Tuple[float, float]
    end: Tuple[float, float]
    width: float
    max_utilities: int  # 最大容纳管网数
    current_utilities: int = 0
    cost_per_meter: float = 500.0  # 管廊单位成本
    
    def can_add_utility(self) -> bool:
        return self.current_utilities < self.max_utilities
    
    def add_utility(self):
        self.current_utilities += 1


# ============================================================================
# 工业园区建模
# ============================================================================

class IndustrialPark:
    """工业园区模型"""
    
    def __init__(self, width: float = 1000, height: float = 800):
        self.width = width
        self.height = height
        self.zones: List[Zone] = []
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.obstacles: List[Obstacle] = []
        self.corridors: List[UtilityCorridor] = []
        self.edge_id_counter = 0
        self.node_id_counter = 0
        
    def add_zone(self, zone: Zone):
        """添加功能区域"""
        self.zones.append(zone)
        
    def add_obstacle(self, obstacle: Obstacle):
        """添加障碍物"""
        self.obstacles.append(obstacle)
        
    def add_corridor(self, corridor: UtilityCorridor):
        """添加综合管廊"""
        self.corridors.append(corridor)
        
    def create_default_layout(self):
        """创建默认园区布局"""
        # 生产区 (3 个)
        self.add_zone(Zone(0, "生产区 A", ZoneType.PRODUCTION, (200, 400), 150, 200, 
                          power_demand=500, data_demand=100, water_demand=50))
        self.add_zone(Zone(1, "生产区 B", ZoneType.PRODUCTION, (500, 400), 150, 200,
                          power_demand=600, data_demand=120, water_demand=60))
        self.add_zone(Zone(2, "生产区 C", ZoneType.PRODUCTION, (800, 400), 150, 200,
                          power_demand=450, data_demand=90, water_demand=45))
        
        # 仓储区 (2 个)
        self.add_zone(Zone(3, "仓储区 A", ZoneType.WAREHOUSE, (200, 200), 120, 150,
                          power_demand=200, data_demand=50, water_demand=20))
        self.add_zone(Zone(4, "仓储区 B", ZoneType.WAREHOUSE, (800, 200), 120, 150,
                          power_demand=180, data_demand=45, water_demand=18))
        
        # 办公区
        self.add_zone(Zone(5, "办公区", ZoneType.OFFICE, (500, 650), 200, 120,
                          power_demand=300, data_demand=500, water_demand=30))
        
        # 动力区 (变电站、水泵房等)
        self.add_zone(Zone(6, "变电站", ZoneType.POWER, (100, 500), 80, 100,
                          power_demand=-2000))  # 负值表示供电
        self.add_zone(Zone(7, "水泵房", ZoneType.POWER, (900, 500), 60, 80,
                          water_demand=-200))
        self.add_zone(Zone(8, "数据中心", ZoneType.POWER, (500, 100), 100, 80,
                          power_demand=400, data_demand=-1000))  # 负值表示数据汇聚
        
        # 添加道路障碍物
        self.add_obstacle(Obstacle(0, "road", 
                                   [(0, 350), (1000, 350), (1000, 380), (0, 380)]))
        self.add_obstacle(Obstacle(1, "road",
                                   [(0, 550), (1000, 550), (1000, 580), (0, 580)]))
        
        # 添加综合管廊 (主干道)
        self.add_corridor(UtilityCorridor(0, (100, 400), (900, 400), 3, 5, cost_per_meter=500))
        self.add_corridor(UtilityCorridor(1, (500, 150), (500, 650), 3, 4, cost_per_meter=500))
        
    def create_nodes(self):
        """为每个区域创建节点"""
        for zone in self.zones:
            cx, cy = zone.center
            # 区域中心节点
            node = Node(self.node_id_counter, cx, cy, 'junction', zone.id)
            
            # 根据区域需求添加管网类型
            if zone.power_demand != 0:
                node.utility_types.append(UtilityType.ELECTRICITY)
            if zone.data_demand != 0:
                node.utility_types.append(UtilityType.COMMUNICATION)
            if zone.water_demand != 0:
                node.utility_types.append(UtilityType.WATER_SUPPLY)
                if zone.water_demand < 0:
                    node.utility_types.append(UtilityType.DRAINAGE)
            
            self.nodes.append(node)
            self.node_id_counter += 1
            
    def calculate_edge_cost(self, node1: Node, node2: Node, 
                           utility_type: UtilityType) -> Tuple[float, float, bool]:
        """计算边的成本（考虑管廊优惠和障碍物避让）"""
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        length = np.sqrt(dx**2 + dy**2)
        
        # 检查是否可以使用管廊
        corridor_discount = 1.0
        use_corridor = False
        
        for corridor in self.corridors:
            if self._segment_near_corridor((node1.x, node1.y), (node2.x, node2.y), corridor):
                if corridor.can_add_utility():
                    corridor_discount = 0.6  # 管廊内成本降低 40%
                    use_corridor = True
                    break
        
        # 检查是否与障碍物相交
        obstacle_penalty = 1.0
        for obstacle in self.obstacles:
            if obstacle.contains_segment((node1.x, node1.y), (node2.x, node2.y)):
                obstacle_penalty = 2.0  # 穿越障碍物成本翻倍
                break
        
        # 基础成本（不同管网类型不同）
        base_cost_per_meter = {
            UtilityType.ELECTRICITY: 150,
            UtilityType.COMMUNICATION: 100,
            UtilityType.WATER_SUPPLY: 200,
            UtilityType.DRAINAGE: 250,
            UtilityType.GAS: 300
        }
        
        base_cost = base_cost_per_meter.get(utility_type, 200)
        cost = length * base_cost * corridor_discount * obstacle_penalty
        
        return cost, length, use_corridor
    
    def _segment_near_corridor(self, p1: Tuple[float, float], 
                               p2: Tuple[float, float], 
                               corridor: UtilityCorridor) -> bool:
        """检查线段是否靠近管廊"""
        # 简化：检查线段中点到管廊线段的距离
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        # 管廊线段
        cx1, cy1 = corridor.start
        cx2, cy2 = corridor.end
        
        # 点到线段距离
        dx = cx2 - cx1
        dy = cy2 - cy1
        if dx == 0 and dy == 0:
            dist = np.sqrt((mid[0] - cx1)**2 + (mid[1] - cy1)**2)
        else:
            t = max(0, min(1, ((mid[0] - cx1) * dx + (mid[1] - cy1) * dy) / (dx**2 + dy**2)))
            proj = (cx1 + t * dx, cy1 + t * dy)
            dist = np.sqrt((mid[0] - proj[0])**2 + (mid[1] - proj[1])**2)
        
        return dist < 30  # 30 米内可使用管廊
    
    def build_complete_graph(self) -> Dict[UtilityType, List[Tuple[int, int, float]]]:
        """为每种管网类型构建完全图"""
        graphs = {ut: [] for ut in UtilityType}
        
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i >= j:
                    continue
                
                # 检查两个节点是否都需要该类型管网
                for ut in node1.utility_types:
                    if ut in node2.utility_types:
                        cost, length, _ = self.calculate_edge_cost(node1, node2, ut)
                        graphs[ut].append((i, j, cost))
        
        return graphs


# ============================================================================
# 优化算法
# ============================================================================

class MultiUtilityOptimizer:
    """多管网联合优化器"""
    
    def __init__(self, park: IndustrialPark):
        self.park = park
        self.results: Dict[UtilityType, Dict] = {}
        
    def optimize_mst(self, utility_type: UtilityType) -> List[Tuple[int, int]]:
        """使用 Kruskal 算法构建 MST"""
        graph = self.park.build_complete_graph()[utility_type]
        
        # 过滤出该类型的节点
        valid_nodes = set()
        for node in self.park.nodes:
            if utility_type in node.utility_types:
                valid_nodes.add(node.id)
        
        # 只保留有效节点间的边
        filtered_graph = [(i, j, c) for i, j, c in graph if i in valid_nodes and j in valid_nodes]
        filtered_graph.sort(key=lambda x: x[2])
        
        # Union-Find
        parent = {node_id: node_id for node_id in valid_nodes}
        rank = {node_id: 0 for node_id in valid_nodes}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        mst_edges = []
        for i, j, cost in filtered_graph:
            if union(i, j):
                mst_edges.append((i, j))
                if len(mst_edges) == len(valid_nodes) - 1:
                    break
        
        return mst_edges
    
    def optimize_all_utilities(self):
        """优化所有管网类型"""
        for ut in UtilityType:
            mst_edges = self.optimize_mst(ut)
            
            # 计算总成本
            total_cost = 0
            total_length = 0
            corridor_usage = 0
            
            for i, j in mst_edges:
                node1, node2 = self.park.nodes[i], self.park.nodes[j]
                cost, length, use_corridor = self.park.calculate_edge_cost(node1, node2, ut)
                total_cost += cost
                total_length += length
                if use_corridor:
                    corridor_usage += 1
                    
                    # 更新管廊使用计数
                    for corridor in self.park.corridors:
                        if self.park._segment_near_corridor(
                            (node1.x, node1.y), (node2.x, node2.y), corridor):
                            corridor.add_utility()
                            break
            
            self.results[ut] = {
                'edges': mst_edges,
                'total_cost': total_cost,
                'total_length': total_length,
                'corridor_usage': corridor_usage,
                'edge_count': len(mst_edges)
            }
            
        return self.results
    
    def calculate_system_reliability(self) -> float:
        """计算系统可靠性（基于冗余度）"""
        total_edges = sum(r['edge_count'] for r in self.results.values())
        corridor_edges = sum(r['corridor_usage'] for r in self.results.values())
        
        # 管廊内的管线更可靠（受环境影响小）
        reliability = 0.85 + 0.15 * (corridor_edges / max(total_edges, 1))
        return reliability
    
    def calculate_construction_difficulty(self) -> float:
        """计算施工难度（基于障碍物穿越）"""
        difficulty = 0
        for ut, result in self.results.items():
            for i, j in result['edges']:
                node1, node2 = self.park.nodes[i], self.park.nodes[j]
                for obstacle in self.park.obstacles:
                    if obstacle.contains_segment((node1.x, node1.y), (node2.x, node2.y)):
                        difficulty += 1
        return difficulty


# ============================================================================
# 可视化
# ============================================================================

class IndustrialParkVisualizer:
    """工业园区可视化"""
    
    def __init__(self, park: IndustrialPark, optimizer: MultiUtilityOptimizer):
        self.park = park
        self.optimizer = optimizer
        
    def plot_park_layout(self, save_path: Optional[str] = None):
        """绘制园区布局"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('工业园区多管网布线优化结果', fontsize=16, fontweight='bold')
        
        # 子图 1: 园区功能分区
        ax1 = axes[0, 0]
        self._plot_zones(ax1)
        ax1.set_title('园区功能分区', fontsize=12)
        
        # 子图 2: 电力管网
        ax2 = axes[0, 1]
        self._plot_utility(ax2, UtilityType.ELECTRICITY, '电力管网')
        
        # 子图 3: 通信管网
        ax3 = axes[1, 0]
        self._plot_utility(ax3, UtilityType.COMMUNICATION, '通信管网')
        
        # 子图 4: 给水管网
        ax4 = axes[1, 1]
        self._plot_utility(ax4, UtilityType.WATER_SUPPLY, '给水管网')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"布局图已保存：{save_path}")
        plt.show()
    
    def _plot_zones(self, ax):
        """绘制功能区域"""
        colors = {
            ZoneType.PRODUCTION: '#FF6B6B',
            ZoneType.WAREHOUSE: '#4ECDC4',
            ZoneType.OFFICE: '#45B7D1',
            ZoneType.POWER: '#FFA07A',
            ZoneType.GREEN: '#98D8AA',
            ZoneType.ROAD: '#95A5A6'
        }
        
        for zone in self.park.zones:
            rect = Rectangle(
                (zone.center[0] - zone.width/2, zone.center[1] - zone.height/2),
                zone.width, zone.height,
                fill=True, alpha=0.6, color=colors.get(zone.zone_type, '#CCCCCC'),
                label=f"{zone.name} ({zone.zone_type.value})"
            )
            ax.add_patch(rect)
            ax.text(zone.center[0], zone.center[1], zone.name, 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 绘制障碍物
        for obstacle in self.park.obstacles:
            poly = Polygon(obstacle.vertices, fill=True, color='#7F8C8D', alpha=0.5)
            ax.add_patch(poly)
        
        # 绘制管廊
        for corridor in self.park.corridors:
            dx = corridor.end[0] - corridor.start[0]
            dy = corridor.end[1] - corridor.start[1]
            length = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx) * 180 / np.pi
            
            rect = Rectangle(
                corridor.start, length, corridor.width,
                angle=angle, fill=True, color='#F39C12', alpha=0.4,
                label='综合管廊'
            )
            ax.add_patch(rect)
        
        ax.set_xlim(0, self.park.width)
        ax.set_ylim(0, self.park.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    def _plot_utility(self, ax, utility_type: UtilityType, title: str):
        """绘制特定管网"""
        self._plot_zones(ax)
        
        if utility_type in self.optimizer.results:
            result = self.optimizer.results[utility_type]
            
            for i, j in result['edges']:
                node1, node2 = self.park.nodes[i], self.park.nodes[j]
                
                # 检查是否使用管廊
                use_corridor = False
                for corridor in self.park.corridors:
                    if self.park._segment_near_corridor(
                        (node1.x, node1.y), (node2.x, node2.y), corridor):
                        use_corridor = True
                        break
                
                color = '#E74C3C' if use_corridor else '#3498DB'
                linewidth = 2.5 if use_corridor else 1.5
                linestyle = '-' if use_corridor else '--'
                
                ax.plot([node1.x, node2.x], [node1.y, node2.y], 
                       color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.7)
            
            # 绘制节点
            for node in self.park.nodes:
                if utility_type in node.utility_types:
                    circle = Circle((node.x, node.y), 8, color='#2C3E50', zorder=5)
                    ax.add_patch(circle)
        
        ax.set_title(f'{title}\n成本：¥{result["total_cost"]:,.0f} | 长度：{result["total_length"]:.0f}m | 管廊使用：{result["corridor_usage"]}条',
                    fontsize=11)
        ax.set_xlim(0, self.park.width)
        ax.set_ylim(0, self.park.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def plot_cost_analysis(self, save_path: Optional[str] = None):
        """绘制成本分析"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('管网成本分析', fontsize=14, fontweight='bold')
        
        utilities = list(self.optimizer.results.keys())
        costs = [self.optimizer.results[ut]['total_cost'] for ut in utilities]
        lengths = [self.optimizer.results[ut]['total_length'] for ut in utilities]
        corridor_usage = [self.optimizer.results[ut]['corridor_usage'] for ut in utilities]
        
        # 子图 1: 成本对比
        ax1 = axes[0]
        bars1 = ax1.barh([ut.value for ut in utilities], costs, color='#3498DB')
        ax1.set_xlabel('成本 (元)')
        ax1.set_title('各管网总成本对比')
        for bar, cost in zip(bars1, costs):
            ax1.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
                    f'¥{cost:,.0f}', va='center', fontsize=9)
        
        # 子图 2: 长度对比
        ax2 = axes[1]
        bars2 = ax2.barh([ut.value for ut in utilities], lengths, color='#2ECC71')
        ax2.set_xlabel('长度 (米)')
        ax2.set_title('各管网总长度对比')
        for bar, length in zip(bars2, lengths):
            ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                    f'{length:.0f}m', va='center', fontsize=9)
        
        # 子图 3: 管廊使用率
        ax3 = axes[2]
        edges_count = [self.optimizer.results[ut]['edge_count'] for ut in utilities]
        usage_rates = [c / max(e, 1) * 100 for c, e in zip(corridor_usage, edges_count)]
        bars3 = ax3.barh([ut.value for ut in utilities], usage_rates, color='#F39C12')
        ax3.set_xlabel('管廊使用率 (%)')
        ax3.set_title('管廊使用率')
        for bar, rate in zip(bars3, usage_rates):
            ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{rate:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"成本分析图已保存：{save_path}")
        plt.show()
    
    def print_summary(self):
        """打印优化结果摘要"""
        print("\n" + "="*70)
        print("工业园区多管网布线优化结果摘要")
        print("="*70)
        
        total_cost = 0
        total_length = 0
        total_corridor_usage = 0
        
        print(f"\n{'管网类型':<12} {'边数':>6} {'长度 (m)':>12} {'成本 (元)':>15} {'管廊使用':>10}")
        print("-"*60)
        
        for ut, result in self.optimizer.results.items():
            print(f"{ut.value:<12} {result['edge_count']:>6} {result['total_length']:>12.1f} "
                  f"¥{result['total_cost']:>14,.0f} {result['corridor_usage']:>10}")
            total_cost += result['total_cost']
            total_length += result['total_length']
            total_corridor_usage += result['corridor_usage']
        
        print("-"*60)
        print(f"{'总计':<12} {'':>6} {total_length:>12.1f} ¥{total_cost:>14,.0f} {total_corridor_usage:>10}")
        
        reliability = self.optimizer.calculate_system_reliability()
        difficulty = self.optimizer.calculate_construction_difficulty()
        
        print(f"\n系统可靠性：{reliability:.2%}")
        print(f"施工难度指数：{difficulty} (穿越障碍物次数)")
        print("="*70 + "\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*70)
    print("工业园区多管网布线优化系统")
    print("="*70)
    
    # 创建园区模型
    park = IndustrialPark()
    park.create_default_layout()
    park.create_nodes()
    
    print(f"\n园区规模：{park.width}m × {park.height}m")
    print(f"功能区域：{len(park.zones)} 个")
    print(f"管网节点：{len(park.nodes)} 个")
    print(f"综合管廊：{len(park.corridors)} 条")
    print(f"障碍物：{len(park.obstacles)} 个")
    
    # 创建优化器
    optimizer = MultiUtilityOptimizer(park)
    
    print("\n正在优化管网布局...")
    results = optimizer.optimize_all_utilities()
    print(f"优化完成！共优化 {len(results)} 种管网类型")
    
    # 可视化
    visualizer = IndustrialParkVisualizer(park, optimizer)
    
    # 保存图表
    import os
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer.plot_park_layout(save_path=os.path.join(output_dir, 'industrial_park_layout.png'))
    visualizer.plot_cost_analysis(save_path=os.path.join(output_dir, 'industrial_park_cost_analysis.png'))
    
    # 打印摘要
    visualizer.print_summary()
    
    print("优化完成！图表已保存到 outputs/ 目录")
    print("="*70)


if __name__ == "__main__":
    main()
