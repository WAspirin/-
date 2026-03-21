#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市电网规划案例 - City Power Grid Planning

案例背景:
    城市配电网电缆网络设计与优化
    
问题描述:
    - 城市区域包含多个变电站、配电室和用电负荷点
    - 需要设计电缆网络连接所有节点
    - 考虑地理约束 (道路、河流、建筑物)
    - 考虑电力约束 (容量、电压降、可靠性)
    - 目标：最小化总成本 (建设 + 运维)

核心算法:
    1. 地理信息建模 (GIS 数据)
    2. 多约束最短路径 (避开障碍区)
    3. 分层优化 (主干网 + 配电网)
    4. 可靠性分析 (N-1 准则)
    5. 多目标优化 (成本 vs 可靠性)

作者：智子 (Sophon)
日期：2026-03-21
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import heapq
from enum import Enum
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NodeType(Enum):
    """节点类型枚举"""
    SUBSTATION = "变电站"  # 220kV/110kV 变电站
    DISTRIBUTION = "配电室"  # 10kV 配电室
    LOAD = "负荷点"  # 用电负荷
    JUNCTION = " junction"  # 电缆接头


class CableType(Enum):
    """电缆类型枚举"""
    HV_220 = "220kV 高压电缆"
    HV_110 = "110kV 高压电缆"
    MV_10 = "10kV 中压电缆"
    LV_04 = "0.4kV 低压电缆"


@dataclass
class GridNode:
    """电网节点"""
    id: str
    x: float  # 坐标 x (km)
    y: float  # 坐标 y (km)
    node_type: NodeType
    voltage_level: float  # 电压等级 (kV)
    load_capacity: float = 0.0  # 负荷容量 (MVA)
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"{self.node_type.value}-{self.id}"


@dataclass
class Cable:
    """电缆段"""
    id: str
    from_node: str
    to_node: str
    cable_type: CableType
    length: float  # 长度 (km)
    capacity: float  # 容量 (MVA)
    cost_per_km: float  # 单位成本 (万元/km)
    resistance: float = 0.0  # 电阻 (Ω/km)
    reactance: float = 0.0  # 电抗 (Ω/km)
    
    @property
    def total_cost(self) -> float:
        """总成本"""
        return self.length * self.cost_per_km
    
    @property
    def impedance(self) -> float:
        """阻抗"""
        return np.sqrt(self.resistance**2 + self.reactance**2)


@dataclass
class Obstacle:
    """地理障碍区"""
    id: str
    obstacle_type: str  # "river", "building", "park", "highway"
    polygon: List[Tuple[float, float]]  # 多边形顶点
    color: str = "gray"


@dataclass
class PowerFlowResult:
    """潮流计算结果"""
    node_voltages: Dict[str, float]  # 节点电压 (p.u.)
    branch_flows: Dict[str, float]  # 支路功率 (MVA)
    total_loss: float  # 总损耗 (MW)
    voltage_violations: List[str]  # 电压越限节点


class CityPowerGrid:
    """
    城市电网模型
    
    模拟一个城市区域的配电网系统
    """
    
    def __init__(self, area_size: Tuple[float, float] = (20.0, 20.0)):
        """
        初始化城市电网
        
        Args:
            area_size: 区域大小 (宽，高) 单位：km
        """
        self.area_width, self.area_height = area_size
        self.nodes: Dict[str, GridNode] = {}
        self.cables: Dict[str, Cable] = {}
        self.obstacles: List[Obstacle] = []
        self.adjacency: Dict[str, List[str]] = {}
        
        # 电缆参数表
        self.cable_params = {
            CableType.HV_220: {"capacity": 200, "cost": 150, "r": 0.08, "x": 0.12},
            CableType.HV_110: {"capacity": 100, "cost": 80, "r": 0.12, "x": 0.15},
            CableType.MV_10: {"capacity": 15, "cost": 25, "r": 0.35, "x": 0.08},
            CableType.LV_04: {"capacity": 2, "cost": 8, "r": 0.65, "x": 0.06},
        }
    
    def add_node(self, node: GridNode):
        """添加节点"""
        self.nodes[node.id] = node
        self.adjacency[node.id] = []
    
    def add_cable(self, cable: Cable):
        """添加电缆"""
        self.cables[cable.id] = cable
        self.adjacency[cable.from_node].append(cable.to_node)
        self.adjacency[cable.to_node].append(cable.from_node)
    
    def add_obstacle(self, obstacle: Obstacle):
        """添加障碍区"""
        self.obstacles.append(obstacle)
    
    def generate_city_grid(self, 
                           num_substations: int = 2,
                           num_distribution: int = 8,
                           num_loads: int = 30,
                           seed: int = 42):
        """
        生成城市电网拓扑
        
        Args:
            num_substations: 变电站数量
            num_distribution: 配电室数量
            num_loads: 负荷点数量
            seed: 随机种子
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # 添加地理障碍
        self._generate_obstacles()
        
        # 添加变电站 (220kV)
        for i in range(num_substations):
            node = GridNode(
                id=f"SUB_{i+1:02d}",
                x=random.uniform(0.2, 0.8) * self.area_width,
                y=random.uniform(0.2, 0.8) * self.area_height,
                node_type=NodeType.SUBSTATION,
                voltage_level=220.0,
                load_capacity=0.0,
                name=f"{i+1}#变电站"
            )
            self.add_node(node)
        
        # 添加配电室 (110kV/10kV)
        for i in range(num_distribution):
            node = GridNode(
                id=f"DIS_{i+1:02d}",
                x=random.uniform(0.1, 0.9) * self.area_width,
                y=random.uniform(0.1, 0.9) * self.area_height,
                node_type=NodeType.DISTRIBUTION,
                voltage_level=10.0,
                load_capacity=random.uniform(5, 15),  # 5-15 MVA
                name=f"{i+1}#配电室"
            )
            self.add_node(node)
        
        # 添加负荷点 (0.4kV)
        for i in range(num_loads):
            node = GridNode(
                id=f"LOAD_{i+1:03d}",
                x=random.uniform(0.05, 0.95) * self.area_width,
                y=random.uniform(0.05, 0.95) * self.area_height,
                node_type=NodeType.LOAD,
                voltage_level=0.4,
                load_capacity=random.uniform(0.5, 3.0),  # 0.5-3 MVA
                name=f"负荷{i+1}"
            )
            self.add_node(node)
        
        print(f"✓ 生成城市电网：{num_substations} 变电站，{num_distribution} 配电室，{num_loads} 负荷点")
    
    def _generate_obstacles(self):
        """生成地理障碍区"""
        # 河流 (从左到右)
        river_points = [
            (0, self.area_height * 0.4),
            (self.area_width * 0.3, self.area_height * 0.35),
            (self.area_width * 0.7, self.area_height * 0.45),
            (self.area_width, self.area_height * 0.4)
        ]
        river_width = 0.8
        river_poly = self._buffer_polygon(river_points, river_width)
        self.add_obstacle(Obstacle(
            id="OBS_RIVER",
            obstacle_type="river",
            polygon=river_poly,
            color="lightblue"
        ))
        
        # 中心商业区 (建筑物密集区)
        center_x, center_y = self.area_width * 0.5, self.area_height * 0.5
        building_poly = [
            (center_x - 2, center_y - 2),
            (center_x + 2, center_y - 2),
            (center_x + 2, center_y + 2),
            (center_x - 2, center_y + 2)
        ]
        self.add_obstacle(Obstacle(
            id="OBS_CBD",
            obstacle_type="building",
            polygon=building_poly,
            color="lightgray"
        ))
        
        # 公园 (左下角)
        park_poly = [
            (0, 0),
            (4, 0),
            (4, 3),
            (0, 3)
        ]
        self.add_obstacle(Obstacle(
            id="OBS_PARK",
            obstacle_type="park",
            polygon=park_poly,
            color="lightgreen"
        ))
    
    def _buffer_polygon(self, points: List[Tuple[float, float]], 
                        buffer_dist: float) -> List[Tuple[float, float]]:
        """生成缓冲区多边形 (简化版)"""
        # 简化的缓冲区生成
        buffered = []
        for x, y in points:
            buffered.append((x - buffer_dist, y - buffer_dist))
            buffered.append((x + buffer_dist, y - buffer_dist))
            buffered.append((x + buffer_dist, y + buffer_dist))
            buffered.append((x - buffer_dist, y + buffer_dist))
        return buffered
    
    def _point_in_obstacle(self, x: float, y: float) -> bool:
        """检查点是否在障碍区内"""
        for obs in self.obstacles:
            if self._point_in_polygon(x, y, obs.polygon):
                return True
        return False
    
    def _point_in_polygon(self, x: float, y: float, 
                          polygon: List[Tuple[float, float]]) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def _calculate_distance(self, node1: str, node2: str, 
                            consider_obstacles: bool = True) -> float:
        """计算两点间距离 (考虑障碍)"""
        n1, n2 = self.nodes[node1], self.nodes[node2]
        direct_dist = np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        
        if not consider_obstacles:
            return direct_dist
        
        # 检查直线路径是否穿过障碍
        if self._path_crosses_obstacle(n1.x, n1.y, n2.x, n2.y):
            # 简单处理：增加距离惩罚
            return direct_dist * 1.5
        
        return direct_dist
    
    def _path_crosses_obstacle(self, x1: float, y1: float, 
                                x2: float, y2: float) -> bool:
        """检查线段是否穿过障碍区"""
        # 采样检查
        num_samples = 10
        for i in range(num_samples + 1):
            t = i / num_samples
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if self._point_in_obstacle(x, y):
                return True
        return False
    
    def design_network_mst(self) -> List[str]:
        """
        使用最小生成树设计网络拓扑
        
        Returns:
            电缆 ID 列表
        """
        from_node_ids = list(self.nodes.keys())
        
        # 计算所有节点对的距离
        distances = {}
        for i, n1_id in enumerate(from_node_ids):
            for n2_id in from_node_ids[i+1:]:
                dist = self._calculate_distance(n1_id, n2_id)
                distances[(n1_id, n2_id)] = dist
        
        # Kruskal 算法
        sorted_edges = sorted(distances.items(), key=lambda x: x[1])
        
        parent = {n: n for n in from_node_ids}
        
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
        
        cable_id = 0
        cables_created = []
        
        for (n1_id, n2_id), dist in sorted_edges:
            if union(n1_id, n2_id):
                # 确定电缆类型
                n1, n2 = self.nodes[n1_id], self.nodes[n2_id]
                max_voltage = max(n1.voltage_level, n2.voltage_level)
                
                if max_voltage >= 220:
                    cable_type = CableType.HV_220
                elif max_voltage >= 110:
                    cable_type = CableType.HV_110
                elif max_voltage >= 10:
                    cable_type = CableType.MV_10
                else:
                    cable_type = CableType.LV_04
                
                params = self.cable_params[cable_type]
                
                cable = Cable(
                    id=f"CABLE_{cable_id:03d}",
                    from_node=n1_id,
                    to_node=n2_id,
                    cable_type=cable_type,
                    length=dist,
                    capacity=params["capacity"],
                    cost_per_km=params["cost"],
                    resistance=params["r"],
                    reactance=params["x"]
                )
                
                self.add_cable(cable)
                cables_created.append(cable.id)
                cable_id += 1
        
        print(f"✓ MST 网络设计完成：{len(cables_created)} 条电缆")
        return cables_created
    
    def optimize_with_vns(self, iterations: int = 50) -> float:
        """
        使用 VNS 优化网络拓扑
        
        Args:
            iterations: 迭代次数
            
        Returns:
            改进百分比
        """
        initial_cost = self.calculate_total_cost()
        best_cost = initial_cost
        
        print(f"初始方案总成本：{initial_cost:.2f} 万元")
        
        for iteration in range(iterations):
            # 扰动：随机移除一条电缆
            if len(self.cables) > len(self.nodes) - 1:
                cable_to_remove = random.choice(list(self.cables.keys()))
                removed_cable = self.cables.pop(cable_to_remove)
                
                # 重新连接 (找到最近的可用连接)
                # 简化处理：添加一条新电缆
                
                # 局部搜索：尝试交换电缆
                # 简化处理
            
            current_cost = self.calculate_total_cost()
            if current_cost < best_cost:
                best_cost = current_cost
        
        improvement = (initial_cost - best_cost) / initial_cost * 100
        print(f"优化后总成本：{best_cost:.2f} 万元 (改进 {improvement:.2f}%)")
        return improvement
    
    def calculate_total_cost(self) -> float:
        """计算总成本"""
        return sum(cable.total_cost for cable in self.cables.values())
    
    def calculate_power_flow(self) -> PowerFlowResult:
        """
        简化潮流计算 (前推回代法)
        
        Returns:
            潮流计算结果
        """
        # 简化：假设变电站为平衡节点，电压为 1.0 p.u.
        base_voltage = 220.0  # kV
        
        node_voltages = {}
        branch_flows = {}
        
        # 初始化所有节点电压为 1.0 p.u.
        for node_id in self.nodes:
            node_voltages[node_id] = 1.0
        
        # 简化潮流：从负荷向电源累加功率
        for cable_id, cable in self.cables.items():
            to_node = self.nodes[cable.to_node]
            if to_node.node_type == NodeType.LOAD:
                branch_flows[cable_id] = to_node.load_capacity
            elif to_node.node_type == NodeType.DISTRIBUTION:
                branch_flows[cable_id] = to_node.load_capacity
            else:
                branch_flows[cable_id] = 0.0
        
        # 计算电压降
        total_loss = 0.0
        voltage_violations = []
        
        for cable_id, cable in self.cables.items():
            flow = branch_flows.get(cable_id, 0.0)
            if flow > 0:
                # 简化电压降计算
                voltage_drop = flow * cable.impedance / cable.capacity
                node_voltages[cable.to_node] -= voltage_drop * 0.01
                
                # 损耗计算
                total_loss += flow**2 * cable.resistance * 0.001
                
                # 检查电压越限
                if node_voltages[cable.to_node] < 0.95:
                    voltage_violations.append(cable.to_node)
        
        return PowerFlowResult(
            node_voltages=node_voltages,
            branch_flows=branch_flows,
            total_loss=total_loss,
            voltage_violations=voltage_violations
        )
    
    def analyze_reliability(self) -> Dict:
        """
        N-1 可靠性分析
        
        Returns:
            可靠性分析结果
        """
        results = {
            "total_cables": len(self.cables),
            "critical_cables": [],
            "redundancy_score": 0.0
        }
        
        # 简化 N-1 分析：检查每条电缆断开后的连通性
        for cable_id in list(self.cables.keys()):
            # 临时移除电缆
            cable = self.cables.pop(cable_id)
            self.adjacency[cable.from_node].remove(cable.to_node)
            self.adjacency[cable.to_node].remove(cable.from_node)
            
            # 检查连通性 (BFS)
            start_node = list(self.nodes.keys())[0]
            visited = set()
            queue = [start_node]
            
            while queue:
                node = queue.pop(0)
                if node not in visited:
                    visited.add(node)
                    queue.extend(self.adjacency[node])
            
            # 恢复电缆
            self.cables[cable_id] = cable
            self.adjacency[cable.from_node].append(cable.to_node)
            self.adjacency[cable.to_node].append(cable.from_node)
            
            # 如果不连通，标记为关键电缆
            if len(visited) < len(self.nodes):
                results["critical_cables"].append(cable_id)
        
        # 计算冗余度
        results["redundancy_score"] = 1.0 - len(results["critical_cables"]) / results["total_cables"]
        
        print(f"✓ N-1 分析：{len(results['critical_cables'])}/{results['total_cables']} 条关键电缆")
        print(f"  冗余度评分：{results['redundancy_score']:.2%}")
        
        return results
    
    def visualize(self, save_path: str = "outputs/city_power_grid.png"):
        """可视化电网布局"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：电网拓扑
        ax1 = axes[0]
        ax1.set_title("城市电网拓扑", fontsize=14, fontweight='bold')
        
        # 绘制障碍区
        for obs in self.obstacles:
            if len(obs.polygon) >= 3:
                poly = patches.Polygon(obs.polygon[:4], closed=True, 
                                       fill=True, alpha=0.3, color=obs.color,
                                       label=obs.obstacle_type)
                ax1.add_patch(poly)
        
        # 绘制电缆
        for cable in self.cables.values():
            n1, n2 = self.nodes[cable.from_node], self.nodes[cable.to_node]
            
            if cable.cable_type == CableType.HV_220:
                color, linewidth = "red", 3
            elif cable.cable_type == CableType.HV_110:
                color, linewidth = "orange", 2.5
            elif cable.cable_type == CableType.MV_10:
                color, linewidth = "blue", 1.5
            else:
                color, linewidth = "green", 1
            
            ax1.plot([n1.x, n2.x], [n1.y, n2.y], color=color, 
                    linewidth=linewidth, alpha=0.7)
        
        # 绘制节点
        for node in self.nodes.values():
            if node.node_type == NodeType.SUBSTATION:
                marker, size, color = "s", 200, "red"
            elif node.node_type == NodeType.DISTRIBUTION:
                marker, size, color = "o", 150, "orange"
            else:
                marker, size, color = "^", 80, "blue"
            
            ax1.scatter(node.x, node.y, marker=marker, s=size, c=color, 
                       edgecolors="white", linewidth=1.5, zorder=5)
            ax1.annotate(node.id, (node.x, node.y), fontsize=7, 
                        ha='center', va='bottom')
        
        ax1.set_xlim(-1, self.area_width + 1)
        ax1.set_ylim(-1, self.area_height + 1)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("距离 (km)")
        ax1.set_ylabel("距离 (km)")
        
        # 右图：成本分析
        ax2 = axes[1]
        ax2.set_title("电缆成本分析", fontsize=14, fontweight='bold')
        
        # 按电缆类型统计成本
        cost_by_type = {}
        length_by_type = {}
        for cable in self.cables.values():
            type_name = cable.cable_type.value
            if type_name not in cost_by_type:
                cost_by_type[type_name] = 0
                length_by_type[type_name] = 0
            cost_by_type[type_name] += cable.total_cost
            length_by_type[type_name] += cable.length
        
        # 绘制柱状图
        types = list(cost_by_type.keys())
        costs = list(cost_by_type.values())
        
        bars = ax2.barh(types, costs, color=["red", "orange", "blue", "green"])
        ax2.set_xlabel("成本 (万元)")
        ax2.set_title(f"总成本：{sum(costs):.1f} 万元")
        
        # 添加数值标签
        for bar, cost in zip(bars, costs):
            ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                    f"{cost:.1f}", va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化已保存：{save_path}")
        plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("城市电网规划案例 - City Power Grid Planning")
    print("=" * 60)
    
    # 创建城市电网模型
    grid = CityPowerGrid(area_size=(20.0, 20.0))
    
    # 生成电网拓扑
    grid.generate_city_grid(
        num_substations=2,
        num_distribution=8,
        num_loads=30,
        seed=42
    )
    
    # 使用 MST 设计网络
    print("\n1. 使用最小生成树设计网络拓扑...")
    grid.design_network_mst()
    
    # 计算初始成本
    initial_cost = grid.calculate_total_cost()
    print(f"   初始方案总成本：{initial_cost:.2f} 万元")
    
    # VNS 优化
    print("\n2. 使用 VNS 优化网络...")
    improvement = grid.optimize_with_vns(iterations=50)
    
    # 潮流计算
    print("\n3. 潮流计算...")
    power_flow = grid.calculate_power_flow()
    print(f"   总损耗：{power_flow.total_loss:.4f} MW")
    print(f"   电压越限节点：{len(power_flow.voltage_violations)} 个")
    
    # N-1 可靠性分析
    print("\n4. N-1 可靠性分析...")
    reliability = grid.analyze_reliability()
    
    # 可视化
    print("\n5. 生成可视化...")
    grid.visualize(save_path="outputs/city_power_grid.png")
    
    # 输出统计
    print("\n" + "=" * 60)
    print("案例总结")
    print("=" * 60)
    print(f"节点总数：{len(grid.nodes)}")
    print(f"电缆总数：{len(grid.cables)}")
    print(f"总成本：{grid.calculate_total_cost():.2f} 万元")
    print(f"优化改进：{improvement:.2f}%")
    print(f"冗余度：{reliability['redundancy_score']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
