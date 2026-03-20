#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海上风电场线缆布线优化案例研究

案例背景:
- 海上风电场包含多个风力涡轮机 (WTG)
- 需要设计集电系统电缆网络将所有 WTG 连接到升压站 (OSS)
- 目标：最小化电缆总成本，同时满足电气约束

关键约束:
1. 电缆容量限制 (电流/功率)
2. 电缆类型选择 (不同截面、不同成本)
3. 拓扑约束 (辐射状/树状结构)
4. 海床地形影响 (安装成本)

作者：智子 (Sophon)
日期：2026-03-20
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import LineCollection
import networkx as nx
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class WindTurbine:
    """风力涡轮机"""
    id: int
    x: float  # x 坐标 (km)
    y: float  # y 坐标 (km)
    capacity: float  # 单机容量 (MW)
    type: str = "WTG"  # WTG 或 OSS


@dataclass
class CableType:
    """电缆类型"""
    name: str
    cross_section: float  # 截面积 (mm²)
    max_current: float  # 最大电流 (A)
    max_power: float  # 最大传输功率 (MW)
    cost_per_km: float  # 单位长度成本 (万元/km)
    resistance: float  # 电阻 (Ω/km)


@dataclass
class SeaCondition:
    """海况条件"""
    depth: float  # 水深 (m)
    seabed_type: str  # 海床类型：sand/rock/clay
    installation_difficulty: float  # 安装难度系数 (1.0-3.0)


# ============================================================================
# 海上风电场环境建模
# ============================================================================

class OffshoreWindFarm:
    """海上风电场模型"""
    
    def __init__(self, name: str = "Demo Wind Farm"):
        self.name = name
        self.turbines: List[WindTurbine] = []
        self.cable_types: List[CableType] = []
        self.sea_conditions: Dict[Tuple[int, int], SeaCondition] = {}
        self.graph = nx.Graph()
        
    def add_turbine(self, x: float, y: float, capacity: float, 
                    turbine_type: str = "WTG") -> int:
        """添加风力涡轮机"""
        turbine_id = len(self.turbines)
        turbine = WindTurbine(
            id=turbine_id,
            x=x, y=y,
            capacity=capacity,
            type=turbine_type
        )
        self.turbines.append(turbine)
        self.graph.add_node(turbine_id, pos=(x, y), turbine=turbine)
        return turbine_id
    
    def add_cable_type(self, name: str, cross_section: float, 
                       max_current: float, max_power: float,
                       cost_per_km: float, resistance: float):
        """添加电缆类型"""
        cable = CableType(name, cross_section, max_current, 
                         max_power, cost_per_km, resistance)
        self.cable_types.append(cable)
        
    def generate_grid_layout(self, rows: int = 4, cols: int = 5,
                            spacing_x: float = 1.0, spacing_y: float = 1.5,
                            turbine_capacity: float = 8.0,
                            oss_position: str = "center") -> None:
        """
        生成网格状风机布局
        
        参数:
            rows: 行数
            cols: 列数
            spacing_x: X 方向间距 (km)
            spacing_y: Y 方向间距 (km)
            turbine_capacity: 单机容量 (MW)
            oss_position: 升压站位置 (center/corner/edge)
        """
        # 生成风机位置
        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x
                y = i * spacing_y
                self.add_turbine(x, y, turbine_capacity, "WTG")
        
        # 添加升压站 (OSS)
        if oss_position == "center":
            oss_x = (cols - 1) * spacing_x / 2
            oss_y = (rows - 1) * spacing_y / 2
        elif oss_position == "corner":
            oss_x = 0
            oss_y = 0
        else:  # edge
            oss_x = (cols - 1) * spacing_x / 2
            oss_y = 0
        
        oss_id = self.add_turbine(oss_x, oss_y, 0, "OSS")
        print(f"✓ 生成 {rows}x{cols} 风电场布局，共 {len(self.turbines)} 个节点")
        print(f"  - 风力涡轮机 (WTG): {len(self.turbines) - 1} 个")
        print(f"  - 升压站 (OSS): 1 个 (位置：{oss_position})")
        
    def add_typical_cable_types(self):
        """添加典型海缆类型"""
        # 常见海缆规格 (基于实际工程数据)
        self.add_cable_type("XLPE-95", 95, 280, 50, 15.0, 0.193)
        self.add_cable_type("XLPE-150", 150, 350, 80, 22.0, 0.128)
        self.add_cable_type("XLPE-240", 240, 450, 120, 32.0, 0.077)
        self.add_cable_type("XLPE-400", 400, 600, 180, 48.0, 0.047)
        self.add_cable_type("XLPE-630", 630, 800, 250, 68.0, 0.028)
        print(f"✓ 添加 {len(self.cable_types)} 种电缆类型")
        
    def calculate_distance(self, id1: int, id2: int) -> float:
        """计算两个节点间的欧氏距离"""
        t1 = self.turbines[id1]
        t2 = self.turbines[id2]
        return np.sqrt((t1.x - t2.x)**2 + **(t1.y - t2.y)2)
    
    def build_complete_graph(self):
        """构建完全图 (所有节点间都有边)"""
        n = len(self.turbines)
        for i in range(n):
            for j in range(i+1, n):
                dist = self.calculate_distance(i, j)
                self.graph.add_edge(i, j, distance=dist, weight=dist)
        print(f"✓ 构建完全图，共 {self.graph.number_of_edges()} 条边")


# ============================================================================
# 优化算法实现
# ============================================================================

class CableRoutingOptimizer:
    """线缆布线优化器"""
    
    def __init__(self, wind_farm: OffshoreWindFarm):
        self.wf = wind_farm
        self.solution = None
        
    def minimum_spanning_tree(self, method: str = "prim") -> nx.Graph:
        """
        使用最小生成树构建初始拓扑
        
        参数:
            method: 'prim' 或 'kruskal'
        """
        if method == "prim":
            mst = nx.minimum_spanning_tree(self.wf.graph, weight='weight')
        else:
            mst = nx.minimum_spanning_tree(self.wf.graph, weight='weight')
        
        print(f"✓ MST 构建完成 ({method}算法)")
        print(f"  - 边数：{mst.number_of_edges()}")
        total_length = sum(d['weight'] for u, v, d in mst.edges(data=True))
        print(f"  - 总长度：{total_length:.2f} km")
        return mst
    
    def optimize_cable_selection(self, topology: nx.Graph) -> Dict:
        """
        优化电缆选型
        
        基于每条支路的功率流选择合适的电缆截面
        """
        # 找到 OSS 节点 (通常是 0 号节点或 type="OSS"的节点)
        oss_node = None
        for node in topology.nodes():
            if self.wf.turbines[node].type == "OSS":
                oss_node = node
                break
        if oss_node is None:
            oss_node = 0
        
        # 计算每个节点的功率流 (从叶子到根)
        power_flow = {}
        for node in topology.nodes():
            power_flow[node] = self.wf.turbines[node].capacity
        
        # BFS 从 OSS 开始遍历
        visited = {oss_node}
        queue = [oss_node]
        cable_selection = {}
        
        while queue:
            current = queue.pop(0)
            neighbors = list(topology.neighbors(current))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
                    # 计算该支路的总功率
                    branch_power = self._calculate_branch_power(
                        topology, current, neighbor, oss_node
                    )
                    
                    # 选择合适的电缆
                    selected_cable = self._select_cable_for_power(branch_power)
                    cable_selection[(current, neighbor)] = {
                        'cable': selected_cable,
                        'power': branch_power,
                        'distance': topology[current][neighbor]['weight']
                    }
        
        return cable_selection
    
    def _calculate_branch_power(self, topology: nx.Graph, 
                                from_node: int, to_node: int, 
                                oss_node: int) -> float:
        """计算支路功率流"""
        # 从 to_node 开始，计算子树总功率
        total_power = 0.0
        visited = {from_node}
        queue = [to_node]
        
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                total_power += self.wf.turbines[node].capacity
                for neighbor in topology.neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return total_power
    
    def _select_cable_for_power(self, power: float) -> CableType:
        """根据功率选择电缆"""
        for cable in sorted(self.wf.cable_types, 
                          key=lambda c: c.max_power):
            if cable.max_power >= power * 1.1:  # 10% 安全裕度
                return cable
        # 如果没有合适的，返回最大的
        return max(self.wf.cable_types, key=lambda c: c.max_power)
    
    def calculate_total_cost(self, cable_selection: Dict) -> float:
        """计算总成本"""
        total_cost = 0.0
        for (u, v), info in cable_selection.items():
            cable = info['cable']
            distance = info['distance']
            cost = cable.cost_per_km * distance
            total_cost += cost
        return total_cost
    
    def optimize_with_vns(self, initial_topology: nx.Graph, 
                         max_iterations: int = 100) -> nx.Graph:
        """
        使用变邻域搜索优化拓扑
        
        通过交换边来改进 MST 初始解
        """
        current_topology = initial_topology.copy()
        best_topology = current_topology.copy()
        best_cost = self._evaluate_topology(current_topology)
        
        print(f"\n🔍 开始 VNS 优化...")
        print(f"  初始成本：{best_cost:.2f} 万元")
        
        for iteration in range(max_iterations):
            # 扰动：随机交换一条边
            neighbor = self._perturb_topology(current_topology)
            
            # 局部搜索：尝试改进
            improved = self._local_search(neighbor)
            
            # 评估
            neighbor_cost = self._evaluate_topology(improved)
            
            if neighbor_cost < best_cost:
                best_topology = improved.copy()
                best_cost = neighbor_cost
                current_topology = improved.copy()
                if iteration % 20 == 0:
                    print(f"  迭代 {iteration}: 新最优成本 = {best_cost:.2f} 万元")
        
        print(f"✓ VNS 优化完成")
        print(f"  最终成本：{best_cost:.2f} 万元")
        print(f"  改进幅度：{(1 - best_cost/self._evaluate_topology(initial_topology))*100:.1f}%")
        
        return best_topology
    
    def _evaluate_topology(self, topology: nx.Graph) -> float:
        """评估拓扑成本"""
        cable_selection = self.optimize_cable_selection(topology)
        return self.calculate_total_cost(cable_selection)
    
    def _perturb_topology(self, topology: nx.Graph) -> nx.Graph:
        """扰动拓扑 (随机交换边)"""
        perturbed = topology.copy()
        
        # 随机选择一条边移除
        edges = list(perturbed.edges())
        if len(edges) > 0:
            edge_to_remove = edges[np.random.randint(len(edges))]
            perturbed.remove_edge(*edge_to_remove)
            
            # 添加一条新边保持连通性
            nodes = list(perturbed.nodes())
            for _ in range(10):  # 尝试 10 次
                n1, n2 = np.random.choice(nodes, 2, replace=False)
                if not perturbed.has_edge(n1, n2):
                    if nx.has_path(perturbed, n1, n2):
                        continue  # 已经连通，不需要加边
                    perturbed.add_edge(n1, n2, 
                                      weight=self.wf.calculate_distance(n1, n2))
                    break
        
        return perturbed
    
    def _local_search(self, topology: nx.Graph) -> nx.Graph:
        """局部搜索 (2-opt 交换)"""
        improved = topology.copy()
        
        # 尝试所有可能的边交换
        edges = list(improved.edges())
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                # 尝试交换
                test = improved.copy()
                u1, v1 = edges[i]
                u2, v2 = edges[j]
                
                test.remove_edge(u1, v1)
                test.remove_edge(u2, v2)
                
                # 检查是否保持连通
                if not nx.is_connected(test):
                    # 恢复
                    test.add_edge(u1, v1, weight=improved[u1][v1]['weight'])
                    test.add_edge(u2, v2, weight=improved[u2][v2]['weight'])
                    continue
                
                # 尝试新的连接方式
                test.add_edge(u1, u2, weight=self.wf.calculate_distance(u1, u2))
                test.add_edge(v1, v2, weight=self.wf.calculate_distance(v1, v2))
                
                if self._evaluate_topology(test) < self._evaluate_topology(improved):
                    improved = test
        
        return improved


# ============================================================================
# 可视化
# ============================================================================

class WindFarmVisualizer:
    """风电场可视化"""
    
    def __init__(self, wind_farm: OffshoreWindFarm):
        self.wf = wind_farm
        
    def plot_layout(self, topology: Optional[nx.Graph] = None,
                   cable_selection: Optional[Dict] = None,
                   title: str = "海上风电场布局",
                   save_path: Optional[str] = None):
        """绘制风电场布局"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 绘制风机
        for turbine in self.wf.turbines:
            if turbine.type == "OSS":
                # 升压站用红色大圆
                circle = Circle((turbine.x, turbine.y), 0.15, 
                              color='red', zorder=5, label='升压站 (OSS)')
                ax.add_patch(circle)
                ax.text(turbine.x, turbine.y + 0.2, f'OSS', 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            else:
                # 风机用蓝色小圆
                circle = Circle((turbine.x, turbine.y), 0.08, 
                              color='steelblue', zorder=3)
                ax.add_patch(circle)
                ax.text(turbine.x, turbine.y + 0.12, f'WTG-{turbine.id}', 
                       ha='center', va='bottom', fontsize=8)
        
        # 绘制电缆
        if topology is not None:
            for u, v in topology.edges():
                t1 = self.wf.turbines[u]
                t2 = self.wf.turbines[v]
                
                # 根据电缆类型设置线宽和颜色
                if cable_selection is not None:
                    key = (min(u,v), max(u,v))
                    if key in cable_selection:
                        cable = cable_selection[key]['cable']
                        # 根据截面积设置线宽
                        linewidth = cable.cross_section / 100
                        # 根据电缆类型设置颜色
                        if cable.cross_section <= 150:
                            color = 'green'
                        elif cable.cross_section <= 400:
                            color = 'orange'
                        else:
                            color = 'red'
                    else:
                        linewidth = 1.5
                        color = 'gray'
                else:
                    linewidth = 1.5
                    color = 'gray'
                
                ax.plot([t1.x, t2.x], [t1.y, t2.y], 
                       color=color, linewidth=linewidth, zorder=1)
        
        ax.set_xlabel('X 坐标 (km)')
        ax.set_ylabel('Y 坐标 (km)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 添加图例
        if cable_selection is not None:
            legend_elements = [
                plt.Line2D([0], [0], color='green', linewidth=1.5, label='XLPE-95/150'),
                plt.Line2D([0], [0], color='orange', linewidth=3, label='XLPE-240/400'),
                plt.Line2D([0], [0], color='red', linewidth=5, label='XLPE-630'),
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图像已保存：{save_path}")
        
        plt.show()
    
    def plot_cost_breakdown(self, cable_selection: Dict):
        """绘制成本分解"""
        # 按电缆类型统计成本
        cost_by_type = {}
        length_by_type = {}
        
        for (u, v), info in cable_selection.items():
            cable = info['cable']
            distance = info['distance']
            cost = cable.cost_per_km * distance
            
            name = cable.name
            if name not in cost_by_type:
                cost_by_type[name] = 0
                length_by_type[name] = 0
            cost_by_type[name] += cost
            length_by_type[name] += distance
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 成本饼图
        colors = plt.cm.Set3(np.linspace(0, 1, len(cost_by_type)))
        wedges1, texts1, autotexts1 = ax1.pie(
            cost_by_type.values(),
            labels=cost_by_type.keys(),
            autopct='%1.1f%%',
            colors=colors
        )
        ax1.set_title('成本分布 (按电缆类型)', fontsize=12, fontweight='bold')
        
        # 长度柱状图
        ax2.bar(length_by_type.keys(), length_by_type.values(), color=colors)
        ax2.set_xlabel('电缆类型')
        ax2.set_ylabel('长度 (km)')
        ax2.set_title('电缆长度分布', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/offshore_cost_breakdown.png', dpi=150, bbox_inches='tight')
        print(f"✓ 成本分解图已保存：outputs/offshore_cost_breakdown.png")
        plt.show()


# ============================================================================
# 主函数 - 完整案例演示
# ============================================================================

def main():
    """海上风电场布线优化完整案例"""
    print("=" * 70)
    print("🌊 海上风电场线缆布线优化案例研究")
    print("=" * 70)
    
    # 1. 创建风电场模型
    print("\n📐 步骤 1: 创建风电场模型")
    wf = OffshoreWindFarm("East China Sea Wind Farm")
    wf.generate_grid_layout(rows=4, cols=5, spacing_x=1.2, spacing_y=1.8,
                           turbine_capacity=8.0, oss_position="center")
    wf.add_typical_cable_types()
    wf.build_complete_graph()
    
    # 2. 使用 MST 构建初始拓扑
    print("\n📐 步骤 2: 构建初始拓扑 (最小生成树)")
    optimizer = CableRoutingOptimizer(wf)
    initial_topology = optimizer.minimum_spanning_tree(method="prim")
    
    # 3. 优化电缆选型
    print("\n📐 步骤 3: 优化电缆选型")
    cable_selection = optimizer.optimize_cable_selection(initial_topology)
    initial_cost = optimizer.calculate_total_cost(cable_selection)
    print(f"  初始方案总成本：{initial_cost:.2f} 万元")
    
    # 4. 使用 VNS 进一步优化拓扑
    print("\n📐 步骤 4: VNS 拓扑优化")
    optimized_topology = optimizer.optimize_with_vns(initial_topology, max_iterations=50)
    
    # 5. 重新优化电缆选型
    optimized_cable_selection = optimizer.optimize_cable_selection(optimized_topology)
    optimized_cost = optimizer.calculate_total_cost(optimized_cable_selection)
    print(f"  优化方案总成本：{optimized_cost:.2f} 万元")
    print(f"  总改进幅度：{(1 - optimized_cost/initial_cost)*100:.1f}%")
    
    # 6. 可视化
    print("\n📐 步骤 5: 可视化结果")
    viz = WindFarmVisualizer(wf)
    
    # 初始方案
    viz.plot_layout(
        initial_topology, cable_selection,
        title=f"初始方案 (MST) - 总成本：{initial_cost:.2f} 万元",
        save_path="outputs/offshore_initial_layout.png"
    )
    
    # 优化方案
    viz.plot_layout(
        optimized_topology, optimized_cable_selection,
        title=f"优化方案 (VNS) - 总成本：{optimized_cost:.2f} 万元",
        save_path="outputs/offshore_optimized_layout.png"
    )
    
    # 成本分解
    viz.plot_cost_breakdown(optimized_cable_selection)
    
    # 7. 输出详细报告
    print("\n" + "=" * 70)
    print("📊 优化结果汇总")
    print("=" * 70)
    print(f"风电场规模：{len(wf.turbines) - 1} 台风机 + 1 个升压站")
    print(f"总装机容量：{sum(t.capacity for t in wf.turbines if t.type == 'WTG'):.0f} MW")
    print(f"\n初始方案 (MST):")
    print(f"  - 电缆总长度：{sum(info['distance'] for info in cable_selection.values()):.2f} km")
    print(f"  - 总成本：{initial_cost:.2f} 万元")
    print(f"\n优化方案 (VNS):")
    print(f"  - 电缆总长度：{sum(info['distance'] for info in optimized_cable_selection.values()):.2f} km")
    print(f"  - 总成本：{optimized_cost:.2f} 万元")
    print(f"  - 成本节省：{initial_cost - optimized_cost:.2f} 万元 ({(1-optimized_cost/initial_cost)*100:.1f}%)")
    
    # 电缆使用统计
    print(f"\n电缆使用统计:")
    cable_usage = {}
    for info in optimized_cable_selection.values():
        name = info['cable'].name
        if name not in cable_usage:
            cable_usage[name] = {'count': 0, 'length': 0}
        cable_usage[name]['count'] += 1
        cable_usage[name]['length'] += info['distance']
    
    for name, stats in sorted(cable_usage.items()):
        print(f"  - {name}: {stats['count']} 条，总长 {stats['length']:.2f} km")
    
    print("\n" + "=" * 70)
    print("✅ 案例研究完成")
    print("=" * 70)
    
    return wf, optimized_topology, optimized_cable_selection


if __name__ == "__main__":
    wf, topology, cable_sel = main()
