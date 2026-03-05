#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变邻域搜索 (Variable Neighborhood Search, VNS) 算法实现
=====================================================

作者：智子 (Sophon)
日期：2026-03-05
用途：线缆布线优化 - VNS 算法完整实现

VNS 核心思想：
- 系统性地改变邻域结构，避免陷入局部最优
- 通过不同邻域的切换，实现全局搜索

算法流程：
1. 初始化当前解
2. 对于每个邻域结构 k = 1, 2, ..., k_max：
   a. 扰动：在当前解的第 k 个邻域内随机扰动
   b. 局部搜索：在新解的邻域内搜索最优
   c. 如果改进：接受新解，重置 k=1
   d. 否则：k = k+1，尝试更大的邻域
3. 重复直到满足终止条件

邻域结构示例（TSP/路径问题）：
- N1: 交换相邻两个城市
- N2: 交换任意两个城市
- N3: 逆转一段路径
- N4: 插入操作（移动一个城市到另一位置）
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
import random
import time
from dataclasses import dataclass
import heapq


# ==================== 数据结构 ====================

@dataclass
class VNSConfig:
    """VNS 算法配置"""
    k_max: int = 4  # 最大邻域数
    max_iterations: int = 1000  # 最大迭代次数
    max_no_improve: int = 100  # 无改进最大迭代次数
    time_limit: float = 60.0  # 时间限制（秒）
    verbose: bool = True  # 是否输出详细信息


@dataclass
class VNSResult:
    """VNS 算法结果"""
    best_solution: List[int]
    best_cost: float
    iterations: int
    convergence_history: List[Tuple[int, float]]
    neighborhood_history: List[int]
    total_time: float


# ==================== 邻域操作 ====================

class NeighborhoodOperators:
    """邻域操作算子集合"""
    
    @staticmethod
    def swap_adjacent(solution: List[int], idx: int) -> List[int]:
        """N1: 交换相邻两个元素"""
        new_sol = solution.copy()
        n = len(new_sol)
        j = (idx + 1) % n
        new_sol[idx], new_sol[j] = new_sol[j], new_sol[idx]
        return new_sol
    
    @staticmethod
    def swap_any(solution: List[int], idx1: int, idx2: int) -> List[int]:
        """N2: 交换任意两个元素"""
        new_sol = solution.copy()
        new_sol[idx1], new_sol[idx2] = new_sol[idx2], new_sol[idx1]
        return new_sol
    
    @staticmethod
    def reverse_segment(solution: List[int], start: int, end: int) -> List[int]:
        """N3: 逆转一段路径"""
        new_sol = solution.copy()
        if start > end:
            start, end = end, start
        new_sol[start:end+1] = reversed(new_sol[start:end+1])
        return new_sol
    
    @staticmethod
    def insert(solution: List[int], from_idx: int, to_idx: int) -> List[int]:
        """N4: 插入操作 - 将一个元素移动到另一位置"""
        new_sol = solution.copy()
        element = new_sol.pop(from_idx)
        new_sol.insert(to_idx, element)
        return new_sol
    
    @staticmethod
    def two_opt(solution: List[int], idx1: int, idx2: int) -> List[int]:
        """N5: 2-opt 操作 - 经典 TSP 邻域"""
        new_sol = solution.copy()
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        new_sol[idx1:idx2+1] = reversed(new_sol[idx1:idx2+1])
        return new_sol


# ==================== VNS 主算法 ====================

class VariableNeighborhoodSearch:
    """变邻域搜索算法实现"""
    
    def __init__(self, 
                 cost_func: Callable[[List[int]], float],
                 config: VNSConfig = None):
        """
        初始化 VNS
        
        参数:
            cost_func: 成本函数，输入路径列表，返回总成本
            config: VNS 配置参数
        """
        self.cost_func = cost_func
        self.config = config or VNSConfig()
        self.operators = NeighborhoodOperators()
        
        # 记录搜索历史
        self.convergence_history = []
        self.neighborhood_history = []
    
    def _get_neighborhood_solution(self, solution: List[int], k: int) -> List[int]:
        """
        在第 k 个邻域内生成新解
        
        邻域定义：
        k=1: 交换相邻 (N1)
        k=2: 交换任意 (N2)
        k=3: 路径逆转 (N3)
        k=4: 插入操作 (N4)
        k>=5: 2-opt (N5)
        """
        n = len(solution)
        
        if k == 1:
            # N1: 交换相邻
            idx = random.randint(0, n - 1)
            return self.operators.swap_adjacent(solution, idx)
        
        elif k == 2:
            # N2: 交换任意两个
            idx1, idx2 = random.sample(range(n), 2)
            return self.operators.swap_any(solution, idx1, idx2)
        
        elif k == 3:
            # N3: 逆转一段路径
            start = random.randint(0, n - 2)
            end = random.randint(start + 1, n - 1)
            return self.operators.reverse_segment(solution, start, end)
        
        elif k == 4:
            # N4: 插入操作
            from_idx = random.randint(0, n - 1)
            to_idx = random.randint(0, n - 1)
            while to_idx == from_idx:
                to_idx = random.randint(0, n - 1)
            return self.operators.insert(solution, from_idx, to_idx)
        
        else:
            # N5: 2-opt
            idx1, idx2 = random.sample(range(n), 2)
            return self.operators.two_opt(solution, idx1, idx2)
    
    def _local_search(self, solution: List[int], k: int) -> List[int]:
        """
        在当前解的第 k 个邻域内进行局部搜索
        
        使用最陡下降法：遍历邻域内所有解，选择最优的
        """
        n = len(solution)
        best_sol = solution.copy()
        best_cost = self.cost_func(solution)
        
        # 根据 k 选择邻域操作
        if k == 1:
            # N1: 尝试所有相邻交换
            for i in range(n):
                new_sol = self.operators.swap_adjacent(solution, i)
                new_cost = self.cost_func(new_sol)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_sol = new_sol
        
        elif k == 2:
            # N2: 尝试所有两两交换
            for i in range(n):
                for j in range(i + 1, n):
                    new_sol = self.operators.swap_any(solution, i, j)
                    new_cost = self.cost_func(new_sol)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_sol = new_sol
        
        elif k == 3:
            # N3: 尝试所有路径逆转
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_sol = self.operators.reverse_segment(solution, i, j)
                    new_cost = self.cost_func(new_sol)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_sol = new_sol
        
        elif k == 4:
            # N4: 尝试所有插入操作
            for i in range(n):
                for j in range(n):
                    if i != j:
                        new_sol = self.operators.insert(solution, i, j)
                        new_cost = self.cost_func(new_sol)
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_sol = new_sol
        
        return best_sol
    
    def solve(self, initial_solution: List[int]) -> VNSResult:
        """
        执行 VNS 算法
        
        参数:
            initial_solution: 初始解（路径排列）
        
        返回:
            VNSResult: 包含最优解和搜索历史
        """
        start_time = time.time()
        
        # 初始化
        current_solution = initial_solution.copy()
        current_cost = self.cost_func(current_solution)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        # 记录历史
        self.convergence_history = [(0, best_cost)]
        self.neighborhood_history = []
        
        k = 1  # 从第一个邻域开始
        iteration = 0
        no_improve_count = 0
        
        if self.config.verbose:
            print(f"VNS 开始搜索 - 初始成本：{current_cost:.2f}")
        
        # 主循环
        while (iteration < self.config.max_iterations and 
               no_improve_count < self.config.max_no_improve and
               time.time() - start_time < self.config.time_limit):
            
            iteration += 1
            
            # 1. 扰动 (Shaking)
            new_solution = self._get_neighborhood_solution(current_solution, k)
            
            # 2. 局部搜索
            new_solution = self._local_search(new_solution, k)
            new_cost = self.cost_func(new_solution)
            
            # 3. 判断是否接受
            if new_cost < best_cost:
                # 接受改进
                best_solution = new_solution.copy()
                best_cost = new_cost
                current_solution = new_solution.copy()
                current_cost = new_cost
                k = 1  # 重置邻域
                no_improve_count = 0
                
                if self.config.verbose and iteration % 10 == 0:
                    print(f"  迭代 {iteration}: 发现更优解 {best_cost:.2f} (邻域 N{k})")
            else:
                # 未改进，尝试更大的邻域
                k += 1
                if k > self.config.k_max:
                    k = 1  # 循环回到第一个邻域
                no_improve_count += 1
            
            # 记录历史
            self.convergence_history.append((iteration, best_cost))
            self.neighborhood_history.append(k)
        
        total_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"VNS 搜索完成 - 最优成本：{best_cost:.2f}, 迭代次数：{iteration}, 用时：{total_time:.2f}s")
        
        return VNSResult(
            best_solution=best_solution,
            best_cost=best_cost,
            iterations=iteration,
            convergence_history=self.convergence_history,
            neighborhood_history=self.neighborhood_history,
            total_time=total_time
        )


# ==================== 线缆布线应用 ====================

class CableRoutingVNS:
    """使用 VNS 解决线缆布线问题"""
    
    def __init__(self, points: np.ndarray, depot_idx: int = 0):
        """
        初始化布线问题
        
        参数:
            points: 所有点的坐标 (n, 2)
            depot_idx: 起点/终点索引（默认 0）
        """
        self.points = points
        self.depot_idx = depot_idx
        self.n = len(points)
        
        # 预计算距离矩阵
        self.dist_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """计算所有点对之间的距离"""
        n = self.n
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.points[i] - self.points[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist
    
    def calculate_route_cost(self, route: List[int]) -> float:
        """计算路径总成本（总距离）"""
        cost = 0.0
        for i in range(len(route) - 1):
            cost += self.dist_matrix[route[i], route[i + 1]]
        # 返回起点（如果需要回路）
        cost += self.dist_matrix[route[-1], route[0]]
        return cost
    
    def create_initial_solution(self, method: str = 'nearest') -> List[int]:
        """
        创建初始解
        
        参数:
            method: 'nearest' (最近邻), 'random' (随机)
        """
        if method == 'random':
            # 随机排列（保持起点固定）
            other_nodes = [i for i in range(self.n) if i != self.depot_idx]
            random.shuffle(other_nodes)
            return [self.depot_idx] + other_nodes
        
        elif method == 'nearest':
            # 最近邻启发式
            visited = {self.depot_idx}
            route = [self.depot_idx]
            current = self.depot_idx
            
            while len(visited) < self.n:
                # 找到最近的未访问节点
                best_next = -1
                best_dist = float('inf')
                for j in range(self.n):
                    if j not in visited and self.dist_matrix[current, j] < best_dist:
                        best_dist = self.dist_matrix[current, j]
                        best_next = j
                
                visited.add(best_next)
                route.append(best_next)
                current = best_next
            
            return route
        
        else:
            raise ValueError(f"未知方法：{method}")
    
    def optimize(self, method: str = 'nearest', config: VNSConfig = None) -> VNSResult:
        """
        使用 VNS 优化布线路径
        
        参数:
            method: 初始解生成方法
            config: VNS 配置
        
        返回:
            VNSResult: 优化结果
        """
        # 创建初始解
        initial_solution = self.create_initial_solution(method)
        initial_cost = self.calculate_route_cost(initial_solution)
        
        print(f"初始解成本：{initial_cost:.2f} (方法：{method})")
        
        # 创建 VNS 求解器
        vns = VariableNeighborhoodSearch(
            cost_func=self.calculate_route_cost,
            config=config
        )
        
        # 执行优化
        result = vns.solve(initial_solution)
        
        # 计算改进
        improvement = (initial_cost - result.best_cost) / initial_cost * 100
        
        print(f"\n优化结果:")
        print(f"  最优成本：{result.best_cost:.2f}")
        print(f"  改进幅度：{improvement:.2f}%")
        print(f"  迭代次数：{result.iterations}")
        print(f"  搜索时间：{result.total_time:.2f}s")
        
        return result
    
    def visualize(self, route: List[int], title: str = "VNS 优化结果"):
        """可视化布线路径"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：路径可视化
        ax1 = axes[0]
        coords = self.points[route]
        
        # 绘制所有点
        ax1.scatter(self.points[:, 0], self.points[:, 1], 
                   c='lightblue', s=100, edgecolors='gray', label='节点')
        
        # 绘制路径
        ax1.plot(coords[:, 0], coords[:, 1], 'o-', c='blue', 
                linewidth=2, markersize=8, label='布线路径')
        
        # 标记起点
        ax1.scatter(self.points[self.depot_idx, 0], 
                   self.points[self.depot_idx, 1],
                   c='red', s=150, marker='*', label='起点/终点')
        
        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 右图：收敛曲线
        ax2 = axes[1]
        iterations = [h[0] for h in self.vns_result.convergence_history]
        costs = [h[1] for h in self.vns_result.convergence_history]
        
        ax2.plot(iterations, costs, 'b-', linewidth=2)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('总成本')
        ax2.set_title('VNS 收敛曲线')
        ax2.grid(True, alpha=0.3)
        
        # 标注关键信息
        ax2.axhline(y=costs[-1], color='r', linestyle='--', 
                   label=f'最优：{costs[-1]:.2f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'vns_result_{time.strftime("%Y%m%d_%H%M%S")}.png', dpi=150)
        print(f"可视化已保存")
        plt.show()


# ==================== 示例运行 ====================

def example_cable_routing():
    """线缆布线优化示例"""
    print("=" * 60)
    print("变邻域搜索 (VNS) - 线缆布线优化示例")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    # 生成测试点（模拟 15 个设备位置）
    n_points = 15
    points = np.random.rand(n_points, 2) * 100  # 100x100 区域
    
    # 创建布线问题
    routing = CableRoutingVNS(points, depot_idx=0)
    
    # 配置 VNS
    config = VNSConfig(
        k_max=5,
        max_iterations=500,
        max_no_improve=50,
        time_limit=30.0,
        verbose=True
    )
    
    # 执行优化
    result = routing.optimize(method='nearest', config=config)
    
    # 保存结果
    routing.vns_result = result
    
    # 可视化
    routing.visualize(result.best_solution, 
                     f"VNS 优化 - 15 节点布线 (成本：{result.best_cost:.2f})")
    
    return routing, result


def compare_initial_methods():
    """比较不同初始解生成方法"""
    print("\n" + "=" * 60)
    print("比较不同初始解方法")
    print("=" * 60)
    
    np.random.seed(42)
    n_points = 20
    points = np.random.rand(n_points, 2) * 100
    
    routing = CableRoutingVNS(points, depot_idx=0)
    
    config = VNSConfig(
        k_max=4,
        max_iterations=300,
        max_no_improve=30,
        time_limit=20.0,
        verbose=False
    )
    
    methods = ['nearest', 'random']
    results = {}
    
    for method in methods:
        print(f"\n方法：{method}")
        result = routing.optimize(method=method, config=config)
        results[method] = result.best_cost
    
    print("\n" + "-" * 40)
    print("结果对比:")
    for method, cost in results.items():
        print(f"  {method:10s}: {cost:.2f}")
    
    best_method = min(results, key=results.get)
    print(f"\n最优初始方法：{best_method}")


def test_neighborhood_operators():
    """测试不同邻域操作"""
    print("\n" + "=" * 60)
    print("测试邻域操作")
    print("=" * 60)
    
    # 测试解
    solution = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"原始解：{solution}")
    
    ops = NeighborhoodOperators()
    
    # N1: 交换相邻
    n1 = ops.swap_adjacent(solution, 3)
    print(f"N1 (交换相邻 idx=3): {n1}")
    
    # N2: 交换任意
    n2 = ops.swap_any(solution, 2, 7)
    print(f"N2 (交换 idx=2,7): {n2}")
    
    # N3: 逆转
    n3 = ops.reverse_segment(solution, 2, 6)
    print(f"N3 (逆转 2-6): {n3}")
    
    # N4: 插入
    n4 = ops.insert(solution, 8, 2)
    print(f"N4 (插入 8→2): {n4}")
    
    # N5: 2-opt
    n5 = ops.two_opt(solution, 3, 7)
    print(f"N5 (2-opt 3-7): {n5}")


if __name__ == "__main__":
    # 运行示例
    example_cable_routing()
    
    # 比较初始方法
    compare_initial_methods()
    
    # 测试邻域操作
    test_neighborhood_operators()
    
    print("\n" + "=" * 60)
    print("VNS 算法演示完成!")
    print("=" * 60)
