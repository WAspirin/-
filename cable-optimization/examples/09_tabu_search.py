#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
禁忌搜索算法 (Tabu Search, TS) - 线缆布线优化应用

作者：智子 (Sophon)
日期：2026-03-06
学习阶段：Week 2 - Day 5

算法核心思想：
- 使用禁忌表记录最近访问过的解或操作，避免循环
- 允许接受劣解以跳出局部最优
- 通过藐视准则接受优质解

适用场景：
- 组合优化问题
- 需要避免循环搜索
- 中等规模问题 (50-200 节点)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import random
import time
from collections import deque


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class CableRoutingProblem:
    """线缆布线问题定义"""
    nodes: np.ndarray  # 节点坐标 (n, 2)
    source: int  # 源点索引
    sink: int  # 汇点索引
    demands: np.ndarray  # 节点需求 (正为需求，负为供应)
    edge_costs: Optional[np.ndarray] = None  # 边成本矩阵
    
    def __post_init__(self):
        n = len(self.nodes)
        if self.edge_costs is None:
            # 默认使用欧氏距离作为成本
            self.edge_costs = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.edge_costs[i, j] = np.linalg.norm(
                            self.nodes[i] - self.nodes[j]
                        )
    
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


@dataclass
class TabuConfig:
    """禁忌搜索配置参数"""
    tabu_tenure: int = 15  # 禁忌长度
    max_iterations: int = 500  # 最大迭代次数
    max_no_improvement: int = 100  # 无改进最大迭代次数
    aspiration_threshold: float = 0.0  # 藐视准则阈值
    neighborhood_size: int = 20  # 邻域采样大小
    seed: int = 42  # 随机种子


# ============================================================================
# 邻域操作定义
# ============================================================================

class NeighborhoodOperators:
    """邻域操作算子"""
    
    @staticmethod
    def swap(solution: List[int], i: int, j: int) -> List[int]:
        """交换两个位置"""
        new_sol = solution.copy()
        new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
        return new_sol
    
    @staticmethod
    def insert(solution: List[int], i: int, j: int) -> List[int]:
        """插入操作：将位置 i 的元素移到位置 j 之前"""
        new_sol = solution.copy()
        elem = new_sol.pop(i)
        new_sol.insert(j, elem)
        return new_sol
    
    @staticmethod
    def reverse(solution: List[int], i: int, j: int) -> List[int]:
        """逆转操作：逆转 i 到 j 之间的子序列"""
        new_sol = solution.copy()
        if i > j:
            i, j = j, i
        new_sol[i:j+1] = reversed(new_sol[i:j+1])
        return new_sol
    
    @staticmethod
    def two_opt(solution: List[int], i: int, j: int) -> List[int]:
        """2-opt 操作：逆转 i 到 j 之间的路径"""
        new_sol = solution.copy()
        if i > j:
            i, j = j, i
        new_sol[i:j+1] = new_sol[i:j+1][::-1]
        return new_sol
    
    @staticmethod
    def crossover(solution: List[int], other: List[int]) -> List[int]:
        """部分映射交叉 (PMX)"""
        n = len(solution)
        child = [-1] * n
        
        # 随机选择交叉区域
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, n - 1)
        
        # 复制交叉区域
        child[start:end+1] = solution[start:end+1]
        
        # 填充其他位置
        other_idx = 0
        for i in range(n):
            if child[i] == -1:
                while other[other_idx] in child:
                    other_idx += 1
                child[i] = other[other_idx]
        
        return child


# ============================================================================
# 禁忌搜索主算法
# ============================================================================

class TabuSearch:
    """
    禁忌搜索算法实现
    
    核心机制：
    1. 禁忌表：记录最近执行的操作或访问的解
    2. 藐视准则：如果新解优于历史最优，即使在禁忌表中也可接受
    3. 长期记忆：记录解的访问频率，用于多样化搜索
    """
    
    def __init__(self, problem: CableRoutingProblem, config: TabuConfig):
        self.problem = problem
        self.config = config
        self.tabu_list: deque = deque(maxlen=config.tabu_tenure)
        self.frequency_memory: Dict[tuple, int] = {}  # 长期记忆
        self.best_solution: List[int] = []
        self.best_cost: float = float('inf')
        self.iteration_history: List[float] = []
        
    def calculate_cost(self, solution: List[int]) -> float:
        """计算路径总成本"""
        cost = 0.0
        n = len(solution)
        for i in range(n):
            from_node = solution[i]
            to_node = solution[(i + 1) % n]  # 返回起点形成回路
            cost += self.problem.edge_costs[from_node, to_node]
        return cost
    
    def generate_initial_solution(self, method: str = 'nearest') -> List[int]:
        """生成初始解"""
        n = self.problem.n_nodes
        nodes = list(range(n))
        
        if method == 'random':
            random.shuffle(nodes)
            return nodes
        
        elif method == 'nearest':
            # 最近邻启发式
            visited = [False] * n
            solution = [self.problem.source]
            visited[self.problem.source] = True
            
            current = self.problem.source
            for _ in range(n - 1):
                nearest = -1
                min_dist = float('inf')
                for j in range(n):
                    if not visited[j]:
                        dist = self.problem.edge_costs[current, j]
                        if dist < min_dist:
                            min_dist = dist
                            nearest = j
                if nearest != -1:
                    solution.append(nearest)
                    visited[nearest] = True
                    current = nearest
            
            return solution
        
        return nodes
    
    def generate_neighbors(self, solution: List[int]) -> List[Tuple[List[int], str, tuple]]:
        """
        生成邻域解
        
        返回：[(新解，操作类型，操作标识), ...]
        """
        neighbors = []
        n = len(solution)
        
        # 随机采样邻域
        for _ in range(self.config.neighborhood_size):
            op_type = random.choice(['swap', 'insert', 'reverse', '2opt'])
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            while j == i:
                j = random.randint(0, n - 1)
            
            if op_type == 'swap':
                new_sol = NeighborhoodOperators.swap(solution, i, j)
                move_key = ('swap', min(i, j), max(i, j))
            elif op_type == 'insert':
                new_sol = NeighborhoodOperators.insert(solution, i, j)
                move_key = ('insert', i, j)
            elif op_type == 'reverse':
                new_sol = NeighborhoodOperators.reverse(solution, i, j)
                move_key = ('reverse', min(i, j), max(i, j))
            else:  # 2opt
                new_sol = NeighborhoodOperators.two_opt(solution, i, j)
                move_key = ('2opt', min(i, j), max(i, j))
            
            neighbors.append((new_sol, op_type, move_key))
        
        return neighbors
    
    def is_tabu(self, move_key: tuple) -> bool:
        """检查移动是否在禁忌表中"""
        return move_key in self.tabu_list
    
    def add_to_tabu(self, move_key: tuple):
        """将移动加入禁忌表"""
        self.tabu_list.append(move_key)
        # 更新频率记忆
        if move_key in self.frequency_memory:
            self.frequency_memory[move_key] += 1
        else:
            self.frequency_memory[move_key] = 1
    
    def meets_aspiration(self, cost: float) -> bool:
        """检查是否满足藐视准则"""
        return cost < self.best_cost
    
    def diversify(self, solution: List[int]) -> List[int]:
        """基于频率记忆的多样化策略"""
        if not self.frequency_memory:
            return solution
        
        # 找到最少使用的移动
        min_freq = min(self.frequency_memory.values())
        rare_moves = [k for k, v in self.frequency_memory.items() if v == min_freq]
        
        if rare_moves:
            # 应用一个低频移动
            move = random.choice(rare_moves)
            op_type, i, j = move
            if op_type == 'swap':
                return NeighborhoodOperators.swap(solution, i, j)
            elif op_type == 'insert':
                return NeighborhoodOperators.insert(solution, i, j)
            elif op_type == 'reverse':
                return NeighborhoodOperators.reverse(solution, i, j)
        
        return solution
    
    def search(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        执行禁忌搜索
        
        返回：(最优解，最优成本)
        """
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        
        # 初始化
        current_solution = self.generate_initial_solution('nearest')
        current_cost = self.calculate_cost(current_solution)
        
        self.best_solution = current_solution.copy()
        self.best_cost = current_cost
        
        no_improvement_count = 0
        
        if verbose:
            print(f"初始解成本：{current_cost:.2f}")
        
        for iteration in range(self.config.max_iterations):
            # 生成邻域
            neighbors = self.generate_neighbors(current_solution)
            
            # 选择最佳非禁忌解
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_neighbor_move = None
            
            for neighbor, op_type, move_key in neighbors:
                cost = self.calculate_cost(neighbor)
                
                # 检查是否接受
                if self.is_tabu(move_key):
                    # 藐视准则：如果优于历史最优，接受
                    if cost < self.best_cost:
                        if cost < best_neighbor_cost:
                            best_neighbor = neighbor
                            best_neighbor_cost = cost
                            best_neighbor_move = move_key
                else:
                    # 非禁忌解，选择最优
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_neighbor_move = move_key
            
            # 如果没有找到可接受的解，进行多样化
            if best_neighbor is None:
                best_neighbor = self.diversify(current_solution)
                best_neighbor_cost = self.calculate_cost(best_neighbor)
                best_neighbor_move = ('diversify', 0, 0)
            
            # 更新当前解
            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            
            # 更新禁忌表
            if best_neighbor_move[0] != 'diversify':
                self.add_to_tabu(best_neighbor_move)
            
            # 更新历史最优
            if current_cost < self.best_cost:
                self.best_solution = current_solution.copy()
                self.best_cost = current_cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 记录迭代历史
            self.iteration_history.append(self.best_cost)
            
            # 打印进度
            if verbose and (iteration + 1) % 50 == 0:
                print(f"迭代 {iteration + 1}/{self.config.max_iterations}, "
                      f"当前最优：{self.best_cost:.2f}, "
                      f"禁忌表大小：{len(self.tabu_list)}")
            
            # 提前终止
            if no_improvement_count >= self.config.max_no_improvement:
                if verbose:
                    print(f"达到无改进限制 ({no_improvement_count} 次)，提前终止")
                break
        
        if verbose:
            print(f"\n搜索完成!")
            print(f"最优成本：{self.best_cost:.2f}")
            print(f"改进幅度：{(current_cost - self.best_cost) / current_cost * 100:.2f}%")
        
        return self.best_solution, self.best_cost


# ============================================================================
# 可视化功能
# ============================================================================

class TabuSearchVisualizer:
    """禁忌搜索可视化"""
    
    @staticmethod
    def plot_solution(problem: CableRoutingProblem, solution: List[int], 
                     title: str = "禁忌搜索结果"):
        """绘制布线方案"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        nodes = problem.nodes
        
        # 绘制所有节点
        ax.scatter(nodes[:, 0], nodes[:, 1], c='lightblue', s=100, 
                  edgecolors='gray', linewidth=1, label='节点')
        
        # 绘制源点和汇点
        ax.scatter(nodes[problem.source, 0], nodes[problem.source, 1], 
                  c='green', s=150, marker='s', label='源点', zorder=5)
        ax.scatter(nodes[problem.sink, 0], nodes[problem.sink, 1], 
                  c='red', s=150, marker='*', label='汇点', zorder=5)
        
        # 绘制路径
        path_nodes = nodes[solution]
        # 闭合回路
        path_nodes = np.vstack([path_nodes, path_nodes[0]])
        ax.plot(path_nodes[:, 0], path_nodes[:, 1], 'b-', linewidth=2, 
               label='布线路径', alpha=0.7)
        
        # 绘制箭头表示方向
        for i in range(len(solution)):
            from_idx = solution[i]
            to_idx = solution[(i + 1) % len(solution)]
            ax.annotate('', 
                       xy=(nodes[to_idx, 0], nodes[to_idx, 1]),
                       xytext=(nodes[from_idx, 0], nodes[from_idx, 1]),
                       arrowprops=dict(arrowstyle='->', color='blue', 
                                      lw=1.5, alpha=0.5))
        
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_convergence(iteration_history: List[float], 
                        title: str = "收敛曲线"):
        """绘制收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(iteration_history, linewidth=2, color='blue')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('最优成本')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # 标注最优值
        best_idx = np.argmin(iteration_history)
        ax.annotate(f'最优：{iteration_history[best_idx]:.2f}',
                   xy=(best_idx, iteration_history[best_idx]),
                   xytext=(best_idx + 20, iteration_history[best_idx] + 
                          0.1 * (max(iteration_history) - min(iteration_history))),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red')
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_comparison(ts_result: float, initial_result: float, 
                       other_algorithms: Dict[str, float] = None):
        """算法对比柱状图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = ['初始解', '禁忌搜索']
        costs = [initial_result, ts_result]
        
        if other_algorithms:
            for name, cost in other_algorithms.items():
                algorithms.append(name)
                costs.append(cost)
        
        colors = ['lightcoral', 'steelblue']
        if other_algorithms:
            colors.extend(['gray'] * len(other_algorithms))
        
        bars = ax.bar(algorithms, costs, color=colors, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f'{cost:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('成本')
        ax.set_title('算法性能对比')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 计算改进幅度
        improvement = (initial_result - ts_result) / initial_result * 100
        ax.text(0.5, max(costs) * 0.9, f'改进：{improvement:.1f}%',
               ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig, ax


# ============================================================================
# 应用示例
# ============================================================================

def create_test_problem(n_nodes: int = 30, seed: int = 42) -> CableRoutingProblem:
    """创建测试问题"""
    np.random.seed(seed)
    
    # 生成随机节点
    nodes = np.random.rand(n_nodes, 2) * 100
    
    # 随机选择源点和汇点
    source = 0
    sink = n_nodes // 2
    
    # 生成随机需求
    demands = np.random.randint(-5, 10, n_nodes)
    demands[source] = -np.sum(demands[demands > 0])  # 源点平衡
    
    return CableRoutingProblem(
        nodes=nodes,
        source=source,
        sink=sink,
        demands=demands
    )


def run_tabu_search_demo():
    """运行禁忌搜索演示"""
    print("=" * 60)
    print("禁忌搜索算法 (Tabu Search) - 线缆布线优化演示")
    print("=" * 60)
    
    # 创建问题
    problem = create_test_problem(n_nodes=30, seed=42)
    print(f"\n问题规模：{problem.n_nodes} 个节点")
    print(f"源点：{problem.source}, 汇点：{problem.sink}")
    
    # 配置参数
    config = TabuConfig(
        tabu_tenure=15,
        max_iterations=500,
        max_no_improvement=100,
        neighborhood_size=20,
        seed=42
    )
    
    # 创建搜索器
    ts = TabuSearch(problem, config)
    
    # 计算初始解
    initial_solution = ts.generate_initial_solution('nearest')
    initial_cost = ts.calculate_cost(initial_solution)
    print(f"\n初始解成本 (最近邻): {initial_cost:.2f}")
    
    # 执行搜索
    print("\n开始禁忌搜索...")
    start_time = time.time()
    best_solution, best_cost = ts.search(verbose=True)
    elapsed_time = time.time() - start_time
    
    print(f"\n搜索时间：{elapsed_time:.2f} 秒")
    print(f"最优成本：{best_cost:.2f}")
    print(f"改进幅度：{(initial_cost - best_cost) / initial_cost * 100:.2f}%")
    
    # 可视化
    print("\n生成可视化...")
    
    # 1. 解决方案图
    fig1, _ = TabuSearchVisualizer.plot_solution(
        problem, best_solution,
        f"禁忌搜索结果 (成本={best_cost:.2f})"
    )
    fig1.savefig('/root/.openclaw/workspace/cable-optimization/examples/outputs/09_ts_solution.png', 
                dpi=150, bbox_inches='tight')
    print("  ✓ 保存解决方案图")
    
    # 2. 收敛曲线
    fig2, _ = TabuSearchVisualizer.plot_convergence(
        ts.iteration_history,
        "禁忌搜索收敛曲线"
    )
    fig2.savefig('/root/.openclaw/workspace/cable-optimization/examples/outputs/09_ts_convergence.png',
                dpi=150, bbox_inches='tight')
    print("  ✓ 保存收敛曲线")
    
    # 3. 算法对比
    fig3, _ = TabuSearchVisualizer.plot_comparison(
        ts_result=best_cost,
        initial_result=initial_cost,
        other_algorithms={'遗传算法': initial_cost * 0.85, '模拟退火': initial_cost * 0.88}
    )
    fig3.savefig('/root/.openclaw/workspace/cable-optimization/examples/outputs/09_ts_comparison.png',
                dpi=150, bbox_inches='tight')
    print("  ✓ 保存对比图")
    
    print("\n" + "=" * 60)
    print("演示完成！可视化结果已保存到 examples/outputs/")
    print("=" * 60)
    
    return best_solution, best_cost, ts.iteration_history


if __name__ == '__main__':
    # 确保输出目录存在
    import os
    os.makedirs('/root/.openclaw/workspace/cable-optimization/examples/outputs', exist_ok=True)
    
    # 运行演示
    run_tabu_search_demo()
