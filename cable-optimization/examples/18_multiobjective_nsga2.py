#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标优化 - NSGA-II 算法实现
Multi-Objective Optimization using NSGA-II

作者：智子 (Sophon)
日期：2026-03-12
主题：Week 3 Day 12 - 多目标优化

线缆布线中的多目标问题：
- 目标 1：最小化布线成本（线缆长度）
- 目标 2：最大化可靠性（冗余路径、避开高风险区域）

NSGA-II 核心机制：
1. 快速非支配排序 (Fast Non-dominated Sorting)
2. 拥挤度距离 (Crowding Distance)
3. 精英保留策略 (Elitism)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class NSGA2Config:
    """NSGA-II 算法配置参数"""
    population_size: int = 100  # 种群大小
    max_generations: int = 200  # 最大迭代次数
    crossover_rate: float = 0.9  # 交叉概率
    mutation_rate: float = 0.1  # 变异概率
    tournament_size: int = 3  # 锦标赛大小
    elite_ratio: float = 0.2  # 精英保留比例
    
    # 问题参数
    num_nodes: int = 20  # 节点数量
    grid_size: int = 100  # 网格大小
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)


# ============================================================================
# 问题定义
# ============================================================================

@dataclass
class Node:
    """布线节点"""
    id: int
    x: float
    y: float
    risk: float = 0.0  # 风险值 (0-1)
    is_start: bool = False
    is_end: bool = False


@dataclass
class Edge:
    """边"""
    from_node: int
    to_node: int
    length: float
    risk: float  # 边的风险（两端点风险平均值）


class MultiObjectiveCableRouting:
    """
    多目标线缆布线问题
    
    目标 1: 最小化总长度 (成本)
    目标 2: 最小化总风险 (最大化可靠性)
    """
    
    def __init__(self, config: NSGA2Config):
        self.config = config
        self.nodes: List[Node] = []
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self.distance_matrix: np.ndarray = None
        self.risk_matrix: np.ndarray = None
        
        self._generate_problem()
    
    def _generate_problem(self):
        """生成布线问题实例"""
        np.random.seed(self.config.seed)
        
        # 生成随机节点
        for i in range(self.config.num_nodes):
            node = Node(
                id=i,
                x=np.random.uniform(10, self.config.grid_size - 10),
                y=np.random.uniform(10, self.config.grid_size - 10),
                risk=np.random.uniform(0, 1)  # 随机风险值
            )
            self.nodes.append(node)
        
        # 设置起点和终点
        self.nodes[0].is_start = True
        self.nodes[-1].is_end = True
        
        # 计算距离矩阵和风险矩阵
        n = len(self.nodes)
        self.distance_matrix = np.zeros((n, n))
        self.risk_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt((self.nodes[i].x - self.nodes[j].x)**2 + 
                                   (self.nodes[i].y - self.nodes[j].y)**2)
                    risk = (self.nodes[i].risk + self.nodes[j].risk) / 2
                    self.distance_matrix[i, j] = dist
                    self.risk_matrix[i, j] = risk
                    
                    # 存储边信息
                    self.edges[(i, j)] = Edge(
                        from_node=i,
                        to_node=j,
                        length=dist,
                        risk=risk
                    )
    
    def evaluate_solution(self, path: List[int]) -> Tuple[float, float]:
        """
        评估解的两个目标
        
        Args:
            path: 节点索引列表，表示布线路径
            
        Returns:
            (total_length, total_risk): 两个目标值
        """
        if len(path) < 2:
            return float('inf'), float('inf')
        
        total_length = 0.0
        total_risk = 0.0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            total_length += self.distance_matrix[from_node, to_node]
            total_risk += self.risk_matrix[from_node, to_node]
        
        return total_length, total_risk
    
    def is_valid_path(self, path: List[int]) -> bool:
        """检查路径是否有效"""
        if len(path) < 2:
            return False
        if path[0] != 0:  # 必须从起点开始
            return False
        if path[-1] != len(self.nodes) - 1:  # 必须到终点结束
            return False
        if len(path) != len(set(path)):  # 不能有重复节点
            return False
        return True
    
    def generate_random_path(self) -> List[int]:
        """生成随机有效路径"""
        intermediate_nodes = list(range(1, len(self.nodes) - 1))
        random.shuffle(intermediate_nodes)
        
        # 随机选择一部分中间节点
        num_selected = random.randint(1, len(intermediate_nodes))
        selected = intermediate_nodes[:num_selected]
        
        path = [0] + selected + [len(self.nodes) - 1]
        return path
    
    def get_nearest_neighbor_path(self) -> List[int]:
        """使用最近邻启发式生成初始解"""
        visited = [False] * len(self.nodes)
        path = [0]  # 从起点开始
        visited[0] = True
        
        current = 0
        while not visited[len(self.nodes) - 1]:  # 直到访问终点
            best_next = -1
            best_dist = float('inf')
            
            for j in range(len(self.nodes)):
                if not visited[j]:
                    dist = self.distance_matrix[current, j]
                    # 考虑风险和距离的加权
                    combined = dist * (1 + self.nodes[j].risk)
                    if combined < best_dist:
                        best_dist = combined
                        best_next = j
            
            if best_next != -1:
                path.append(best_next)
                visited[best_next] = True
                current = best_next
            else:
                # 如果没有未访问节点，直接到终点
                if not visited[len(self.nodes) - 1]:
                    path.append(len(self.nodes) - 1)
                break
        
        return path


# ============================================================================
# NSGA-II 算法
# ============================================================================

class Individual:
    """个体（解）"""
    
    def __init__(self, path: List[int]):
        self.path = path
        self.objectives: Tuple[float, float] = (float('inf'), float('inf'))
        self.rank: int = 0  # 非支配排序的层级
        self.crowding_distance: float = 0.0  # 拥挤度距离
        self.domination_count: int = 0  # 支配该个体的个体数
        self.dominated_solutions: List['Individual'] = []  # 被该个体支配的解


class NSGA2Optimizer:
    """NSGA-II 多目标优化器"""
    
    def __init__(self, problem: MultiObjectiveCableRouting, config: NSGA2Config):
        self.problem = problem
        self.config = config
        self.population: List[Individual] = []
        self.pareto_front: List[Individual] = []
        
        # 历史记录
        self.history = {
            'generations': [],
            'pareto_size': [],
            'avg_objective1': [],
            'avg_objective2': []
        }
    
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        
        # 添加一个启发式解（最近邻）
        heuristic_path = self.problem.get_nearest_neighbor_path()
        self.population.append(Individual(heuristic_path))
        
        # 其余随机生成
        for _ in range(self.config.population_size - 1):
            path = self.problem.generate_random_path()
            self.population.append(Individual(path))
        
        # 评估所有个体
        self._evaluate_population()
    
    def _evaluate_population(self):
        """评估种群中所有个体"""
        for ind in self.population:
            ind.objectives = self.problem.evaluate_solution(ind.path)
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """
        判断 ind1 是否支配 ind2
        
        ind1 支配 ind2 当且仅当：
        1. ind1 的所有目标都不差于 ind2
        2. ind1 至少有一个目标严格优于 ind2
        """
        not_worse = all(o1 <= o2 for o1, o2 in zip(ind1.objectives, ind2.objectives))
        strictly_better = any(o1 < o2 for o1, o2 in zip(ind1.objectives, ind2.objectives))
        return not_worse and strictly_better
    
    def _fast_non_dominated_sort(self):
        """快速非支配排序"""
        # 初始化
        for ind in self.population:
            ind.domination_count = 0
            ind.dominated_solutions = []
            ind.rank = 0
        
        # 计算支配关系
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                ind_i = self.population[i]
                ind_j = self.population[j]
                
                if self._dominates(ind_i, ind_j):
                    ind_i.dominated_solutions.append(ind_j)
                    ind_j.domination_count += 1
                elif self._dominates(ind_j, ind_i):
                    ind_j.dominated_solutions.append(ind_i)
                    ind_i.domination_count += 1
        
        # 分层
        fronts: List[List[Individual]] = [[]]
        for ind in self.population:
            if ind.domination_count == 0:
                ind.rank = 0
                fronts[0].append(ind)
        
        # 继续分层
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for ind in fronts[current_front]:
                for dominated in ind.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = current_front + 1
                        next_front.append(dominated)
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def _calculate_crowding_distance(self, front: List[Individual]):
        """计算拥挤度距离"""
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        # 初始化
        for ind in front:
            ind.crowding_distance = 0.0
        
        # 对每个目标计算
        num_objectives = 2
        for m in range(num_objectives):
            # 按目标 m 排序
            sorted_front = sorted(front, key=lambda ind: ind.objectives[m])
            
            # 边界点设为无穷大
            sorted_front[0].crowding_distance = float('inf')
            sorted_front[-1].crowding_distance = float('inf')
            
            # 计算中间点的拥挤度
            obj_min = sorted_front[0].objectives[m]
            obj_max = sorted_front[-1].objectives[m]
            
            if obj_max - obj_min == 0:
                continue
            
            for i in range(1, len(sorted_front) - 1):
                sorted_front[i].crowding_distance += (
                    (sorted_front[i + 1].objectives[m] - sorted_front[i - 1].objectives[m]) /
                    (obj_max - obj_min)
                )
    
    def _tournament_selection(self, front: List[Individual]) -> Individual:
        """锦标赛选择"""
        candidates = random.sample(front, min(self.config.tournament_size, len(front)))
        
        # 选择等级最好的，等级相同时选择拥挤度最大的
        best = max(candidates, key=lambda ind: (-ind.rank, ind.crowding_distance))
        return best
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        顺序交叉 (Order Crossover, OX)
        保持路径的合法性
        """
        path1 = parent1.path
        path2 = parent2.path
        
        if random.random() > self.config.crossover_rate:
            return Individual(path1.copy())
        
        # 确保两个路径长度相同（填充到相同长度）
        min_len = min(len(path1), len(path2))
        path1 = path1[:min_len]
        path2 = path2[:min_len]
        
        # 选择交叉区域
        start = random.randint(0, min_len - 2)
        end = random.randint(start + 1, min_len - 1)
        
        # 子代 1：从 parent1 继承交叉区域，其余从 parent2 按顺序填充
        child_path = [None] * min_len
        child_path[start:end + 1] = path1[start:end + 1]
        
        # 从 parent2 填充剩余位置
        remaining = [node for node in path2 if node not in child_path[start:end + 1]]
        fill_idx = 0
        for i in range(min_len):
            if child_path[i] is None:
                if fill_idx < len(remaining):
                    child_path[i] = remaining[fill_idx]
                    fill_idx += 1
        
        # 确保起点和终点正确
        if child_path[0] != 0:
            child_path[0] = 0
        if child_path[-1] != len(self.problem.nodes) - 1:
            child_path[-1] = len(self.problem.nodes) - 1
        
        # 移除重复（除了起点终点）
        seen = set()
        unique_path = []
        for node in child_path:
            if node not in seen or node in [0, len(self.problem.nodes) - 1]:
                unique_path.append(node)
                if node not in [0, len(self.problem.nodes) - 1]:
                    seen.add(node)
        
        return Individual(unique_path)
    
    def _mutate(self, individual: Individual) -> Individual:
        """
        变异操作：交换两个中间节点
        """
        path = individual.path.copy()
        
        if random.random() > self.config.mutation_rate:
            return Individual(path)
        
        # 只能变异中间节点（不包括起点和终点）
        if len(path) <= 3:
            return Individual(path)
        
        # 随机选择两个位置交换
        idx1 = random.randint(1, len(path) - 2)
        idx2 = random.randint(1, len(path) - 2)
        
        path[idx1], path[idx2] = path[idx2], path[idx1]
        return Individual(path)
    
    def _create_child(self, parent1: Individual, parent2: Individual) -> Individual:
        """创建子代"""
        child = self._crossover(parent1, parent2)
        child = self._mutate(child)
        return child
    
    def optimize(self, verbose: bool = True) -> List[Individual]:
        """
        运行 NSGA-II 优化
        
        Returns:
            Pareto 前沿的个体列表
        """
        self.initialize_population()
        
        for gen in range(self.config.max_generations):
            # 快速非支配排序
            fronts = self._fast_non_dominated_sort()
            
            # 计算拥挤度
            for front in fronts:
                self._calculate_crowding_distance(front)
            
            # 记录历史
            if gen % 10 == 0 or gen == self.config.max_generations - 1:
                pareto_size = len(fronts[0]) if fronts else 0
                avg_obj1 = np.mean([ind.objectives[0] for ind in self.population])
                avg_obj2 = np.mean([ind.objectives[1] for ind in self.population])
                self.history['generations'].append(gen)
                self.history['pareto_size'].append(pareto_size)
                self.history['avg_objective1'].append(avg_obj1)
                self.history['avg_objective2'].append(avg_obj2)
                
                if verbose:
                    print(f"Generation {gen}: Pareto 大小={pareto_size}, "
                          f"平均长度={avg_obj1:.2f}, 平均风险={avg_obj2:.2f}")
            
            # 创建子代种群
            offspring = []
            while len(offspring) < self.config.population_size:
                parent1 = self._tournament_selection(fronts[0])
                parent2 = self._tournament_selection(fronts[0])
                child = self._create_child(parent1, parent2)
                offspring.append(child)
            
            # 评估子代
            for ind in offspring:
                ind.objectives = self.problem.evaluate_solution(ind.path)
            
            # 合并父代和子代
            combined = self.population + offspring
            
            # 精英保留：从合并种群中选择最好的 N 个
            fronts = self._fast_non_dominated_sort()
            
            new_population = []
            front_idx = 0
            
            while len(new_population) + len(fronts[front_idx]) <= self.config.population_size:
                for ind in fronts[front_idx]:
                    new_population.append(ind)
                front_idx += 1
                if front_idx >= len(fronts):
                    break
            
            # 如果还没满，从当前层按拥挤度选择
            if len(new_population) < self.config.population_size and front_idx < len(fronts):
                last_front = fronts[front_idx]
                self._calculate_crowding_distance(last_front)
                last_front.sort(key=lambda ind: (-ind.rank, -ind.crowding_distance))
                
                remaining = self.config.population_size - len(new_population)
                new_population.extend(last_front[:remaining])
            
            self.population = new_population
        
        # 获取最终 Pareto 前沿
        fronts = self._fast_non_dominated_sort()
        self.pareto_front = fronts[0] if fronts else []
        
        return self.pareto_front


# ============================================================================
# 可视化
# ============================================================================

class NSGA2Visualizer:
    """NSGA-II 可视化"""
    
    def __init__(self, problem: MultiObjectiveCableRouting, optimizer: NSGA2Optimizer):
        self.problem = problem
        self.optimizer = optimizer
    
    def plot_pareto_front(self, save_path: str = None):
        """绘制 Pareto 前沿"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制所有个体的目标
        all_obj1 = [ind.objectives[0] for ind in self.optimizer.population]
        all_obj2 = [ind.objectives[1] for ind in self.optimizer.population]
        ax.scatter(all_obj1, all_obj2, alpha=0.3, c='gray', label='所有解', s=30)
        
        # 绘制 Pareto 前沿
        if self.optimizer.pareto_front:
            pareto_obj1 = [ind.objectives[0] for ind in self.optimizer.pareto_front]
            pareto_obj2 = [ind.objectives[1] for ind in self.optimizer.pareto_front]
            
            # 按目标 1 排序
            sorted_indices = np.argsort(pareto_obj1)
            pareto_obj1 = [pareto_obj1[i] for i in sorted_indices]
            pareto_obj2 = [pareto_obj2[i] for i in sorted_indices]
            
            ax.scatter(pareto_obj1, pareto_obj2, c='red', s=100, 
                      label='Pareto 前沿', marker='o', edgecolors='black', linewidth=1.5)
            ax.plot(pareto_obj1, pareto_obj2, 'r--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('总长度 (成本)', fontsize=12)
        ax.set_ylabel('总风险 (可靠性)', fontsize=12)
        ax.set_title('NSGA-II Pareto 前沿\n(成本 vs 可靠性)', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Pareto 前沿图已保存：{save_path}")
        plt.show()
    
    def plot_convergence(self, save_path: str = None):
        """绘制收敛曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Pareto 前沿大小变化
        axes[0].plot(self.optimizer.history['generations'],
                    self.optimizer.history['pareto_size'], 'b-o', linewidth=2)
        axes[0].set_xlabel('迭代次数')
        axes[0].set_ylabel('Pareto 前沿大小')
        axes[0].set_title('Pareto 前沿演化')
        axes[0].grid(True, alpha=0.3)
        
        # 平均目标 1（长度）
        axes[1].plot(self.optimizer.history['generations'],
                    self.optimizer.history['avg_objective1'], 'g-o', linewidth=2)
        axes[1].set_xlabel('迭代次数')
        axes[1].set_ylabel('平均总长度')
        axes[1].set_title('平均成本收敛')
        axes[1].grid(True, alpha=0.3)
        
        # 平均目标 2（风险）
        axes[2].plot(self.optimizer.history['generations'],
                    self.optimizer.history['avg_objective2'], 'r-o', linewidth=2)
        axes[2].set_xlabel('迭代次数')
        axes[2].set_ylabel('平均总风险')
        axes[2].set_title('平均风险收敛')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"收敛曲线图已保存：{save_path}")
        plt.show()
    
    def plot_solution_paths(self, solution_indices: List[int] = None, save_path: str = None):
        """绘制选定的解路径"""
        if not self.optimizer.pareto_front:
            print("没有 Pareto 前沿解")
            return
        
        if solution_indices is None:
            # 默认选择 3 个代表性解：最小长度、最小风险、折中
            if len(self.optimizer.pareto_front) >= 3:
                sorted_by_length = sorted(self.optimizer.pareto_front, 
                                         key=lambda ind: ind.objectives[0])
                sorted_by_risk = sorted(self.optimizer.pareto_front,
                                       key=lambda ind: ind.objectives[1])
                
                # 找到折中解（距离两个极端都较远）
                mid_idx = len(self.optimizer.pareto_front) // 2
                solution_indices = [
                    self.optimizer.pareto_front.index(sorted_by_length[0]),
                    self.optimizer.pareto_front.index(sorted_by_risk[0]),
                    mid_idx
                ]
            else:
                solution_indices = list(range(len(self.optimizer.pareto_front)))
        
        fig, axes = plt.subplots(1, len(solution_indices), figsize=(5 * len(solution_indices), 5))
        if len(solution_indices) == 1:
            axes = [axes]
        
        titles = ['最小成本解', '最小风险解', '折中解']
        
        for idx, ax in zip(solution_indices, axes):
            if idx >= len(self.optimizer.pareto_front):
                continue
            
            ind = self.optimizer.pareto_front[idx]
            path = ind.path
            
            # 绘制节点
            for node in self.problem.nodes:
                if node.is_start:
                    ax.plot(node.x, node.y, 'go', markersize=15, label='起点', zorder=5)
                elif node.is_end:
                    ax.plot(node.x, node.y, 'ro', markersize=15, label='终点', zorder=5)
                else:
                    # 根据风险值着色
                    color = plt.cm.Reds(node.risk)
                    ax.plot(node.x, node.y, 'o', color=color, markersize=10, alpha=0.6)
            
            # 绘制路径
            path_x = [self.problem.nodes[i].x for i in path]
            path_y = [self.problem.nodes[i].y for i in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
            ax.plot(path_x, path_y, 'bo-', markersize=6, linewidth=2)
            
            # 标注目标值
            ax.set_title(f'{titles[solution_indices.index(idx)] if idx < len(titles) else "解"}\n'
                        f'长度={ind.objectives[0]:.1f}, 风险={ind.objectives[1]:.2f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, self.problem.config.grid_size)
            ax.set_ylim(0, self.problem.config.grid_size)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"路径图已保存：{save_path}")
        plt.show()
    
    def plot_risk_heatmap(self, save_path: str = None):
        """绘制风险热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建风险热力图背景
        x = np.linspace(0, self.problem.config.grid_size, 50)
        y = np.linspace(0, self.problem.config.grid_size, 50)
        X, Y = np.meshgrid(x, y)
        
        # 插值风险场
        Z = np.zeros_like(X)
        for node in self.problem.nodes:
            dist = np.sqrt((X - node.x)**2 + (Y - node.y)**2)
            influence = np.exp(-dist / 20) * node.risk
            Z += influence
        
        # 归一化
        Z = Z / (Z.max() + 1e-10)
        
        # 绘制热力图
        contour = ax.contourf(X, Y, Z, levels=20, cmap='Reds', alpha=0.6)
        plt.colorbar(contour, ax=ax, label='风险强度')
        
        # 绘制节点
        for node in self.problem.nodes:
            if node.is_start:
                ax.plot(node.x, node.y, 'go', markersize=15, label='起点', zorder=5)
            elif node.is_end:
                ax.plot(node.x, node.y, 'ro', markersize=15, label='终点', zorder=5)
            else:
                ax.plot(node.x, node.y, 'k.', markersize=8, alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('风险热力图\n(红色=高风险区域)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"风险热力图已保存：{save_path}")
        plt.show()


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("NSGA-II 多目标线缆布线优化")
    print("=" * 70)
    
    # 创建配置
    config = NSGA2Config(
        population_size=100,
        max_generations=200,
        crossover_rate=0.9,
        mutation_rate=0.15,
        num_nodes=20,
        seed=42
    )
    
    print(f"\n配置参数:")
    print(f"  种群大小：{config.population_size}")
    print(f"  最大迭代：{config.max_generations}")
    print(f"  节点数量：{config.num_nodes}")
    print(f"  交叉概率：{config.crossover_rate}")
    print(f"  变异概率：{config.mutation_rate}")
    
    # 创建问题
    print("\n生成布线问题...")
    problem = MultiObjectiveCableRouting(config)
    print(f"  节点数：{len(problem.nodes)}")
    print(f"  起点：({problem.nodes[0].x:.1f}, {problem.nodes[0].y:.1f})")
    print(f"  终点：({problem.nodes[-1].x:.1f}, {problem.nodes[-1].y:.1f})")
    
    # 创建优化器
    print("\n初始化 NSGA-II 优化器...")
    optimizer = NSGA2Optimizer(problem, config)
    
    # 运行优化
    print("\n开始优化...")
    print("-" * 70)
    pareto_front = optimizer.optimize(verbose=True)
    print("-" * 70)
    
    # 输出结果
    print(f"\n优化完成!")
    print(f"Pareto 前沿大小：{len(pareto_front)}")
    
    if pareto_front:
        print("\nPareto 前沿解:")
        print(f"{'编号':<6} {'总长度':<12} {'总风险':<12} {'路径长度':<10}")
        print("-" * 50)
        
        sorted_pareto = sorted(pareto_front, key=lambda ind: ind.objectives[0])
        for i, ind in enumerate(sorted_pareto[:10]):  # 只显示前 10 个
            print(f"{i+1:<6} {ind.objectives[0]:<12.2f} {ind.objectives[1]:<12.2f} {len(ind.path):<10}")
        
        # 极端解
        min_length = min(pareto_front, key=lambda ind: ind.objectives[0])
        min_risk = min(pareto_front, key=lambda ind: ind.objectives[1])
        
        print(f"\n极端解对比:")
        print(f"  最小长度解：长度={min_length.objectives[0]:.2f}, 风险={min_length.objectives[1]:.2f}")
        print(f"  最小风险解：长度={min_risk.objectives[0]:.2f}, 风险={min_risk.objectives[1]:.2f}")
        
        # 计算 trade-off
        length_diff = abs(min_risk.objectives[0] - min_length.objectives[0])
        risk_diff = abs(min_risk.objectives[1] - min_length.objectives[1])
        if risk_diff > 0:
            tradeoff = length_diff / risk_diff
            print(f"\nTrade-off 分析:")
            print(f"  为减少 1 单位风险，需要增加 {tradeoff:.2f} 单位长度")
    
    # 可视化
    print("\n生成可视化...")
    visualizer = NSGA2Visualizer(problem, optimizer)
    
    # 创建输出目录
    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图表
    visualizer.plot_pareto_front(save_path=os.path.join(output_dir, 'nsga2_pareto_front.png'))
    visualizer.plot_convergence(save_path=os.path.join(output_dir, 'nsga2_convergence.png'))
    visualizer.plot_solution_paths(save_path=os.path.join(output_dir, 'nsga2_paths.png'))
    visualizer.plot_risk_heatmap(save_path=os.path.join(output_dir, 'nsga2_risk_heatmap.png'))
    
    print("\n" + "=" * 70)
    print("NSGA-II 优化完成!")
    print("=" * 70)
    
    return problem, optimizer, visualizer


if __name__ == "__main__":
    main()
