#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合膜算法 (Memetic Algorithm) - GA + 局部搜索

算法核心:
- 全局搜索：遗传算法 (GA) 提供探索能力
- 局部优化：变邻域搜索 (VNS) 提供开发能力
- 协同机制：每一代对精英个体进行局部搜索

应用场景：线缆布线优化问题

作者：智子 (Sophon)
日期：2026-03-11
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass
import random


# ============================================
# 配置类
# ============================================

@dataclass
class MemeticConfig:
    """膜算法配置参数"""
    # GA 参数
    population_size: int = 100
    elite_size: int = 10
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    generations: int = 200
    
    # 局部搜索参数
    local_search_probability: float = 0.3  # 个体接受局部搜索的概率
    local_search_iterations: int = 50  # 每个个体的局部搜索迭代次数
    
    # VNS 参数
    k_max: int = 5  # 最大邻域数
    
    # 其他
    random_seed: int = 42
    verbose: bool = True


# ============================================
# 问题定义：线缆布线
# ============================================

class CableRoutingProblem:
    """线缆布线问题定义"""
    
    def __init__(self, n_nodes: int = 30, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.n_nodes = n_nodes
        # 生成随机节点坐标 (0-100 范围内)
        self.nodes = np.random.rand(n_nodes, 2) * 100
        
        # 起点和终点
        self.start_node = 0
        self.end_node = n_nodes - 1
        
        # 预计算距离矩阵
        self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """计算欧氏距离矩阵"""
        diff = self.nodes[:, np.newaxis, :] - self.nodes[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    def compute_path_cost(self, path: List[int]) -> float:
        """计算路径总成本（欧氏距离）"""
        if len(path) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.distance_matrix[path[i], path[i + 1]]
        
        # 添加返回起点的成本（如果需要闭合路径）
        # cost += self.distance_matrix[path[-1], path[0]]
        
        return cost
    
    def validate_path(self, path: List[int]) -> bool:
        """验证路径是否有效"""
        if len(path) != self.n_nodes:
            return False
        if set(path) != set(range(self.n_nodes)):
            return False
        if path[0] != self.start_node:
            return False
        if path[-1] != self.end_node:
            return False
        return True
    
    def get_nearest_neighbor_solution(self) -> List[int]:
        """生成最近邻启发式初始解"""
        unvisited = set(range(self.n_nodes))
        path = [self.start_node]
        unvisited.remove(self.start_node)
        
        current = self.start_node
        while unvisited:
            # 选择最近的未访问节点
            nearest = min(unvisited, key=lambda x: self.distance_matrix[current, x])
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # 确保终点在最后
        if path[-1] != self.end_node:
            path.remove(self.end_node)
            path.append(self.end_node)
        
        return path


# ============================================
# 邻域操作（用于局部搜索）
# ============================================

class NeighborhoodOperators:
    """邻域操作算子"""
    
    @staticmethod
    def swap(path: List[int], i: int, j: int) -> List[int]:
        """交换两个位置"""
        new_path = path.copy()
        new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path
    
    @staticmethod
    def insert(path: List[int], i: int, j: int) -> List[int]:
        """插入操作：将位置 i 的元素移到位置 j"""
        new_path = path.copy()
        element = new_path.pop(i)
        new_path.insert(j, element)
        return new_path
    
    @staticmethod
    def reverse(path: List[int], i: int, j: int) -> List[int]:
        """逆转子序列"""
        new_path = path.copy()
        if i > j:
            i, j = j, i
        new_path[i:j+1] = new_path[i:j+1][::-1]
        return new_path
    
    @staticmethod
    def two_opt(path: List[int], i: int, j: int) -> List[int]:
        """2-opt 操作"""
        new_path = path.copy()
        if i > j:
            i, j = j, i
        new_path[i:j+1] = new_path[i:j+1][::-1]
        return new_path
    
    @staticmethod
    def or_opt(path: List[int], i: int, j: int, k: int) -> List[int]:
        """Or-opt: 将从 i 开始的长度为 k 的子序列移到位置 j"""
        new_path = path.copy()
        if i > j:
            # 向前移动
            segment = new_path[i:i+k]
            del new_path[i:i+k]
            # 调整插入位置
            insert_pos = j if j < i else j - k
            for idx, elem in enumerate(segment):
                new_path.insert(insert_pos + idx, elem)
        else:
            # 向后移动
            segment = new_path[i:i+k]
            del new_path[i:i+k]
            insert_pos = j if j < i else j - k + 1
            for idx, elem in enumerate(segment):
                new_path.insert(insert_pos + idx, elem)
        return new_path


# ============================================
# 变邻域搜索 (VNS) - 局部搜索
# ============================================

class VariableNeighborhoodSearch:
    """变邻域搜索局部优化器"""
    
    def __init__(self, problem: CableRoutingProblem, config: MemeticConfig):
        self.problem = problem
        self.config = config
        self.operators = NeighborhoodOperators()
    
    def generate_neighborhood(self, path: List[int], k: int) -> List[int]:
        """生成第 k 个邻域的解"""
        n = len(path)
        
        if k == 1:
            # N1: 交换相邻节点
            i = random.randint(1, n - 3)  # 避开起点和终点
            return self.operators.swap(path, i, i + 1)
        
        elif k == 2:
            # N2: 交换任意两个节点
            i = random.randint(1, n - 2)
            j = random.randint(1, n - 2)
            while j == i:
                j = random.randint(1, n - 2)
            return self.operators.swap(path, i, j)
        
        elif k == 3:
            # N3: 插入操作
            i = random.randint(1, n - 2)
            j = random.randint(1, n - 2)
            while j == i:
                j = random.randint(1, n - 2)
            return self.operators.insert(path, i, j)
        
        elif k == 4:
            # N4: 逆转操作
            i = random.randint(1, n - 3)
            j = random.randint(i + 1, n - 2)
            return self.operators.reverse(path, i, j)
        
        elif k == 5:
            # N5: 2-opt
            i = random.randint(1, n - 3)
            j = random.randint(i + 1, n - 2)
            return self.operators.two_opt(path, i, j)
        
        else:
            # 默认使用交换
            i = random.randint(1, n - 2)
            j = random.randint(1, n - 2)
            return self.operators.swap(path, i, j)
    
    def local_search(self, path: List[int], max_iterations: int = 50) -> Tuple[List[int], float]:
        """执行局部搜索"""
        current_path = path.copy()
        current_cost = self.problem.compute_path_cost(current_path)
        
        best_path = current_path.copy()
        best_cost = current_cost
        
        for iteration in range(max_iterations):
            # 遍历所有邻域
            improved = False
            
            for k in range(1, self.config.k_max + 1):
                # 扰动
                neighbor_path = self.generate_neighborhood(current_path, k)
                neighbor_cost = self.problem.compute_path_cost(neighbor_path)
                
                # 接受改进
                if neighbor_cost < current_cost:
                    current_path = neighbor_path
                    current_cost = neighbor_cost
                    improved = True
                    
                    if current_cost < best_cost:
                        best_path = current_path.copy()
                        best_cost = current_cost
                    
                    break  # 找到改进就重置 k
            
            if not improved:
                break
        
        return best_path, best_cost


# ============================================
# 遗传算法组件
# ============================================

class GeneticAlgorithm:
    """遗传算法实现"""
    
    def __init__(self, problem: CableRoutingProblem, config: MemeticConfig):
        self.problem = problem
        self.config = config
        self.population = []
        self.fitness = []
    
    def initialize_population(self, size: int) -> List[List[int]]:
        """初始化种群"""
        population = []
        
        # 添加一个最近邻解作为精英
        nearest = self.problem.get_nearest_neighbor_solution()
        population.append(nearest)
        
        # 随机生成其余个体
        for _ in range(size - 1):
            individual = list(range(self.problem.n_nodes))
            # 保持起点和终点固定
            middle_nodes = individual[1:-1]
            random.shuffle(middle_nodes)
            individual = [individual[0]] + middle_nodes + [individual[-1]]
            population.append(individual)
        
        return population
    
    def tournament_selection(self, k: int = 3) -> List[int]:
        """锦标赛选择"""
        indices = random.sample(range(len(self.population)), k)
        best_idx = min(indices, key=lambda i: self.fitness[i])
        return self.population[best_idx].copy()
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """顺序交叉 (OX)"""
        size = len(parent1)
        
        # 选择交叉点
        point1 = random.randint(1, size - 3)
        point2 = random.randint(point1 + 1, size - 2)
        
        # 创建子代
        child = [None] * size
        child[0] = parent1[0]  # 保持起点
        child[-1] = parent1[-1]  # 保持终点
        
        # 复制父代 1 的中间段
        child[point1:point2] = parent1[point1:point2]
        
        # 从父代 2 填充剩余位置
        remaining = [gene for gene in parent2 if gene not in child[point1:point2]]
        remaining = [g for g in remaining if g not in [child[0], child[-1]]]
        
        fill_pos = 0
        for gene in remaining:
            while child[fill_pos] is not None:
                fill_pos += 1
            if fill_pos == 0 or fill_pos == size - 1:
                fill_pos += 1
                continue
            child[fill_pos] = gene
            fill_pos += 1
        
        return child
    
    def swap_mutation(self, individual: List[int]) -> List[int]:
        """交换变异"""
        mutant = individual.copy()
        
        if random.random() < self.config.mutation_rate:
            # 选择两个中间位置进行交换
            i = random.randint(1, len(mutant) - 2)
            j = random.randint(1, len(mutant) - 2)
            while j == i:
                j = random.randint(1, len(mutant) - 2)
            mutant[i], mutant[j] = mutant[j], mutant[i]
        
        return mutant
    
    def evaluate_population(self):
        """评估种群适应度"""
        self.fitness = [self.problem.compute_path_cost(ind) for ind in self.population]


# ============================================
# 膜算法 (Memetic Algorithm)
# ============================================

class MemeticAlgorithm:
    """
    膜算法：遗传算法 + 局部搜索
    
    核心思想：
    1. 使用 GA 进行全局探索
    2. 对精英个体使用 VNS 进行局部开发
    3. 平衡探索与开发
    """
    
    def __init__(self, problem: CableRoutingProblem, config: MemeticConfig):
        self.problem = problem
        self.config = config
        self.ga = GeneticAlgorithm(problem, config)
        self.vns = VariableNeighborhoodSearch(problem, config)
        
        # 记录搜索历史
        self.history = {
            'best_cost': [],
            'avg_cost': [],
            'local_search_count': []
        }
    
    def optimize(self) -> Tuple[List[int], float]:
        """执行膜算法优化"""
        start_time = time.time()
        
        # 初始化种群
        self.ga.population = self.ga.initialize_population(self.config.population_size)
        self.ga.evaluate_population()
        
        global_best_path = self.ga.population[0].copy()
        global_best_cost = self.ga.fitness[0]
        
        local_search_total = 0
        
        if self.config.verbose:
            print(f"{'='*60}")
            print(f"膜算法优化开始")
            print(f"{'='*60}")
            print(f"种群大小：{self.config.population_size}")
            print(f"迭代次数：{self.config.generations}")
            print(f"局部搜索概率：{self.config.local_search_probability}")
            print(f"{'='*60}\n")
        
        for gen in range(self.config.generations):
            new_population = []
            local_search_count = 0
            
            # 精英保留
            elite_indices = np.argsort(self.ga.fitness)[:self.config.elite_size]
            for idx in elite_indices:
                elite = self.ga.population[idx].copy()
                
                # 对精英个体进行局部搜索（概率性）
                if random.random() < self.config.local_search_probability:
                    elite, elite_cost = self.vns.local_search(
                        elite, 
                        max_iterations=self.config.local_search_iterations
                    )
                    local_search_count += 1
                
                new_population.append(elite)
            
            local_search_total += local_search_count
            
            # 生成新个体
            while len(new_population) < self.config.population_size:
                # 选择
                parent1 = self.ga.tournament_selection()
                parent2 = self.ga.tournament_selection()
                
                # 交叉
                if random.random() < self.config.crossover_rate:
                    child = self.ga.order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # 变异
                child = self.ga.swap_mutation(child)
                
                new_population.append(child)
            
            # 更新种群
            self.ga.population = new_population
            self.ga.evaluate_population()
            
            # 记录历史
            best_idx = np.argmin(self.ga.fitness)
            best_cost = self.ga.fitness[best_idx]
            avg_cost = np.mean(self.ga.fitness)
            
            self.history['best_cost'].append(best_cost)
            self.history['avg_cost'].append(avg_cost)
            self.history['local_search_count'].append(local_search_count)
            
            # 更新全局最优
            if best_cost < global_best_cost:
                global_best_cost = best_cost
                global_best_path = self.ga.population[best_idx].copy()
            
            # 打印进度
            if self.config.verbose and (gen + 1) % 20 == 0:
                print(f"Generation {gen + 1}/{self.config.generations}: "
                      f"Best={best_cost:.2f}, Avg={avg_cost:.2f}, "
                      f"Local Search={local_search_count}")
        
        elapsed_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"优化完成")
            print(f"{'='*60}")
            print(f"最优成本：{global_best_cost:.2f}")
            print(f"搜索时间：{elapsed_time:.2f}s")
            print(f"局部搜索总次数：{local_search_total}")
            print(f"{'='*60}\n")
        
        return global_best_path, global_best_cost


# ============================================
# 可视化
# ============================================

class MemeticVisualizer:
    """膜算法可视化"""
    
    def __init__(self, problem: CableRoutingProblem, history: Dict):
        self.problem = problem
        self.history = history
    
    def plot_convergence(self, save_path: str = None):
        """绘制收敛曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 收敛曲线
        ax1.plot(self.history['best_cost'], 'b-', linewidth=2, label='最优成本')
        ax1.plot(self.history['avg_cost'], 'r--', linewidth=2, label='平均成本')
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('路径成本', fontsize=12)
        ax1.set_title('膜算法收敛曲线', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 局部搜索使用统计
        ax2.bar(range(len(self.history['local_search_count'])), 
                self.history['local_search_count'],
                color='green', alpha=0.6)
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('局部搜索次数', fontsize=12)
        ax2.set_title('每代局部搜索使用情况', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"收敛曲线已保存：{save_path}")
        else:
            plt.show()
    
    def plot_path(self, path: List[int], title: str = "最优路径", save_path: str = None):
        """绘制路径"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制所有节点
        ax.scatter(self.problem.nodes[:, 0], self.problem.nodes[:, 1], 
                  c='lightblue', s=100, edgecolors='blue', linewidths=2, 
                  label='中间节点', alpha=0.7)
        
        # 绘制起点和终点
        ax.scatter(self.problem.nodes[self.problem.start_node, 0],
                  self.problem.nodes[self.problem.start_node, 1],
                  c='green', s=200, marker='s', edgecolors='darkgreen',
                  linewidths=3, label='起点', zorder=5)
        
        ax.scatter(self.problem.nodes[self.problem.end_node, 0],
                  self.problem.nodes[self.problem.end_node, 1],
                  c='red', s=200, marker='s', edgecolors='darkred',
                  linewidths=3, label='终点', zorder=5)
        
        # 绘制路径
        path_nodes = self.problem.nodes[path]
        ax.plot(path_nodes[:, 0], path_nodes[:, 1], 'b-', linewidth=2, alpha=0.6)
        ax.plot(path_nodes[:, 0], path_nodes[:, 1], 'bo-', markersize=8, alpha=0.6)
        
        # 标注节点编号
        for i, (x, y) in enumerate(self.problem.nodes):
            ax.annotate(str(i), (x, y), fontsize=8, alpha=0.7)
        
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        ax.set_title(f'{title}\n成本：{self.problem.compute_path_cost(path):.2f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"路径图已保存：{save_path}")
        else:
            plt.show()
    
    def plot_comparison(self, paths: Dict[str, List[int]], save_path: str = None):
        """对比不同算法的路径"""
        fig, axes = plt.subplots(1, len(paths), figsize=(6 * len(paths), 5))
        if len(paths) == 1:
            axes = [axes]
        
        for ax, (name, path) in zip(axes, paths.items()):
            cost = self.problem.compute_path_cost(path)
            
            # 绘制节点
            ax.scatter(self.problem.nodes[:, 0], self.problem.nodes[:, 1],
                      c='lightblue', s=80, edgecolors='blue', alpha=0.5)
            ax.scatter(self.problem.nodes[self.problem.start_node, 0],
                      self.problem.nodes[self.problem.start_node, 1],
                      c='green', s=150, marker='s', zorder=5)
            ax.scatter(self.problem.nodes[self.problem.end_node, 0],
                      self.problem.nodes[self.problem.end_node, 1],
                      c='red', s=150, marker='s', zorder=5)
            
            # 绘制路径
            path_nodes = self.problem.nodes[path]
            ax.plot(path_nodes[:, 0], path_nodes[:, 1], 'b-', linewidth=2, alpha=0.6)
            
            ax.set_title(f'{name}\n成本：{cost:.2f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"对比图已保存：{save_path}")
        else:
            plt.show()


# ============================================
# 主程序
# ============================================

def main():
    """主函数：演示膜算法在线缆布线中的应用"""
    
    print("\n" + "="*70)
    print(" " * 20 + "膜算法 (Memetic Algorithm) 演示")
    print(" " * 25 + "GA + 变邻域搜索")
    print("="*70 + "\n")
    
    # 创建问题实例
    problem = CableRoutingProblem(n_nodes=30, seed=42)
    print(f"问题规模：{problem.n_nodes} 个节点")
    print(f"坐标范围：0-100\n")
    
    # 配置算法
    config = MemeticConfig(
        population_size=100,
        elite_size=15,
        crossover_rate=0.85,
        mutation_rate=0.15,
        generations=200,
        local_search_probability=0.4,
        local_search_iterations=50,
        k_max=5,
        verbose=True
    )
    
    # 创建并运行膜算法
    ma = MemeticAlgorithm(problem, config)
    best_path, best_cost = ma.optimize()
    
    # 获取基准解
    nearest_path = problem.get_nearest_neighbor_solution()
    nearest_cost = problem.compute_path_cost(nearest_path)
    
    # 计算改进
    improvement = (nearest_cost - best_cost) / nearest_cost * 100
    
    print(f"\n{'='*70}")
    print("结果对比")
    print(f"{'='*70}")
    print(f"最近邻启发式成本：{nearest_cost:.2f}")
    print(f"膜算法优化成本：  {best_cost:.2f}")
    print(f"改进幅度：{improvement:.2f}%")
    print(f"{'='*70}\n")
    
    # 可视化
    visualizer = MemeticVisualizer(problem, ma.history)
    
    # 保存可视化结果
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer.plot_convergence(save_path=os.path.join(output_dir, 'memetic_convergence.png'))
    visualizer.plot_path(best_path, title="膜算法最优路径", 
                        save_path=os.path.join(output_dir, 'memetic_best_path.png'))
    visualizer.plot_comparison(
        {'最近邻': nearest_path, '膜算法': best_path},
        save_path=os.path.join(output_dir, 'memetic_comparison.png')
    )
    
    # 验证路径
    assert problem.validate_path(best_path), "路径验证失败!"
    print("✅ 路径验证通过")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70 + "\n")
    
    return best_path, best_cost


if __name__ == "__main__":
    main()
