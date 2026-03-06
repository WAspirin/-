#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蚁群优化算法 (Ant Colony Optimization, ACO)
==========================================

灵感来源：蚂蚁觅食行为 —— 蚂蚁在路径上留下信息素，后续蚂蚁倾向于选择信息素浓度高的路径

核心机制：
1. **信息素正反馈**：好路径吸引更多蚂蚁，留下更多信息素
2. **启发式信息**：蚂蚁也考虑距离等启发式因素
3. **信息素蒸发**：避免过早收敛到局部最优

算法流程：
1. 初始化信息素矩阵
2. 每只蚂蚁构建解决方案（概率选择路径）
3. 更新信息素（蒸发 + 新沉积）
4. 重复 2-3 直到收敛

适用场景：
- 组合优化问题（TSP、VRP、调度）
- 图上的路径优化
- 需要平衡探索与利用的问题

作者：智子 (Sophon)
日期：2026-03-06
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import heapq


@dataclass
class ACOConfig:
    """ACO 算法参数配置"""
    n_ants: int = 30  # 蚂蚁数量
    n_iterations: int = 200  # 最大迭代次数
    alpha: float = 1.0  # 信息素重要程度因子
    beta: float = 2.0  # 启发式因子重要程度（距离的倒数）
    rho: float = 0.1  # 信息素蒸发率 (0-1)
    Q: float = 1.0  # 信息素强度系数
    evaporation_type: str = 'all'  # 'all' (全部蒸发) 或 'best' (仅最优蚂蚁)
    elite_ants: int = 0  # 精英蚂蚁数量（额外强化最优路径）
    random_seed: Optional[int] = None


class AntColonyOptimizer:
    """
    蚁群优化算法主类
    
    模拟蚂蚁群体通过信息素通信找到最优路径的过程
    """
    
    def __init__(self, config: ACOConfig):
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # 算法状态
        self.pheromone = None  # 信息素矩阵
        self.heuristic = None  # 启发式矩阵（距离的倒数）
        self.best_solution = None  # 历史最优解
        self.best_cost = float('inf')  # 历史最优成本
        self.convergence_history = []  # 收敛历史
        
    def init_pheromone(self, n_nodes: int, initial_value: float = 1.0):
        """初始化信息素矩阵"""
        self.pheromone = np.full((n_nodes, n_nodes), initial_value)
        # 对角线设为 0（不自环）
        np.fill_diagonal(self.pheromone, 0)
        
    def compute_heuristic(self, distance_matrix: np.ndarray):
        """计算启发式矩阵（距离的倒数）"""
        # 避免除零
        self.heuristic = np.zeros_like(distance_matrix)
        mask = distance_matrix > 0
        self.heuristic[mask] = 1.0 / distance_matrix[mask]
        
    def select_next_node(self, current: int, visited: set, 
                         available_nodes: List[int]) -> int:
        """
        概率选择下一个节点
        
        选择概率：P(i,j) = [tau(i,j)^alpha * eta(i,j)^beta] / sum(...)
        其中：tau 是信息素，eta 是启发式（1/distance）
        """
        if len(available_nodes) == 1:
            return available_nodes[0]
        
        # 计算选择概率
        probabilities = []
        for node in available_nodes:
            tau = self.pheromone[current, node] ** self.config.alpha
            eta = self.heuristic[current, node] ** self.config.beta
            probabilities.append(tau * eta)
        
        # 归一化
        total = sum(probabilities)
        if total == 0:
            # 如果所有概率为 0，随机选择
            return np.random.choice(available_nodes)
        
        probabilities = np.array(probabilities) / total
        
        # 轮盘赌选择
        return np.random.choice(available_nodes, p=probabilities)
    
    def construct_solution(self, distance_matrix: np.ndarray, 
                          start_node: int = 0) -> List[int]:
        """
        单只蚂蚁构建完整路径
        
        从起点开始，每次选择未访问节点中概率最高的，直到访问所有节点
        """
        n_nodes = len(distance_matrix)
        visited = {start_node}
        path = [start_node]
        current = start_node
        
        while len(visited) < n_nodes:
            # 获取未访问节点
            available = [i for i in range(n_nodes) if i not in visited]
            
            # 概率选择下一个节点
            next_node = self.select_next_node(current, visited, available)
            
            # 更新状态
            visited.add(next_node)
            path.append(next_node)
            current = next_node
        
        return path
    
    def compute_path_cost(self, path: List[int], 
                         distance_matrix: np.ndarray) -> float:
        """计算路径总成本（包括返回起点）"""
        cost = 0.0
        for i in range(len(path) - 1):
            cost += distance_matrix[path[i], path[i+1]]
        # 返回起点（如果是回路问题）
        cost += distance_matrix[path[-1], path[0]]
        return cost
    
    def update_pheromone(self, solutions: List[List[int]], 
                        costs: List[float], 
                        distance_matrix: np.ndarray):
        """
        更新信息素
        
        1. 蒸发：所有信息素乘以 (1 - rho)
        2. 沉积：每只蚂蚁在其路径上留下信息素 Q / cost
        """
        n_nodes = len(distance_matrix)
        
        # 1. 蒸发
        self.pheromone *= (1 - self.config.rho)
        
        # 2. 沉积
        if self.config.evaporation_type == 'all':
            # 所有蚂蚁都沉积
            for solution, cost in zip(solutions, costs):
                if cost > 0:
                    delta_pheromone = self.config.Q / cost
                    for i in range(len(solution) - 1):
                        self.pheromone[solution[i], solution[i+1]] += delta_pheromone
                    # 返回边
                    self.pheromone[solution[-1], solution[0]] += delta_pheromone
                    
        elif self.config.evaporation_type == 'best':
            # 仅最优蚂蚁沉积
            best_idx = np.argmin(costs)
            best_solution = solutions[best_idx]
            best_cost = costs[best_idx]
            
            if best_cost > 0:
                delta_pheromone = self.config.Q / best_cost
                for i in range(len(best_solution) - 1):
                    self.pheromone[best_solution[i], best_solution[best_solution[i+1]]] += delta_pheromone
                self.pheromone[best_solution[-1], best_solution[0]] += delta_pheromone
        
        # 3. 精英蚂蚁额外强化
        if self.config.elite_ants > 0 and self.best_solution is not None:
            elite_delta = self.config.elite_ants * self.config.Q / self.best_cost
            for i in range(len(self.best_solution) - 1):
                self.pheromone[self.best_solution[i], self.best_solution[i+1]] += elite_delta
            self.pheromone[self.best_solution[-1], self.best_solution[0]] += elite_delta
        
        # 限制信息素范围（避免数值问题）
        self.pheromone = np.clip(self.pheromone, 0.1, 10.0)
    
    def optimize(self, distance_matrix: np.ndarray, 
                verbose: bool = True) -> Tuple[List[int], float]:
        """
        执行 ACO 优化
        
        参数：
            distance_matrix: 距离矩阵 (n x n)
            verbose: 是否打印进度
        
        返回：
            (最优路径，最优成本)
        """
        n_nodes = len(distance_matrix)
        
        # 初始化
        self.init_pheromone(n_nodes)
        self.compute_heuristic(distance_matrix)
        self.best_solution = None
        self.best_cost = float('inf')
        self.convergence_history = []
        
        if verbose:
            print(f"🐜 开始蚁群优化")
            print(f"   节点数：{n_nodes}")
            print(f"   蚂蚁数：{self.config.n_ants}")
            print(f"   最大迭代：{self.config.n_iterations}")
            print(f"   参数：alpha={self.config.alpha}, beta={self.config.beta}, rho={self.config.rho}")
            print()
        
        for iteration in range(self.config.n_iterations):
            # 1. 每只蚂蚁构建解决方案
            solutions = []
            costs = []
            
            for _ in range(self.config.n_ants):
                # 随机起点或固定起点
                start = np.random.randint(n_nodes)
                path = self.construct_solution(distance_matrix, start)
                cost = self.compute_path_cost(path, distance_matrix)
                
                solutions.append(path)
                costs.append(cost)
                
                # 更新历史最优
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = path
            
            # 2. 更新信息素
            self.update_pheromone(solutions, costs, distance_matrix)
            
            # 3. 记录收敛历史
            self.convergence_history.append(min(costs))
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"   迭代 {iteration+1}/{self.config.n_iterations}: "
                      f"当前最优 = {self.best_cost:.2f}, "
                      f"平均成本 = {np.mean(costs):.2f}")
        
        if verbose:
            print()
            print(f"✅ 优化完成")
            print(f"   最优成本：{self.best_cost:.2f}")
            print(f"   最优路径：{self.best_solution}")
        
        return self.best_solution, self.best_cost


class CableRoutingACO:
    """
    线缆布线问题的 ACO 应用
    
    将布线问题建模为 TSP 类问题：
    - 每个设备是一个节点
    - 需要访问所有设备一次
    - 最小化总布线长度
    """
    
    def __init__(self, config: ACOConfig):
        self.optimizer = AntColonyOptimizer(config)
        self.nodes = None
        self.distance_matrix = None
        
    def setup_problem(self, nodes: np.ndarray):
        """
        设置布线问题
        
        参数：
            nodes: 节点坐标数组 (n x 2)
        """
        self.nodes = nodes
        n_nodes = len(nodes)
        
        # 计算欧氏距离矩阵
        self.distance_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                dist = np.linalg.norm(nodes[i] - nodes[j])
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist
    
    def optimize(self, verbose: bool = True) -> Tuple[List[int], float]:
        """执行优化"""
        return self.optimizer.optimize(self.distance_matrix, verbose)
    
    def get_convergence_history(self) -> List[float]:
        """获取收敛历史"""
        return self.optimizer.convergence_history


class ACOVisualizer:
    """ACO 算法可视化工具"""
    
    @staticmethod
    def plot_solution(nodes: np.ndarray, path: List[int], 
                     title: str = "ACO 最优路径", 
                     save_path: Optional[str] = None):
        """绘制最优路径"""
        plt.figure(figsize=(10, 8))
        
        # 绘制所有节点
        plt.scatter(nodes[:, 0], nodes[:, 1], c='lightblue', 
                   s=100, edgecolors='blue', linewidths=2, label='设备节点')
        
        # 绘制起点（特殊标记）
        plt.scatter(nodes[path[0], 0], nodes[path[0], 1], 
                   c='red', s=150, marker='*', label='起点')
        
        # 绘制路径
        path_nodes = nodes[path]
        # 闭合路径（返回起点）
        path_nodes = np.vstack([path_nodes, path_nodes[0]])
        plt.plot(path_nodes[:, 0], path_nodes[:, 1], 
                'b-', linewidth=2, alpha=0.6, label='布线路径')
        
        # 标注节点编号
        for i, (x, y) in enumerate(nodes):
            plt.annotate(f'{i}', (x, y), fontsize=9, 
                        ha='center', va='center')
        
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   已保存：{save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_convergence(history: List[float], 
                        title: str = "ACO 收敛曲线",
                        save_path: Optional[str] = None):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(history, 'b-', linewidth=2, label='每代最优成本')
        
        # 绘制累积最优
        best_so_far = np.minimum.accumulate(history)
        plt.plot(best_so_far, 'r--', linewidth=2, label='历史最优')
        
        plt.xlabel('迭代次数')
        plt.ylabel('路径成本')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   已保存：{save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_pheromone(pheromone: np.ndarray, 
                      nodes: np.ndarray,
                      title: str = "信息素分布",
                      save_path: Optional[str] = None):
        """绘制信息素分布热力图"""
        plt.figure(figsize=(10, 8))
        
        # 绘制信息素热力图
        plt.imshow(pheromone, cmap='hot', interpolation='nearest')
        plt.colorbar(label='信息素浓度')
        
        # 叠加节点位置
        plt.scatter(range(len(nodes)), range(len(nodes)), 
                   c='cyan', s=50, marker='x')
        
        plt.xlabel('节点索引')
        plt.ylabel('节点索引')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   已保存：{save_path}")
        
        plt.close()
    
    @staticmethod
    def compare_algorithms(histories: Dict[str, List[float]], 
                          title: str = "算法对比",
                          save_path: Optional[str] = None):
        """对比多个算法的收敛曲线"""
        plt.figure(figsize=(12, 7))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (name, history) in enumerate(histories.items()):
            color = colors[i % len(colors)]
            best_so_far = np.minimum.accumulate(history)
            plt.plot(best_so_far, color=color, linewidth=2, 
                    label=f'{name} (最优：{min(history):.2f})')
        
        plt.xlabel('迭代次数')
        plt.ylabel('路径成本')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   已保存：{save_path}")
        
        plt.close()


def generate_test_nodes(n_nodes: int, area_size: float = 100.0, 
                       seed: Optional[int] = None) -> np.ndarray:
    """生成随机测试节点"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n_nodes, 2) * area_size


def main():
    """主函数 - 演示 ACO 在线缆布线中的应用"""
    print("=" * 60)
    print("蚁群优化算法 (ACO) - 线缆布线应用")
    print("=" * 60)
    print()
    
    # 配置参数
    config = ACOConfig(
        n_ants=30,
        n_iterations=200,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        Q=1.0,
        elite_ants=5,  # 5 只精英蚂蚁
        random_seed=42
    )
    
    # 创建问题实例
    n_nodes = 30
    nodes = generate_test_nodes(n_nodes, area_size=100.0, seed=42)
    
    print(f"📍 问题规模：{n_nodes} 个设备节点")
    print(f"📐 区域大小：100 x 100")
    print()
    
    # 创建并运行优化器
    routing = CableRoutingACO(config)
    routing.setup_problem(nodes)
    
    optimal_path, optimal_cost = routing.optimize(verbose=True)
    
    print()
    print("=" * 60)
    print("📊 优化结果")
    print("=" * 60)
    print(f"最优路径成本：{optimal_cost:.2f}")
    print(f"最优路径：{optimal_path}")
    print()
    
    # 可视化
    print("🎨 生成可视化...")
    
    # 1. 最优路径图
    ACOVisualizer.plot_solution(
        nodes, optimal_path,
        title=f"ACO 最优布线路径 (成本={optimal_cost:.2f})",
        save_path="examples/outputs/10_aco_solution.png"
    )
    
    # 2. 收敛曲线
    history = routing.get_convergence_history()
    ACOVisualizer.plot_convergence(
        history,
        title="ACO 收敛曲线",
        save_path="examples/outputs/10_aco_convergence.png"
    )
    
    # 3. 信息素分布（最终状态）
    ACOVisualizer.plot_pheromone(
        routing.optimizer.pheromone,
        nodes,
        title="最终信息素分布",
        save_path="examples/outputs/10_aco_pheromone.png"
    )
    
    # 4. 参数敏感性分析
    print()
    print("=" * 60)
    print("🔬 参数敏感性分析")
    print("=" * 60)
    
    # 测试不同的 alpha 值
    alphas = [0.5, 1.0, 2.0]
    results = {}
    
    for alpha in alphas:
        config_test = ACOConfig(
            n_ants=20,
            n_iterations=100,
            alpha=alpha,
            beta=2.0,
            rho=0.1,
            random_seed=42
        )
        routing_test = CableRoutingACO(config_test)
        routing_test.setup_problem(nodes)
        _, cost = routing_test.optimize(verbose=False)
        results[f'alpha={alpha}'] = routing_test.get_convergence_history()
        print(f"   alpha={alpha}: 最优成本 = {cost:.2f}")
    
    # 绘制对比
    ACOVisualizer.compare_algorithms(
        results,
        title="不同 alpha 参数对比",
        save_path="examples/outputs/10_aco_alpha_comparison.png"
    )
    
    print()
    print("=" * 60)
    print("✅ 所有任务完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
