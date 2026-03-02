"""
启发式算法 - 遗传算法 (Genetic Algorithm)

问题描述:
使用遗传算法求解路径优化问题（类似 TSP 的布线问题）

算法原理:
1. 编码：将解编码为染色体（路径序列）
2. 初始化：随机生成初始种群
3. 评估：计算每个个体的适应度（路径总长度）
4. 选择：轮盘赌/锦标赛选择优秀个体
5. 交叉：部分映射交叉 (PMX) 或顺序交叉 (OX)
6. 变异：交换变异或逆转变异
7. 重复 3-6 直到满足终止条件
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple


class GeneticAlgorithm:
    """遗传算法求解器"""
    
    def __init__(self, 
                 population_size: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 generations: int = 200,
                 elitism_count: int = 5):
        self.pop_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.elitism_count = elitism_count
        
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
    
    def create_individual(self, n_cities: int) -> List[int]:
        """创建随机个体（路径）"""
        individual = list(range(n_cities))
        random.shuffle(individual)
        return individual
    
    def create_population(self, n_cities: int) -> List[List[int]]:
        """创建初始种群"""
        return [self.create_individual(n_cities) for _ in range(self.pop_size)]
    
    def calculate_fitness(self, individual: List[int], distance_matrix: np.ndarray) -> float:
        """计算个体适应度（路径总长度）"""
        total_distance = 0
        for i in range(len(individual) - 1):
            total_distance += distance_matrix[individual[i], individual[i+1]]
        # 返回起点形成回路
        total_distance += distance_matrix[individual[-1], individual[0]]
        return total_distance
    
    def tournament_selection(self, population: List[List[int]], 
                            distance_matrix: np.ndarray) -> List[int]:
        """锦标赛选择"""
        tournament_size = 5
        participants = random.sample(population, tournament_size)
        
        # 选择适应度最好的
        best = min(participants, 
                  key=lambda ind: self.calculate_fitness(ind, distance_matrix))
        return best.copy()
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """顺序交叉 (OX)"""
        size = len(parent1)
        child = [-1] * size
        
        # 随机选择交叉区域
        start, end = sorted(random.sample(range(size), 2))
        
        # 复制父本 1 的交叉区域
        child[start:end+1] = parent1[start:end+1]
        
        # 填充父本 2 的剩余基因
        child_pos = end + 1
        parent2_pos = end + 1
        
        while -1 in child:
            gene = parent2[parent2_pos % size]
            if gene not in child:
                child[child_pos % size] = gene
                child_pos += 1
            parent2_pos += 1
        
        return child
    
    def swap_mutation(self, individual: List[int]) -> List[int]:
        """交换变异"""
        mutant = individual.copy()
        idx1, idx2 = random.sample(range(len(mutant)), 2)
        mutant[idx1], mutant[idx2] = mutant[idx2], mutant[idx1]
        return mutant
    
    def evolve(self, population: List[List[int]], 
               distance_matrix: np.ndarray) -> List[List[int]]:
        """进化一代"""
        new_population = []
        
        # 精英保留
        population.sort(key=lambda ind: self.calculate_fitness(ind, distance_matrix))
        new_population.extend(population[:self.elitism_count])
        
        # 生成剩余个体
        while len(new_population) < self.pop_size:
            # 选择
            parent1 = self.tournament_selection(population, distance_matrix)
            parent2 = self.tournament_selection(population, distance_matrix)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child = self.order_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # 变异
            if random.random() < self.mutation_rate:
                child = self.swap_mutation(child)
            
            new_population.append(child)
        
        return new_population
    
    def solve(self, distance_matrix: np.ndarray, verbose: bool = True) -> Tuple[List[int], float]:
        """
        求解 TSP 问题
        
        参数:
            distance_matrix: 距离矩阵
            verbose: 是否打印进度
        
        返回:
            best_solution: 最优路径
            best_fitness: 最优适应度
        """
        n_cities = distance_matrix.shape[0]
        
        # 初始化种群
        population = self.create_population(n_cities)
        
        if verbose:
            print(f"开始进化...")
            print(f"  种群大小：{self.pop_size}")
            print(f"  代数：{self.generations}")
            print(f"  交叉率：{self.crossover_rate}")
            print(f"  变异率：{self.mutation_rate}")
            print()
        
        for generation in range(self.generations):
            # 评估当前种群
            fitnesses = [self.calculate_fitness(ind, distance_matrix) 
                        for ind in population]
            
            # 更新全局最优
            min_fitness = min(fitnesses)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_solution = population[fitnesses.index(min_fitness)].copy()
            
            self.history.append(self.best_fitness)
            
            # 进化
            population = self.evolve(population, distance_matrix)
            
            # 打印进度
            if verbose and (generation + 1) % 20 == 0:
                avg_fitness = np.mean(fitnesses)
                print(f"  第 {generation+1}/{self.generations} 代: "
                      f"最优 = {self.best_fitness:.2f}, 平均 = {avg_fitness:.2f}")
        
        if verbose:
            print(f"\n✓ 进化完成！最优解适应度：{self.best_fitness:.2f}")
        
        return self.best_solution, self.best_fitness


def create_cable_network_matrix(n_nodes: int = 10, seed: int = 42) -> np.ndarray:
    """创建电缆网络距离矩阵"""
    np.random.seed(seed)
    
    # 随机生成节点坐标
    coords = np.random.rand(n_nodes, 2) * 100
    
    # 计算欧氏距离矩阵
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                distance_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
    
    return distance_matrix, coords


def visualize_ga_result(solution: List[int], 
                       distance_matrix: np.ndarray,
                       coords: np.ndarray,
                       history: List[float]):
    """可视化遗传算法结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：最优路径
    ax1.scatter(coords[:, 0], coords[:, 1], c='lightblue', s=200, edgecolors='blue')
    
    # 绘制路径
    for i in range(len(solution)):
        start = solution[i]
        end = solution[(i + 1) % len(solution)]
        ax1.plot([coords[start, 0], coords[end, 0]], 
                [coords[start, 1], coords[end, 1]], 
                'r-', linewidth=2)
        ax1.annotate(f'{start}', (coords[start, 0], coords[start, 1]),
                    fontsize=10, ha='center', va='center')
    
    ax1.set_title(f'最优布线路径\n总长度 = {sum(distance_matrix[solution[i], solution[(i+1)%len(solution)]] for i in range(len(solution))):.2f}', 
                 fontsize=14)
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右图：收敛曲线
    ax2.plot(history, 'b-', linewidth=2)
    ax2.set_title('算法收敛曲线', fontsize=14)
    ax2.set_xlabel('代数')
    ax2.set_ylabel('最优适应度（路径长度）')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("启发式算法 - 遗传算法 (GA)")
    print("=" * 60)
    
    # 创建问题
    n_nodes = 15
    distance_matrix, coords = create_cable_network_matrix(n_nodes, seed=42)
    print(f"\n✓ 创建电缆网络：{n_nodes} 个节点")
    
    # 创建 GA 求解器
    ga = GeneticAlgorithm(
        population_size=150,
        crossover_rate=0.85,
        mutation_rate=0.15,
        generations=300,
        elitism_count=10
    )
    
    # 求解
    print("\n⏳ 运行遗传算法...")
    best_solution, best_fitness = ga.solve(distance_matrix, verbose=True)
    
    # 输出结果
    print(f"\n📊 优化结果:")
    print(f"  最优路径：{best_solution}")
    print(f"  路径长度：{best_fitness:.2f}")
    
    # 可视化
    fig = visualize_ga_result(best_solution, distance_matrix, coords, ga.history)
    
    # 保存图像
    output_path = "/root/.openclaw/workspace/cable-optimization/examples/ga_result.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 结果图已保存到：{output_path}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
