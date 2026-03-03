"""
启发式算法 - 模拟退火 (Simulated Annealing, SA)

问题描述:
使用 SA 求解组合优化问题，应用于线缆布线路径优化

算法原理:
1. 初始化：随机生成初始解，设置初始温度
2. 产生新解：在当前解的邻域内随机扰动
3. 评估：计算新解的适应度
4. Metropolis 准则接受:
   - 如果新解更好：接受
   - 如果新解更差：以概率 exp(-ΔE/T) 接受
5. 降温：T = T * cooling_rate
6. 重复 2-5 直到温度低于阈值

核心思想:
- 高温时：接受较差解的概率高，有利于跳出局部最优
- 低温时：几乎只接受更好的解，趋于收敛

参数说明:
- initial_temp: 初始温度（越高越容易接受差解）
- cooling_rate: 降温系数 (0.9-0.99)
- min_temp: 终止温度
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
import random
import math


class SimulatedAnnealing:
    """模拟退火求解器"""
    
    def __init__(self,
                 initial_temp: float = 1000.0,
                 min_temp: float = 1e-6,
                 cooling_rate: float = 0.995,
                 max_iterations_per_temp: int = 100):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iterations_per_temp = max_iterations_per_temp
        
        self.current_solution: Optional[np.ndarray] = None
        self.current_fitness: float = float('inf')
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')
        
        self.history = []  # 记录每代最优适应度
        self.temperature_history = []  # 记录温度变化
        self.acceptance_history = []  # 记录接受率
    
    def initialize(self, initial_solution: np.ndarray, 
                   evaluate_func: Callable[[np.ndarray], float]):
        """初始化求解器"""
        self.current_solution = initial_solution.copy()
        self.current_fitness = evaluate_func(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_fitness = self.current_fitness
    
    def generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """
        生成邻域解
        
        这里使用高斯扰动，适用于连续优化
        对于离散问题，可以使用交换、逆转等操作
        """
        neighbor = solution.copy()
        
        # 随机选择一些维度进行扰动
        n_perturb = max(1, len(solution) // 5)
        indices = random.sample(range(len(solution)), n_perturb)
        
        for idx in indices:
            # 高斯扰动
            neighbor[idx] += np.random.normal(0, 1)
        
        return neighbor
    
    def acceptance_probability(self, delta_fitness: float, temperature: float) -> float:
        """
        Metropolis 准则计算接受概率
        
        如果新解更好 (delta < 0)，返回 1
        否则返回 exp(-delta / T)
        """
        if delta_fitness < 0:
            return 1.0
        return math.exp(-delta_fitness / temperature)
    
    def optimize(self, 
                 initial_solution: np.ndarray,
                 evaluate_func: Callable[[np.ndarray], float],
                 verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        执行模拟退火优化
        
        Args:
            initial_solution: 初始解
            evaluate_func: 适应度评估函数（越小越好）
            verbose: 是否打印进度
        
        Returns:
            (最优解，最优适应度)
        """
        # 初始化
        self.initialize(initial_solution, evaluate_func)
        
        temperature = self.initial_temp
        iteration = 0
        total_accepted = 0
        total_trials = 0
        
        while temperature > self.min_temp:
            accepted_at_temp = 0
            
            for _ in range(self.max_iterations_per_temp):
                # 生成邻域解
                neighbor = self.generate_neighbor(self.current_solution)
                neighbor_fitness = evaluate_func(neighbor)
                
                # 计算适应度变化
                delta_fitness = neighbor_fitness - self.current_fitness
                
                # Metropolis 准则
                if random.random() < self.acceptance_probability(delta_fitness, temperature):
                    self.current_solution = neighbor.copy()
                    self.current_fitness = neighbor_fitness
                    accepted_at_temp += 1
                    total_accepted += 1
                
                # 更新全局最优
                if self.current_fitness < self.best_fitness:
                    self.best_fitness = self.current_fitness
                    self.best_solution = self.current_solution.copy()
                
                total_trials += 1
                iteration += 1
            
            # 记录历史
            self.history.append(self.best_fitness)
            self.temperature_history.append(temperature)
            acceptance_rate = accepted_at_temp / self.max_iterations_per_temp
            self.acceptance_history.append(acceptance_rate)
            
            if verbose and len(self.history) % 50 == 0:
                print(f"迭代 {iteration}, 温度 {temperature:.4f}, "
                      f"最优适应度：{self.best_fitness:.6f}, "
                      f"接受率：{acceptance_rate:.2%}")
            
            # 降温
            temperature *= self.cooling_rate
        
        return self.best_solution, self.best_fitness
    
    def plot_results(self, save_path: Optional[str] = None):
        """绘制优化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 收敛曲线
        ax1 = axes[0, 0]
        ax1.plot(self.history, linewidth=2, color='blue')
        ax1.set_xlabel('迭代次数 (温度批次)', fontsize=12)
        ax1.set_ylabel('最优适应度', fontsize=12)
        ax1.set_title('SA 收敛曲线', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 2. 温度变化曲线
        ax2 = axes[0, 1]
        ax2.semilogy(self.temperature_history, linewidth=2, color='red')
        ax2.set_xlabel('迭代次数 (温度批次)', fontsize=12)
        ax2.set_ylabel('温度 (对数坐标)', fontsize=12)
        ax2.set_title('退火过程温度变化', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. 接受率变化
        ax3 = axes[1, 0]
        ax3.plot(self.acceptance_history, linewidth=2, color='green')
        ax3.set_xlabel('迭代次数 (温度批次)', fontsize=12)
        ax3.set_ylabel('接受率', fontsize=12)
        ax3.set_title('Metropolis 准则接受率', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0.37, color='gray', linestyle='--', label='1/e ≈ 0.37')
        ax3.legend()
        
        # 4. 适应度分布直方图
        ax4 = axes[1, 1]
        ax4.hist(self.history, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('适应度值', fontsize=12)
        ax4.set_ylabel('频次', fontsize=12)
        ax4.set_title('搜索过程中适应度分布', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"结果图已保存到：{save_path}")
        
        plt.show()


def test_sa_on_sphere():
    """测试 SA 在 Sphere 函数上的表现"""
    print("=" * 60)
    print("SA 测试 - Sphere 函数")
    print("=" * 60)
    
    # Sphere 函数
    def sphere(x):
        return np.sum(x**2)
    
    # 创建求解器
    sa = SimulatedAnnealing(
        initial_temp=1000.0,
        min_temp=1e-6,
        cooling_rate=0.995,
        max_iterations_per_temp=50
    )
    
    # 随机初始解
    initial_solution = np.random.uniform(-5, 5, 5)
    
    # 执行优化
    best_sol, best_fit = sa.optimize(initial_solution, sphere)
    
    print(f"\n初始解：{initial_solution}")
    print(f"最优解：{best_sol}")
    print(f"最优适应度：{best_fit:.6f}")
    print(f"理论最优值：0.0")
    
    return sa


def test_sa_on_tsp():
    """测试 SA 在 TSP 问题上的表现"""
    print("\n" + "=" * 60)
    print("SA 测试 - TSP 问题")
    print("=" * 60)
    
    # 生成随机城市
    n_cities = 20
    np.random.seed(42)
    cities = np.random.rand(n_cities, 2) * 100
    
    # 距离矩阵
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            dist_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
    
    def tsp_fitness(tour: np.ndarray) -> float:
        """计算 TSP 路径总长度"""
        total = 0
        for i in range(len(tour) - 1):
            total += dist_matrix[int(tour[i]), int(tour[i+1])]
        total += dist_matrix[int(tour[-1]), int(tour[0])]  # 返回起点
        return total
    
    # 创建求解器
    sa = SimulatedAnnealing(
        initial_temp=10000.0,
        min_temp=1e-4,
        cooling_rate=0.99,
        max_iterations_per_temp=100
    )
    
    # 重写邻域生成（交换两个城市）
    def tsp_neighbor(solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        i, j = random.sample(range(len(solution)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    sa.generate_neighbor = tsp_neighbor
    
    # 初始解（随机排列）
    initial_tour = np.random.permutation(n_cities).astype(float)
    
    # 执行优化
    best_tour, best_length = sa.optimize(initial_tour, tsp_fitness)
    
    print(f"\n城市数量：{n_cities}")
    print(f"初始路径长度：{tsp_fitness(initial_tour):.2f}")
    print(f"最优路径长度：{best_length:.2f}")
    print(f"优化比例：{(1 - best_length/tsp_fitness(initial_tour))*100:.1f}%")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 左图：城市分布和最优路径
    plt.subplot(1, 2, 1)
    plt.plot(cities[:, 0], cities[:, 1], 'bo', markersize=8, label='城市')
    
    best_tour_int = best_tour.astype(int)
    tour_cities = np.vstack([cities[best_tour_int], cities[best_tour_int[0]]])
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], 'r-o', linewidth=2, markersize=6, label='最优路径')
    
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), fontsize=8, alpha=0.7)
    
    plt.xlabel('X 坐标', fontsize=12)
    plt.ylabel('Y 坐标', fontsize=12)
    plt.title(f'TSP 最优路径 (长度={best_length:.2f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：收敛曲线
    plt.subplot(1, 2, 2)
    plt.plot(sa.history, linewidth=2, color='blue')
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('路径长度', fontsize=12)
    plt.title('SA 收敛过程', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/.openclaw/workspace/cable-optimization/examples/sa_tsp_result.png', dpi=150)
    print("\nTSP 结果图已保存到：sa_tsp_result.png")
    plt.show()
    
    return sa


def apply_sa_to_cable_routing():
    """
    SA 在线缆布线中的应用示例
    
    问题：优化线缆路径，避开障碍物并最小化长度
    """
    print("\n" + "=" * 60)
    print("SA 应用 - 线缆路径优化")
    print("=" * 60)
    
    # 布线场景
    start = np.array([0, 0])
    end = np.array([10, 10])
    n_control_points = 4
    
    # 障碍物
    obstacles = [
        np.array([3, 4]),
        np.array([5, 6]),
        np.array([7, 5]),
        np.array([4, 7])
    ]
    
    def cable_fitness(positions: np.ndarray) -> float:
        """适应度函数：长度 + 障碍惩罚"""
        control_points = positions.reshape(-1, 2)
        path = np.vstack([start, control_points, end])
        
        # 总长度
        total_length = sum(np.linalg.norm(path[i+1] - path[i]) 
                          for i in range(len(path) - 1))
        
        # 障碍惩罚
        penalty = 0
        min_safe_dist = 1.2
        for cp in control_points:
            for obs in obstacles:
                dist = np.linalg.norm(cp - obs)
                if dist < min_safe_dist:
                    penalty += (min_safe_dist - dist) ** 2 * 50
        
        return total_length + penalty
    
    # 创建求解器
    sa = SimulatedAnnealing(
        initial_temp=5000.0,
        min_temp=1e-5,
        cooling_rate=0.992,
        max_iterations_per_temp=80
    )
    
    # 自定义邻域生成（针对控制点位置）
    def cable_neighbor(solution: np.ndarray) -> np.ndarray:
        neighbor = solution.copy()
        control_points = neighbor.reshape(-1, 2)
        
        # 随机选择一个控制点进行扰动
        idx = random.randint(0, len(control_points) - 1)
        control_points[idx] += np.random.normal(0, 0.5, 2)
        
        # 边界限制
        control_points[idx] = np.clip(control_points[idx], 0, 10)
        
        return neighbor
    
    sa.generate_neighbor = cable_neighbor
    
    # 初始解
    initial_positions = np.random.uniform(0, 10, n_control_points * 2)
    
    # 执行优化
    best_positions, best_fitness = sa.optimize(initial_positions, cable_fitness)
    
    # 可视化
    control_points = best_positions.reshape(-1, 2)
    path = np.vstack([start, control_points, end])
    
    plt.figure(figsize=(10, 8))
    
    # 绘制障碍物
    for obs in obstacles:
        circle = plt.Circle(obs, 1.2, color='red', alpha=0.3)
        plt.gca().add_patch(circle)
        plt.plot(obs[0], obs[1], 'r*', markersize=15)
    
    # 绘制路径
    plt.plot(path[:, 0], path[:, 1], 'b-o', linewidth=2, markersize=8, label='优化路径')
    plt.plot(start[0], start[1], 'gs', markersize=15, label='起点')
    plt.plot(end[0], end[1], 'r^', markersize=15, label='终点')
    
    # 绘制控制点
    plt.plot(control_points[:, 0], control_points[:, 1], 'mo', 
             markersize=12, label='控制点', fillstyle='none')
    
    plt.xlabel('X 坐标', fontsize=12)
    plt.ylabel('Y 坐标', fontsize=12)
    plt.title('SA 优化的线缆布线路径', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.tight_layout()
    plt.savefig('/root/.openclaw/workspace/cable-optimization/examples/sa_cable_routing.png', dpi=150)
    print(f"\n优化后线缆总长度：{best_fitness:.2f}")
    print("路径图已保存到：sa_cable_routing.png")
    plt.show()
    
    return sa


if __name__ == "__main__":
    print("\n" + "🔥" * 30)
    print("模拟退火算法 (SA) - 线缆布线优化")
    print("🔥" * 30 + "\n")
    
    # 测试 1: Sphere 函数
    sa1 = test_sa_on_sphere()
    sa1.plot_results('/root/.openclaw/workspace/cable-optimization/examples/sa_sphere_results.png')
    
    # 测试 2: TSP 问题
    sa2 = test_sa_on_tsp()
    
    # 测试 3: 线缆布线应用
    sa3 = apply_sa_to_cable_routing()
    
    print("\n" + "=" * 60)
    print("SA 算法学习完成!")
    print("=" * 60)
    print("\n关键收获:")
    print("1. SA 模拟金属退火过程，通过温度控制搜索")
    print("2. Metropolis 准则允许接受差解，避免局部最优")
    print("3. 退火计划（降温策略）影响收敛质量")
    print("4. 适合组合优化和连续优化问题")
    print("5. 实现简单，参数较少")
    print("\n明日预告：A* 搜索算法")
