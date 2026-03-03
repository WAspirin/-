"""
启发式算法 - 粒子群优化 (Particle Swarm Optimization, PSO)

问题描述:
使用 PSO 求解连续优化问题，应用于线缆布线的路径优化

算法原理:
1. 初始化：随机生成粒子群（位置和速度）
2. 评估：计算每个粒子的适应度
3. 更新个体最优 (pbest) 和全局最优 (gbest)
4. 更新速度和位置:
   v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
   x = x + v
5. 重复 2-4 直到满足终止条件

参数说明:
- w: 惯性权重 (0.4-0.9)，平衡全局和局部搜索
- c1: 个体学习因子 (通常 2.0)
- c2: 群体学习因子 (通常 2.0)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class Particle:
    """粒子类"""
    position: np.ndarray  # 当前位置
    velocity: np.ndarray  # 当前速度
    pbest_position: np.ndarray  # 个体最优位置
    pbest_fitness: float  # 个体最优适应度
    
    def __init__(self, dim: int, bounds: Tuple[float, float]):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')


class ParticleSwarmOptimizer:
    """粒子群优化求解器"""
    
    def __init__(self,
                 n_particles: int = 30,
                 n_dimensions: int = 2,
                 bounds: Tuple[float, float] = (-10, 10),
                 w: float = 0.7,  # 惯性权重
                 c1: float = 1.5,  # 个体学习因子
                 c2: float = 1.5,  # 群体学习因子
                 max_iterations: int = 100):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        
        self.particles: List[Particle] = []
        self.gbest_position: Optional[np.ndarray] = None
        self.gbest_fitness: float = float('inf')
        self.history = []
        
        self._initialize_particles()
    
    def _initialize_particles(self):
        """初始化粒子群"""
        self.particles = [
            Particle(self.n_dimensions, self.bounds)
            for _ in range(self.n_particles)
        ]
    
    def evaluate(self, position: np.ndarray) -> float:
        """
        适应度函数 - 最小化目标
        
        这里使用 Rastrigin 函数作为测试（多峰函数，适合测试全局搜索能力）
        f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        """
        n = len(position)
        return 10 * n + np.sum(position**2 - 10 * np.cos(2 * np.pi * position))
    
    def update_velocity(self, particle: Particle):
        """更新粒子速度"""
        r1 = np.random.random(self.n_dimensions)
        r2 = np.random.random(self.n_dimensions)
        
        # 速度更新公式
        cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (self.gbest_position - particle.position)
        
        particle.velocity = self.w * particle.velocity + cognitive + social
        
        # 速度限制
        max_vel = (self.bounds[1] - self.bounds[0]) * 0.2
        particle.velocity = np.clip(particle.velocity, -max_vel, max_vel)
    
    def update_position(self, particle: Particle):
        """更新粒子位置"""
        particle.position = particle.position + particle.velocity
        
        # 边界处理（反射）
        for i in range(self.n_dimensions):
            if particle.position[i] < self.bounds[0]:
                particle.position[i] = self.bounds[0]
                particle.velocity[i] = abs(particle.velocity[i])
            elif particle.position[i] > self.bounds[1]:
                particle.position[i] = self.bounds[1]
                particle.velocity[i] = -abs(particle.velocity[i])
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """执行 PSO 优化"""
        for iteration in range(self.max_iterations):
            # 评估每个粒子
            for particle in self.particles:
                fitness = self.evaluate(particle.position)
                
                # 更新个体最优
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = particle.position.copy()
                
                # 更新全局最优
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = particle.position.copy()
            
            # 记录历史
            self.history.append(self.gbest_fitness)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration+1}/{self.max_iterations}, "
                      f"最优适应度：{self.gbest_fitness:.6f}")
            
            # 更新粒子速度和位置
            for particle in self.particles:
                self.update_velocity(particle)
                self.update_position(particle)
        
        return self.gbest_position, self.gbest_fitness
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, linewidth=2, color='blue', label='全局最优')
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('适应度值', fontsize=12)
        plt.title('PSO 收敛曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"收敛曲线已保存到：{save_path}")
        
        plt.show()


def test_pso_on_sphere():
    """测试 PSO 在 Sphere 函数上的表现"""
    print("=" * 60)
    print("PSO 测试 - Sphere 函数")
    print("=" * 60)
    
    # Sphere 函数：f(x) = sum(x_i^2), 最小值在 (0,0,...,0)
    def sphere(x):
        return np.sum(x**2)
    
    # 创建优化器
    pso = ParticleSwarmOptimizer(
        n_particles=30,
        n_dimensions=5,
        bounds=(-5, 5),
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iterations=100
    )
    
    # 重写评估函数
    pso.evaluate = sphere
    
    # 执行优化
    best_pos, best_fit = pso.optimize()
    
    print(f"\n最优位置：{best_pos}")
    print(f"最优适应度：{best_fit:.6f}")
    print(f"理论最优值：0.0")
    
    return pso


def test_pso_on_rastrigin():
    """测试 PSO 在 Rastrigin 函数上的表现"""
    print("\n" + "=" * 60)
    print("PSO 测试 - Rastrigin 函数")
    print("=" * 60)
    
    pso = ParticleSwarmOptimizer(
        n_particles=40,
        n_dimensions=3,
        bounds=(-5.12, 5.12),
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iterations=150
    )
    
    best_pos, best_fit = pso.optimize()
    
    print(f"\n最优位置：{best_pos}")
    print(f"最优适应度：{best_fit:.6f}")
    print(f"理论最优值：0.0 (在全局最优处)")
    
    return pso


def apply_pso_to_cable_routing():
    """
    PSO 在线缆布线中的应用示例
    
    问题：优化线缆路径的多个控制点位置，使得总长度最小
    """
    print("\n" + "=" * 60)
    print("PSO 应用 - 线缆路径优化")
    print("=" * 60)
    
    # 模拟布线场景：起点 (0,0), 终点 (10,10), 中间有 3 个控制点
    start = np.array([0, 0])
    end = np.array([10, 10])
    n_control_points = 3
    
    # 障碍物位置
    obstacles = [
        np.array([4, 5]),
        np.array([6, 6]),
        np.array([5, 3])
    ]
    
    def cable_length_fitness(positions: np.ndarray) -> float:
        """
        适应度函数：线缆总长度 + 障碍物惩罚
        
        positions: 控制点坐标 [x1, y1, x2, y2, x3, y3]
        """
        # 重构控制点
        control_points = positions.reshape(-1, 2)
        
        # 构建完整路径
        path = np.vstack([start, control_points, end])
        
        # 计算总长度
        total_length = 0
        for i in range(len(path) - 1):
            total_length += np.linalg.norm(path[i+1] - path[i])
        
        # 障碍物惩罚（如果太接近障碍物）
        penalty = 0
        min_safe_distance = 1.5
        for cp in control_points:
            for obs in obstacles:
                dist = np.linalg.norm(cp - obs)
                if dist < min_safe_distance:
                    penalty += (min_safe_distance - dist) * 100
        
        return total_length + penalty
    
    # 创建优化器
    pso = ParticleSwarmOptimizer(
        n_particles=50,
        n_dimensions=n_control_points * 2,  # 每个控制点有 x,y 坐标
        bounds=(0, 10),
        w=0.8,
        c1=1.5,
        c2=1.5,
        max_iterations=200
    )
    
    # 重写评估函数
    pso.evaluate = cable_length_fitness
    
    # 执行优化
    best_pos, best_fit = pso.optimize()
    
    # 可视化结果
    control_points = best_pos.reshape(-1, 2)
    path = np.vstack([start, control_points, end])
    
    print(f"\n优化后的控制点位置:")
    for i, cp in enumerate(control_points):
        print(f"  控制点 {i+1}: ({cp[0]:.2f}, {cp[1]:.2f})")
    print(f"\n优化后线缆总长度：{best_fit:.2f}")
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 绘制障碍物
    for obs in obstacles:
        circle = plt.Circle(obs, 1.5, color='red', alpha=0.3, label='障碍区域')
        plt.gca().add_patch(circle)
        plt.plot(obs[0], obs[1], 'r*', markersize=15)
    
    # 绘制路径
    plt.plot(path[:, 0], path[:, 1], 'b-o', linewidth=2, markersize=8, label='优化路径')
    plt.plot(start[0], start[1], 'gs', markersize=15, label='起点')
    plt.plot(end[0], end[1], 'r^', markersize=15, label='终点')
    
    plt.xlabel('X 坐标', fontsize=12)
    plt.ylabel('Y 坐标', fontsize=12)
    plt.title('PSO 优化的线缆布线路径', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('/root/.openclaw/workspace/cable-optimization/examples/pso_cable_routing.png', dpi=150)
    print("\n路径图已保存到：pso_cable_routing.png")
    plt.show()
    
    return pso


if __name__ == "__main__":
    print("\n" + "🧠" * 30)
    print("粒子群优化算法 (PSO) - 线缆布线优化")
    print("🧠" * 30 + "\n")
    
    # 测试 1: Sphere 函数
    pso1 = test_pso_on_sphere()
    pso1.plot_convergence('/root/.openclaw/workspace/cable-optimization/examples/pso_sphere_convergence.png')
    
    # 测试 2: Rastrigin 函数
    pso2 = test_pso_on_rastrigin()
    pso2.plot_convergence('/root/.openclaw/workspace/cable-optimization/examples/pso_rastrigin_convergence.png')
    
    # 测试 3: 线缆布线应用
    pso3 = apply_pso_to_cable_routing()
    
    print("\n" + "=" * 60)
    print("PSO 算法学习完成!")
    print("=" * 60)
    print("\n关键收获:")
    print("1. PSO 通过模拟鸟群行为进行全局搜索")
    print("2. 速度和位置更新公式是核心")
    print("3. 参数 w, c1, c2 影响收敛性能")
    print("4. 适合连续优化问题")
    print("5. 可应用于路径控制点优化")
    print("\n明日预告：模拟退火算法 (SA)")
