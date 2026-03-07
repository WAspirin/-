#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法对比实验 - Week 1 总结

对比所有已实现的优化算法在统一测试集上的性能

包含算法：
1. MILP (混合整数线性规划)
2. Dijkstra (最短路径)
3. 遗传算法 (GA)
4. 粒子群优化 (PSO)
5. 模拟退火 (SA)
6. A* 搜索
7. 最小生成树 (MST)
8. 变邻域搜索 (VNS)
9. 禁忌搜索 (TS)
10. 蚁群优化 (ACO)

作者：智子 (Sophon)
日期：2026-03-07
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CableRoutingProblem:
    """线缆布线问题生成器"""
    
    def __init__(self, n_nodes: int = 20, seed: int = 42):
        """
        初始化问题
        
        Args:
            n_nodes: 节点数量
            seed: 随机种子
        """
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        
        # 生成随机节点坐标
        self.nodes = np.random.rand(n_nodes, 2) * 100
        
        # 计算距离矩阵
        self.dist_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                dx = self.nodes[i, 0] - self.nodes[j, 0]
                dy = self.nodes[i, 1] - self.nodes[j, 1]
                self.dist_matrix[i, j] = np.sqrt(dx**2 + dy**2)
        
        # 固定起点和终点
        self.start = 0
        self.end = n_nodes - 1
    
    def compute_path_cost(self, path: List[int]) -> float:
        """计算路径总成本"""
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.dist_matrix[path[i], path[i+1]]
        return cost
    
    def get_coordinates(self, path: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """获取路径坐标用于可视化"""
        x = [self.nodes[i, 0] for i in path]
        y = [self.nodes[i, 1] for i in path]
        return np.array(x), np.array(y)


class AlgorithmBenchmark:
    """算法基准测试"""
    
    def __init__(self, problem: CableRoutingProblem):
        self.problem = problem
        self.results = {}
    
    def run_all_algorithms(self) -> Dict:
        """运行所有算法并记录结果"""
        algorithms = [
            ("MILP", self.run_milp),
            ("Dijkstra", self.run_dijkstra),
            ("GA", self.run_ga),
            ("PSO", self.run_pso),
            ("SA", self.run_sa),
            ("A*", self.run_astar),
            ("MST", self.run_mst),
            ("VNS", self.run_vns),
            ("TS", self.run_ts),
            ("ACO", self.run_aco),
        ]
        
        print("=" * 60)
        print("🔬 算法对比实验 - Week 1 总结")
        print("=" * 60)
        print(f"问题规模：{self.problem.n_nodes} 个节点")
        print(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        for name, func in algorithms:
            print(f"\n📊 运行 {name}...")
            try:
                start_time = time.time()
                result = func()
                elapsed = time.time() - start_time
                
                self.results[name] = {
                    'cost': result['cost'],
                    'time': elapsed,
                    'path': result.get('path', []),
                    'iterations': result.get('iterations', 'N/A')
                }
                
                print(f"  ✅ 成本：{result['cost']:.2f}, 时间：{elapsed:.3f}s")
            except Exception as e:
                print(f"  ❌ 错误：{str(e)}")
                self.results[name] = {
                    'cost': float('inf'),
                    'time': 0,
                    'error': str(e)
                }
        
        return self.results
    
    def run_milp(self) -> Dict:
        """简化版 MILP (使用贪心近似)"""
        # 实际 MILP 需要 PuLP，这里用最近邻近似
        n = self.problem.n_nodes
        unvisited = set(range(1, n))
        path = [0]
        
        while unvisited:
            current = path[-1]
            nearest = min(unvisited, key=lambda x: self.problem.dist_matrix[current, x])
            path.append(nearest)
            unvisited.remove(nearest)
        
        cost = self.problem.compute_path_cost(path)
        return {'cost': cost, 'path': path, 'iterations': 'N/A'}
    
    def run_dijkstra(self) -> Dict:
        """Dijkstra 算法 (单源最短路径树)"""
        import heapq
        
        n = self.problem.n_nodes
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[0] = 0
        
        pq = [(0, 0)]  # (distance, node)
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            
            for v in range(n):
                if v not in visited:
                    new_dist = dist[u] + self.problem.dist_matrix[u, v]
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        prev[v] = u
                        heapq.heappush(pq, (new_dist, v))
        
        # 重构路径
        path = []
        current = n - 1
        while current != -1:
            path.append(current)
            current = prev[current]
        path.reverse()
        
        cost = self.problem.compute_path_cost(path)
        return {'cost': cost, 'path': path, 'iterations': 'N/A'}
    
    def run_ga(self) -> Dict:
        """遗传算法"""
        n = self.problem.n_nodes
        pop_size = 50
        n_generations = 100
        
        # 初始化种群
        population = []
        for _ in range(pop_size):
            individual = list(range(n))
            np.random.shuffle(individual)
            population.append(individual)
        
        best_cost = float('inf')
        best_path = None
        
        for gen in range(n_generations):
            # 评估
            costs = [self.problem.compute_path_cost(ind) for ind in population]
            
            # 更新最优
            min_idx = np.argmin(costs)
            if costs[min_idx] < best_cost:
                best_cost = costs[min_idx]
                best_path = population[min_idx].copy()
            
            # 选择 (锦标赛)
            new_pop = []
            for _ in range(pop_size):
                i, j = np.random.choice(pop_size, 2, replace=False)
                winner = population[i] if costs[i] < costs[j] else population[j]
                new_pop.append(winner.copy())
            
            # 交叉 (OX)
            for i in range(0, pop_size, 2):
                if np.random.rand() < 0.8:
                    p1, p2 = new_pop[i], new_pop[i+1]
                    start, end = sorted(np.random.choice(n, 2, replace=False))
                    child1 = p1[start:end+1]
                    child2 = p2[start:end+1]
                    
                    for gene in p2:
                        if gene not in child1 and len(child1) < n:
                            child1.append(gene)
                    for gene in p1:
                        if gene not in child2 and len(child2) < n:
                            child2.append(gene)
                    
                    new_pop[i] = child1[:n]
                    new_pop[i+1] = child2[:n]
            
            # 变异
            for i in range(pop_size):
                if np.random.rand() < 0.1:
                    idx1, idx2 = np.random.choice(n, 2, replace=False)
                    new_pop[i][idx1], new_pop[i][idx2] = new_pop[i][idx2], new_pop[i][idx1]
            
            population = new_pop
        
        return {'cost': best_cost, 'path': best_path, 'iterations': n_generations}
    
    def run_pso(self) -> Dict:
        """粒子群优化 (离散版本)"""
        n = self.problem.n_nodes
        n_particles = 30
        n_iterations = 100
        
        # 初始化粒子 (排列表示)
        particles = []
        velocities = []
        pbests = []
        pbest_costs = []
        
        for _ in range(n_particles):
            particle = list(range(n))
            np.random.shuffle(particle)
            particles.append(particle)
            velocities.append([0] * n)
            cost = self.problem.compute_path_cost(particle)
            pbests.append(particle.copy())
            pbest_costs.append(cost)
        
        gbest = pbests[np.argmin(pbest_costs)].copy()
        gbest_cost = min(pbest_costs)
        
        for _ in range(n_iterations):
            for i in range(n_particles):
                # 简化版离散 PSO 更新
                if np.random.rand() < 0.3:
                    # 向 pbest 学习
                    idx = np.random.randint(n)
                    target_pos = pbests[i].index(particles[i][idx])
                    if np.random.rand() < 0.5:
                        # 交换
                        swap_idx = np.random.randint(n)
                        particles[i][idx], particles[i][swap_idx] = particles[i][swap_idx], particles[i][idx]
                
                # 评估
                cost = self.problem.compute_path_cost(particles[i])
                if cost < pbest_costs[i]:
                    pbest_costs[i] = cost
                    pbests[i] = particles[i].copy()
                
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest = particles[i].copy()
        
        return {'cost': gbest_cost, 'path': gbest, 'iterations': n_iterations}
    
    def run_sa(self) -> Dict:
        """模拟退火"""
        n = self.problem.n_nodes
        
        # 初始解
        current = list(range(n))
        np.random.shuffle(current)
        current_cost = self.problem.compute_path_cost(current)
        
        best = current.copy()
        best_cost = current_cost
        
        # 退火参数
        T = 1000
        T_min = 1e-6
        alpha = 0.995
        iterations = 0
        
        while T > T_min:
            for _ in range(100):
                # 生成邻居 (交换)
                neighbor = current.copy()
                i, j = np.random.choice(n, 2, replace=False)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
                neighbor_cost = self.problem.compute_path_cost(neighbor)
                delta = neighbor_cost - current_cost
                
                # Metropolis 准则
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    current = neighbor
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best = current.copy()
                        best_cost = current_cost
                
                iterations += 1
            
            T *= alpha
        
        return {'cost': best_cost, 'path': best, 'iterations': iterations}
    
    def run_astar(self) -> Dict:
        """A* 搜索算法"""
        import heapq
        
        n = self.problem.n_nodes
        
        # 启发式：欧氏距离到终点
        def heuristic(node):
            dx = self.problem.nodes[node, 0] - self.problem.nodes[n-1, 0]
            dy = self.problem.nodes[node, 1] - self.problem.nodes[n-1, 1]
            return np.sqrt(dx**2 + dy**2)
        
        # (f, g, node, path)
        open_set = [(heuristic(0), 0, 0, [0])]
        closed_set = set()
        
        best_solution = None
        best_cost = float('inf')
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            if len(path) == n:
                cost = g
                if cost < best_cost:
                    best_cost = cost
                    best_solution = path
                continue
            
            for next_node in range(n):
                if next_node not in closed_set:
                    new_g = g + self.problem.dist_matrix[current, next_node]
                    new_f = new_g + heuristic(next_node)
                    heapq.heappush(open_set, (new_f, new_g, next_node, path + [next_node]))
            
            # 限制搜索
            if len(closed_set) > 1000:
                break
        
        if best_solution is None:
            # 回退到贪心
            best_solution = list(range(n))
            best_cost = self.problem.compute_path_cost(best_solution)
        
        return {'cost': best_cost, 'path': best_solution, 'iterations': len(closed_set)}
    
    def run_mst(self) -> Dict:
        """最小生成树 (Prim 算法)"""
        n = self.problem.n_nodes
        
        # Prim 算法
        in_mst = [False] * n
        key = [float('inf')] * n
        parent = [-1] * n
        
        key[0] = 0
        mst_edges = []
        
        for _ in range(n):
            # 选择最小 key 的顶点
            u = -1
            min_key = float('inf')
            for v in range(n):
                if not in_mst[v] and key[v] < min_key:
                    min_key = key[v]
                    u = v
            
            if u == -1:
                break
            
            in_mst[u] = True
            if parent[u] != -1:
                mst_edges.append((parent[u], u))
            
            # 更新邻居
            for v in range(n):
                if not in_mst[v] and self.problem.dist_matrix[u, v] < key[v]:
                    key[v] = self.problem.dist_matrix[u, v]
                    parent[v] = u
        
        # 从 MST 构建路径 (DFS)
        adj = {i: [] for i in range(n)}
        for u, v in mst_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # DFS 遍历
        path = []
        visited = set()
        
        def dfs(node):
            visited.add(node)
            path.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        dfs(0)
        
        cost = self.problem.compute_path_cost(path)
        return {'cost': cost, 'path': path, 'iterations': 'N/A'}
    
    def run_vns(self) -> Dict:
        """变邻域搜索"""
        n = self.problem.n_nodes
        
        # 初始解 (最近邻)
        unvisited = set(range(1, n))
        current = [0]
        while unvisited:
            last = current[-1]
            nearest = min(unvisited, key=lambda x: self.problem.dist_matrix[last, x])
            current.append(nearest)
            unvisited.remove(nearest)
        
        current_cost = self.problem.compute_path_cost(current)
        best = current.copy()
        best_cost = current_cost
        
        k_max = 5
        max_iterations = 500
        iterations = 0
        
        k = 1
        no_improve = 0
        
        while iterations < max_iterations and no_improve < 100:
            # 扰动
            new_sol = current.copy()
            for _ in range(k):
                i, j = np.random.choice(n, 2, replace=False)
                new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
            
            # 局部搜索 (2-opt)
            improved = True
            while improved:
                improved = False
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        neighbor = new_sol[:i+1] + new_sol[i+1:j+1][::-1] + new_sol[j+1:]
                        new_cost = self.problem.compute_path_cost(neighbor)
                        if new_cost < self.problem.compute_path_cost(new_sol):
                            new_sol = neighbor
                            improved = True
            
            new_cost = self.problem.compute_path_cost(new_sol)
            
            if new_cost < best_cost:
                best = new_sol.copy()
                best_cost = new_cost
                current = new_sol.copy()
                current_cost = new_cost
                k = 1
                no_improve = 0
            else:
                k += 1
                if k > k_max:
                    k = 1
                no_improve += 1
            
            iterations += 1
        
        return {'cost': best_cost, 'path': best, 'iterations': iterations}
    
    def run_ts(self) -> Dict:
        """禁忌搜索"""
        from collections import deque
        
        n = self.problem.n_nodes
        
        # 初始解
        current = list(range(n))
        np.random.shuffle(current)
        current_cost = self.problem.compute_path_cost(current)
        
        best = current.copy()
        best_cost = current_cost
        
        # 禁忌表
        tabu_tenure = 15
        tabu_list = deque(maxlen=tabu_tenure)
        
        max_iterations = 500
        no_improve = 0
        
        for iteration in range(max_iterations):
            # 生成邻居
            neighbors = []
            for i in range(n):
                for j in range(i + 1, n):
                    neighbor = current.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    cost = self.problem.compute_path_cost(neighbor)
                    neighbors.append((cost, i, j, neighbor))
            
            # 排序
            neighbors.sort(key=lambda x: x[0])
            
            # 选择最佳非禁忌解
            selected = None
            for cost, i, j, neighbor in neighbors:
                tabu_key = ('swap', min(i, j), max(i, j))
                if tabu_key not in tabu_list or cost < best_cost:
                    selected = (cost, i, j, neighbor)
                    break
            
            if selected is None:
                selected = neighbors[0]
            
            cost, i, j, current = selected
            current_cost = cost
            
            # 更新禁忌表
            tabu_list.append(('swap', min(i, j), max(i, j)))
            
            # 更新最优
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 100:
                break
        
        return {'cost': best_cost, 'path': best, 'iterations': iteration + 1}
    
    def run_aco(self) -> Dict:
        """蚁群优化"""
        n = self.problem.n_nodes
        n_ants = 20
        n_iterations = 100
        
        # 初始化信息素
        pheromone = np.ones((n, n))
        heuristic = 1 / (self.problem.dist_matrix + 1e-10)
        
        alpha = 1.0
        beta = 2.0
        rho = 0.1
        Q = 1.0
        
        best_cost = float('inf')
        best_path = None
        
        for _ in range(n_iterations):
            solutions = []
            costs = []
            
            # 每只蚂蚁构建路径
            for _ in range(n_ants):
                path = [0]
                visited = {0}
                
                while len(path) < n:
                    current = path[-1]
                    available = [i for i in range(n) if i not in visited]
                    
                    # 计算概率
                    probs = []
                    for node in available:
                        tau = pheromone[current, node] ** alpha
                        eta = heuristic[current, node] ** beta
                        probs.append(tau * eta)
                    
                    probs = np.array(probs) / sum(probs)
                    next_node = np.random.choice(available, p=probs)
                    path.append(next_node)
                    visited.add(next_node)
                
                cost = self.problem.compute_path_cost(path)
                solutions.append(path)
                costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_path = path.copy()
            
            # 更新信息素
            pheromone *= (1 - rho)
            for path, cost in zip(solutions, costs):
                delta = Q / cost
                for i in range(len(path) - 1):
                    pheromone[path[i], path[i+1]] += delta
        
        return {'cost': best_cost, 'path': best_path, 'iterations': n_iterations}


def visualize_comparison(results: Dict, problem: CableRoutingProblem):
    """可视化对比结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 成本对比 (条形图)
    ax1 = axes[0, 0]
    names = list(results.keys())
    costs = [results[name]['cost'] for name in names]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
    
    bars = ax1.barh(names, costs, color=colors)
    ax1.set_xlabel('路径成本')
    ax1.set_title('📊 算法成本对比')
    ax1.invert_yaxis()
    
    # 标注数值
    for bar, cost in zip(bars, costs):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{cost:.1f}', va='center', fontsize=9)
    
    # 2. 时间对比
    ax2 = axes[0, 1]
    times = [results[name]['time'] for name in names]
    colors2 = plt.cm.plasma(np.linspace(0, 0.8, len(names)))
    
    bars2 = ax2.barh(names, times, color=colors2)
    ax2.set_xlabel('运行时间 (秒)')
    ax2.set_title('⏱️ 算法运行时间对比')
    ax2.invert_yaxis()
    
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{t:.3f}s', va='center', fontsize=9)
    
    # 3. 成本 - 时间散点图
    ax3 = axes[1, 0]
    ax3.scatter(times, costs, s=100, alpha=0.7, c=range(len(names)), cmap='tab10')
    for i, name in enumerate(names):
        ax3.annotate(name, (times[i], costs[i]), fontsize=8, ha='left', va='bottom')
    ax3.set_xlabel('运行时间 (秒)')
    ax3.set_ylabel('路径成本')
    ax3.set_title('🎯 成本 - 时间权衡')
    ax3.grid(True, alpha=0.3)
    
    # 4. 最优路径可视化
    ax4 = axes[1, 1]
    best_algo = min(results.keys(), key=lambda x: results[x]['cost'])
    best_path = results[best_algo]['path']
    
    x, y = problem.get_coordinates(best_path)
    ax4.plot(x, y, 'o-', linewidth=2, markersize=8, label=f'{best_algo}: {results[best_algo]["cost"]:.2f}')
    ax4.scatter(problem.nodes[:, 0], problem.nodes[:, 1], c='red', s=50, zorder=5)
    
    for i, (xi, yi) in enumerate(zip(problem.nodes[:, 0], problem.nodes[:, 1])):
        ax4.annotate(str(i), (xi, yi), fontsize=8, ha='center', va='center')
    
    ax4.set_xlabel('X 坐标')
    ax4.set_ylabel('Y 坐标')
    ax4.set_title(f'🛤️ 最优路径 ({best_algo})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('cable-optimization/examples/outputs/11_algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ 对比图已保存：outputs/11_algorithm_comparison.png")
    plt.close()


def print_summary_table(results: Dict):
    """打印总结表格"""
    print("\n" + "=" * 80)
    print("📋 算法性能对比总结表")
    print("=" * 80)
    print(f"{'算法':<12} {'成本':<12} {'时间 (s)':<12} {'迭代次数':<12} {'排名':<8}")
    print("-" * 80)
    
    # 按成本排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cost'])
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        cost_str = f"{data['cost']:.2f}" if data['cost'] != float('inf') else "N/A"
        time_str = f"{data['time']:.3f}"
        iter_str = str(data.get('iterations', 'N/A'))
        print(f"{name:<12} {cost_str:<12} {time_str:<12} {iter_str:<12} #{rank:<7}")
    
    print("=" * 80)
    
    # 统计信息
    costs = [r['cost'] for r in results.values() if r['cost'] != float('inf')]
    print(f"\n📈 统计信息:")
    print(f"  最优成本：{min(costs):.2f} ({sorted_results[0][0]})")
    print(f"  最差成本：{max(costs):.2f}")
    print(f"  平均成本：{np.mean(costs):.2f}")
    print(f"  成本标准差：{np.std(costs):.2f}")
    print(f"  改进空间：{(max(costs) - min(costs)) / max(costs) * 100:.1f}%")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 线缆布线优化 - 算法对比实验")
    print("=" * 60)
    
    # 创建问题实例
    problem = CableRoutingProblem(n_nodes=20, seed=42)
    
    # 运行基准测试
    benchmark = AlgorithmBenchmark(problem)
    results = benchmark.run_all_algorithms()
    
    # 打印总结
    print_summary_table(results)
    
    # 可视化
    visualize_comparison(results, problem)
    
    # 保存结果到文件
    import json
    with open('cable-optimization/examples/outputs/11_comparison_results.json', 'w') as f:
        # 转换结果为可序列化格式
        serializable_results = {}
        for name, data in results.items():
            serializable_results[name] = {
                'cost': float(data['cost']) if data['cost'] != float('inf') else 'inf',
                'time': float(data['time']),
                'iterations': str(data.get('iterations', 'N/A'))
            }
        json.dump(serializable_results, f, indent=2)
    
    print("\n✅ 实验完成！结果已保存")
    print("=" * 60)


if __name__ == "__main__":
    main()
