#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模性能基准测试 (简化版)

测试不同规模问题下各算法的性能表现
- 100 节点
- 200 节点  
- 500 节点

对比算法:
- 分解算法 (简化版，使用简单聚类)
- VNS
- 遗传算法
- 贪心最近邻 (基准)
"""

import numpy as np
import time
import sys
import os
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_distance_matrix(coordinates):
    """计算欧氏距离矩阵"""
    n = len(coordinates)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
    return dist_matrix


def compute_path_cost(path, dist_matrix):
    """计算路径总成本"""
    cost = 0.0
    for i in range(len(path) - 1):
        cost += dist_matrix[path[i], path[i+1]]
    return cost


def greedy_nearest_neighbor(dist_matrix, start=0):
    """贪心最近邻算法 (基准)"""
    n = len(dist_matrix)
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    for _ in range(n - 1):
        current = path[-1]
        nearest = -1
        nearest_dist = float('inf')
        
        for j in range(n):
            if not visited[j] and dist_matrix[current, j] < nearest_dist:
                nearest = j
                nearest_dist = dist_matrix[current, j]
        
        path.append(nearest)
        visited[nearest] = True
    
    return path, compute_path_cost(path, dist_matrix)


def simple_kmeans(coordinates, k, random_seed=42):
    """简化版 K-Means 聚类 (不依赖 sklearn)"""
    np.random.seed(random_seed)
    n = len(coordinates)
    
    # 随机初始化中心
    indices = np.random.choice(n, k, replace=False)
    centers = coordinates[indices].copy()
    
    labels = np.zeros(n, dtype=int)
    
    for _ in range(20):  # 最多 20 次迭代
        # 分配点到最近中心
        for i in range(n):
            distances = [np.sum((coordinates[i] - centers[c])**2) for c in range(k)]
            labels[i] = np.argmin(distances)
        
        # 更新中心
        new_centers = np.zeros_like(centers)
        for c in range(k):
            cluster_points = coordinates[labels == c]
            if len(cluster_points) > 0:
                new_centers[c] = cluster_points.mean(axis=0)
            else:
                new_centers[c] = centers[c]
        
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return labels, centers


def two_opt_improve(path, dist_matrix, max_iterations=100):
    """2-opt 局部优化"""
    n = len(path)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                old_cost = dist_matrix[path[i-1], path[i]] + dist_matrix[path[j], path[j+1]]
                new_cost = dist_matrix[path[i-1], path[j]] + dist_matrix[path[i], path[j+1]]
                
                if new_cost < old_cost:
                    path[i:j+1] = reversed(path[i:j+1])
                    improved = True
    
    return path, compute_path_cost(path, dist_matrix)


def decomposition_optimizer(coordinates, dist_matrix, n_clusters=10, random_seed=42):
    """简化版分解优化器"""
    np.random.seed(random_seed)
    n = len(coordinates)
    
    # Step 1: 聚类
    labels, centers = simple_kmeans(coordinates, n_clusters, random_seed)
    
    # Step 2: 为每个簇找到边界节点 (离其他簇中心最近的节点)
    clusters = {}
    for i in range(n_clusters):
        clusters[i] = np.where(labels == i)[0].tolist()
    
    # Step 3: 簇间路径优化 (贪心)
    cluster_order = [0]  # 从包含起点 (节点 0) 的簇开始
    remaining = set(range(n_clusters)) - {labels[0]}
    current_cluster = labels[0]
    
    while remaining:
        # 找到离当前簇中心最近的未访问簇
        best_next = None
        best_dist = float('inf')
        
        for c in remaining:
            dist = np.sum((centers[current_cluster] - centers[c])**2)
            if dist < best_dist:
                best_dist = dist
                best_next = c
        
        cluster_order.append(best_next)
        remaining.remove(best_next)
        current_cluster = best_next
    
    # Step 4: 簇内路径优化 (贪心 + 2-opt)
    full_path = []
    for cluster_id in cluster_order:
        cluster_nodes = clusters[cluster_id]
        
        if len(cluster_nodes) == 0:
            continue
        
        # 贪心生成初始路径
        if len(full_path) == 0:
            current = cluster_nodes[0]
        else:
            # 从离上一个簇最近的节点开始
            last_node = full_path[-1]
            distances = [dist_matrix[last_node, n] for n in cluster_nodes]
            current = cluster_nodes[np.argmin(distances)]
        
        cluster_path = [current]
        visited = {current}
        
        while len(cluster_path) < len(cluster_nodes):
            nearest = None
            nearest_dist = float('inf')
            
            for node in cluster_nodes:
                if node not in visited:
                    d = dist_matrix[current, node]
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest = node
            
            cluster_path.append(nearest)
            visited.add(nearest)
            current = nearest
        
        # 2-opt 优化
        cluster_path, _ = two_opt_improve(cluster_path, dist_matrix, max_iterations=50)
        
        # 连接到主路径
        if len(full_path) == 0:
            full_path.extend(cluster_path)
        else:
            # 找到最佳连接点
            best_insert_cost = float('inf')
            best_insert_pos = -1
            
            for i, node in enumerate(cluster_path):
                cost = dist_matrix[full_path[-1], node]
                if cost < best_insert_cost:
                    best_insert_cost = cost
                    best_insert_pos = i
            
            # 重新排列簇路径，从最佳连接点开始
            cluster_path = cluster_path[best_insert_pos:] + cluster_path[:best_insert_pos]
            full_path.extend(cluster_path)
    
    # Step 5: 全局 2-opt 优化
    full_path, total_cost = two_opt_improve(full_path, dist_matrix, max_iterations=100)
    
    return full_path, total_cost, {'n_clusters': n_clusters, 'cluster_labels': labels.tolist()}


def vns_optimizer(coordinates, dist_matrix, max_iterations=300, random_seed=42):
    """简化版 VNS 优化器"""
    np.random.seed(random_seed)
    n = len(dist_matrix)
    
    # 初始解 (最近邻)
    current_path, current_cost = greedy_nearest_neighbor(dist_matrix, start=0)
    best_path = current_path[:]
    best_cost = current_cost
    
    # 邻域操作
    def swap(path, i, j):
        new_path = path[:]
        new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path
    
    def reverse_segment(path, i, j):
        new_path = path[:]
        new_path[i:j+1] = reversed(new_path[i:j+1])
        return new_path
    
    def insert(path, i, j):
        new_path = path[:]
        node = new_path.pop(i)
        new_path.insert(j, node)
        return new_path
    
    # VNS 主循环
    k_max = 5
    k = 1
    no_improve = 0
    
    for iteration in range(max_iterations):
        # 扰动
        new_path = current_path[:]
        
        if k == 1:  # 交换相邻
            i = random.randint(1, n - 3)
            new_path = swap(new_path, i, i + 1)
        elif k == 2:  # 交换任意
            i, j = random.sample(range(1, n - 1), 2)
            new_path = swap(new_path, i, j)
        elif k == 3:  # 逆转
            i, j = sorted(random.sample(range(1, n - 1), 2))
            new_path = reverse_segment(new_path, i, j)
        elif k == 4:  # 插入
            i, j = random.sample(range(1, n - 1), 2)
            new_path = insert(new_path, i, j)
        else:  # 2-opt
            i, j = sorted(random.sample(range(1, n - 1), 2))
            new_path = reverse_segment(new_path, i, j)
        
        # 局部搜索 (最陡下降)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    neighbor = reverse_segment(new_path, i, j)
                    neighbor_cost = compute_path_cost(neighbor, dist_matrix)
                    if neighbor_cost < compute_path_cost(new_path, dist_matrix):
                        new_path = neighbor
                        improved = True
        
        new_cost = compute_path_cost(new_path, dist_matrix)
        
        # 判断是否接受
        if new_cost < best_cost:
            best_path = new_path[:]
            best_cost = new_cost
            current_path = new_path[:]
            k = 1
            no_improve = 0
        else:
            k += 1
            no_improve += 1
            if k > k_max:
                k = 1
            if no_improve > 50:
                break
    
    return best_path, best_cost, {'iterations': iteration + 1}


def ga_optimizer(dist_matrix, population_size=80, generations=150, random_seed=42):
    """简化版遗传算法"""
    np.random.seed(random_seed)
    n = len(dist_matrix)
    
    # 初始化种群
    def random_path():
        path = list(range(n))
        random.shuffle(path)
        # 确保起点是 0
        path[0], path[path.index(0)] = 0, path[0]
        return path
    
    population = [random_path() for _ in range(population_size)]
    
    def fitness(path):
        return compute_path_cost(path, dist_matrix)
    
    def order_crossover(p1, p2):
        """顺序交叉"""
        i, j = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[i:j+1] = p1[i:j+1]
        
        pos = (j + 1) % n
        for gene in p2:
            if gene not in child:
                while child[pos] != -1:
                    pos = (pos + 1) % n
                child[pos] = gene
        
        return child
    
    def swap_mutation(path):
        """交换变异"""
        i, j = random.sample(range(1, n), 2)
        new_path = path[:]
        new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path
    
    # 进化循环
    for gen in range(generations):
        # 评估
        fitnesses = [fitness(ind) for ind in population]
        
        # 精英保留
        elite_indices = np.argsort(fitnesses)[:population_size // 10]
        elites = [population[i] for i in elite_indices]
        
        # 生成新种群
        new_population = elites[:]
        
        while len(new_population) < population_size:
            # 锦标赛选择
            def tournament():
                candidates = random.sample(range(len(population)), 3)
                return population[min(candidates, key=lambda i: fitnesses[i])]
            
            p1 = tournament()
            p2 = tournament()
            
            # 交叉
            if random.random() < 0.85:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            
            # 变异
            if random.random() < 0.15:
                child = swap_mutation(child)
            
            new_population.append(child)
        
        population = new_population
    
    # 返回最优
    best_idx = np.argmin([fitness(ind) for ind in population])
    return population[best_idx], fitness(population[best_idx]), {'generations': generations}


def benchmark_algorithm(algo_name, algo_func, coordinates, dist_matrix):
    """基准测试单个算法"""
    start_time = time.time()
    
    try:
        result = algo_func(coordinates, dist_matrix)
        if len(result) == 3:
            path, cost, details = result
        else:
            path, cost = result
            details = {}
        
        elapsed = time.time() - start_time
        
        return {
            'algorithm': algo_name,
            'cost': cost,
            'time': elapsed,
            'path_length': len(path),
            'success': True,
            'details': details
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'algorithm': algo_name,
            'cost': float('inf'),
            'time': elapsed,
            'error': str(e),
            'success': False
        }


def run_benchmark(n_nodes, random_seed=42):
    """运行完整基准测试"""
    print(f"\n{'='*70}")
    print(f"测试规模：{n_nodes} 节点")
    print(f"{'='*70}\n")
    
    # 生成随机坐标
    np.random.seed(random_seed)
    coordinates = np.random.rand(n_nodes, 2) * 100
    dist_matrix = compute_distance_matrix(coordinates)
    
    results = []
    
    # 1. 贪心最近邻 (基准)
    print("📊 运行贪心最近邻算法...")
    result = benchmark_algorithm(
        "贪心最近邻",
        lambda coords, dm: (lambda p, c: (p, c))(*greedy_nearest_neighbor(dm, start=0)),
        coordinates, dist_matrix
    )
    results.append(result)
    print(f"   成本：{result['cost']:.2f}, 时间：{result['time']:.4f}秒")
    
    # 2. 分解算法
    print("📊 运行分解算法...")
    n_clusters = max(5, int(np.sqrt(n_nodes / 2)))
    result = benchmark_algorithm(
        f"分解算法 (k={n_clusters})",
        lambda coords, dm: decomposition_optimizer(coords, dm, n_clusters=n_clusters, random_seed=random_seed),
        coordinates, dist_matrix
    )
    results.append(result)
    if result['success']:
        print(f"   成本：{result['cost']:.2f}, 时间：{result['time']:.2f}秒")
    else:
        print(f"   ❌ 错误：{result.get('error', 'Unknown')}")
    
    # 3. VNS (仅用于中小规模)
    if n_nodes <= 100:
        print("📊 运行 VNS 算法...")
        result = benchmark_algorithm(
            "VNS",
            lambda coords, dm: vns_optimizer(coords, dm, max_iterations=300, random_seed=random_seed),
            coordinates, dist_matrix
        )
        results.append(result)
        if result['success']:
            print(f"   成本：{result['cost']:.2f}, 时间：{result['time']:.2f}秒")
        else:
            print(f"   ❌ 错误：{result.get('error', 'Unknown')}")
    
    # 4. 遗传算法 (仅用于中小规模)
    if n_nodes <= 100:
        print("📊 运行遗传算法...")
        result = benchmark_algorithm(
            "遗传算法",
            lambda coords, dm: ga_optimizer(dm, population_size=80, generations=150, random_seed=random_seed),
            coordinates, dist_matrix
        )
        results.append(result)
        if result['success']:
            print(f"   成本：{result['cost']:.2f}, 时间：{result['time']:.2f}秒")
        else:
            print(f"   ❌ 错误：{result.get('error', 'Unknown')}")
    
    return results, coordinates, dist_matrix


def print_summary_table(all_results):
    """打印汇总表格"""
    print(f"\n{'='*80}")
    print("性能基准测试汇总")
    print(f"{'='*80}\n")
    
    for n_nodes, results, _, _ in all_results:
        print(f"📍 {n_nodes} 节点规模:\n")
        print(f"{'算法':<30} {'成本':<15} {'时间 (s)':<12} {'状态':<8}")
        print("-" * 65)
        
        sorted_results = sorted(results, key=lambda x: x['cost'] if x['success'] else float('inf'))
        for res in sorted_results:
            status = "✅" if res['success'] else "❌"
            cost_str = f"{res['cost']:.2f}" if res['success'] else "N/A"
            print(f"{res['algorithm']:<30} {cost_str:<15} {res['time']:<12.4f} {status:<8}")
        
        print()


def main():
    """主函数"""
    print("="*70)
    print("线缆布线优化 - 大规模性能基准测试")
    print("="*70)
    
    test_scales = [100, 200, 500]
    all_results = []
    
    for n_nodes in test_scales:
        results, coordinates, dist_matrix = run_benchmark(n_nodes)
        all_results.append((n_nodes, results, coordinates, dist_matrix))
    
    # 打印汇总
    print_summary_table(all_results)
    
    # 保存结果
    print("\n💾 保存测试结果...")
    summary = {
        'test_scales': test_scales,
        'results': [
            {
                'n_nodes': n,
                'results': [
                    {k: v for k, v in r.items() if k != 'details'}
                    for r in res
                ]
            }
            for n, res, _, _ in all_results
        ]
    }
    
    import json
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'benchmark_results.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存到：{output_path}")
    
    # 生成可视化图表
    try:
        generate_visualization(all_results, output_dir)
    except Exception as e:
        print(f"⚠️  可视化生成失败：{e}")


def generate_visualization(all_results, output_dir):
    """生成可视化图表"""
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 收集数据
    algo_names = set()
    for n_nodes, results, _, _ in all_results:
        for res in results:
            if res['success']:
                algo_names.add(res['algorithm'])
    
    algo_names = sorted(algo_names)
    colors = plt.cm.tab10(np.linspace(0, 1, len(algo_names)))
    color_map = {name: colors[i] for i, name in enumerate(algo_names)}
    
    # 图 1: 成本对比
    x_positions = np.arange(len(all_results))
    width = 0.2
    
    for i, algo_name in enumerate(algo_names):
        costs = []
        positions = []
        for j, (n_nodes, results, _, _) in enumerate(all_results):
            for res in results:
                if res['algorithm'] == algo_name and res['success']:
                    costs.append(res['cost'])
                    positions.append(j + i * width - len(algo_names) * width / 2 + width / 2)
                    break
        
        if costs:
            ax1.bar(positions, costs, width, label=algo_name, color=color_map[algo_name])
    
    ax1.set_xlabel('问题规模 (节点数)')
    ax1.set_ylabel('路径成本')
    ax1.set_title('算法成本对比')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{n}节点' for n, _, _, _ in all_results])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 图 2: 时间对比 (对数尺度)
    for i, algo_name in enumerate(algo_names):
        times = []
        positions = []
        for j, (n_nodes, results, _, _) in enumerate(all_results):
            for res in results:
                if res['algorithm'] == algo_name and res['success']:
                    times.append(res['time'])
                    positions.append(j + i * width - len(algo_names) * width / 2 + width / 2)
                    break
        
        if times:
            ax2.bar(positions, times, width, label=algo_name, color=color_map[algo_name])
    
    ax2.set_xlabel('问题规模 (节点数)')
    ax2.set_ylabel('计算时间 (秒)')
    ax2.set_title('算法时间对比 (对数尺度)')
    ax2.set_yscale('log')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'{n}节点' for n, _, _, _ in all_results])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'benchmark_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化图表已保存到：{output_path}")
    plt.close()


if __name__ == '__main__':
    main()
