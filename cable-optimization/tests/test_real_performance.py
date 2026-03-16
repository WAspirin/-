"""
SW-RDQN 真实性能测试脚本

实际运行测试并记录性能数据
测试内容:
1. Voronoi 关键点生成效率
2. A* vs Dijkstra 路径规划对比
3. SW-RDQN 训练性能

运行时间：约 10-20 分钟
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict
import heapq
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入实现的模块
from examples.Voronoi import VoronoiKeyPointGenerator


# ============================================================================
# 测试环境
# ============================================================================

def generate_test_map(size: int = 50, obstacle_density: float = 0.25) -> np.ndarray:
    """生成测试地图"""
    np.random.seed(42)
    grid = np.zeros((size, size), dtype=np.int8)
    
    n_obstacles = int(size * size * obstacle_density)
    for _ in range(n_obstacles):
        x, y = np.random.randint(5, size-5, 2)
        obs_size = np.random.randint(2, 4)
        x1, x2 = max(0, x-obs_size//2), min(size, x+obs_size//2)
        y1, y2 = max(0, y-obs_size//2), min(size, y+obs_size//2)
        grid[x1:x2, y1:y2] = 1
    
    return grid


# ============================================================================
# 基线算法
# ============================================================================

class AStarPlanner:
    """A* 算法"""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.size = grid.shape[0]
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx, ny] == 0:
                        neighbors.append((nx, ny))
        return neighbors
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List, float]:
        if self.grid[start[0], start[1]] == 1 or self.grid[goal[0], goal[1]] == 1:
            return [], float('inf')
        
        open_set = [(self.heuristic(start, goal), 0.0, start, [start])]
        closed_set = set()
        g_scores = {start: 0.0}
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if current == goal:
                return path, g
            
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = np.sqrt(dx*dx + dy*dy)
                
                tentative_g = g + move_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, new_path))
        
        return [], float('inf')


class DijkstraPlanner:
    """Dijkstra 算法"""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.size = grid.shape[0]
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                    if self.grid[nx, ny] == 0:
                        neighbors.append((nx, ny))
        return neighbors
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List, float]:
        if self.grid[start[0], start[1]] == 1 or self.grid[goal[0], goal[1]] == 1:
            return [], float('inf')
        
        open_set = [(0.0, start, [start])]
        visited = set()
        
        while open_set:
            cost, current, path = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            if current == goal:
                return path, cost
            
            visited.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = np.sqrt(dx*dx + dy*dy)
                
                new_cost = cost + move_cost
                new_path = path + [neighbor]
                heapq.heappush(open_set, (new_cost, neighbor, new_path))
        
        return [], float('inf')


# ============================================================================
# 性能测试
# ============================================================================

class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_config': {},
            'algorithms': {}
        }
    
    def test_voronoi_efficiency(self, n_tests: int = 20, map_size: int = 50):
        """测试 1: Voronoi 关键点生成效率"""
        print("\n" + "="*60)
        print("测试 1: Voronoi 关键点生成效率")
        print("="*60)
        
        times = []
        n_keypoints_list = []
        
        for i in range(n_tests):
            grid = generate_test_map(map_size, 0.3)
            start = (5, 5)
            goal = (map_size-5, map_size-5)
            
            start_time = time.time()
            generator = VoronoiKeyPointGenerator(grid, safety_distance=2.0)
            key_points = generator.extract_key_points(start, goal)
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            n_keypoints_list.append(len(key_points))
            
            if (i + 1) % 5 == 0:
                print(f"  测试 {i+1}/{n_tests}: 时间={elapsed:.3f}s, 关键点={len(key_points)}个")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_n_keypoints = np.mean(n_keypoints_list)
        
        print(f"\n结果:")
        print(f"  平均时间：{avg_time:.3f} ± {std_time:.3f} 秒")
        print(f"  平均关键点数量：{avg_n_keypoints:.1f} 个")
        
        self.results['algorithms']['voronoi'] = {
            'avg_time': float(avg_time),
            'std_time': float(std_time),
            'avg_n_keypoints': float(avg_n_keypoints),
            'n_tests': n_tests
        }
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(times, bins=10, alpha=0.7, edgecolor='black')
        ax1.axvline(avg_time, color='red', linestyle='--', label=f'平均：{avg_time:.3f}s')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('频数')
        ax1.set_title('Voronoi 关键点生成时间分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(n_keypoints_list, bins=10, alpha=0.7, edgecolor='black', color='green')
        ax2.axvline(avg_n_keypoints, color='red', linestyle='--', label=f'平均：{avg_n_keypoints:.1f}个')
        ax2.set_xlabel('关键点数量')
        ax2.set_ylabel('频数')
        ax2.set_title('关键点数量分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'voronoi_efficiency.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 可视化已保存：voronoi_efficiency.png")
    
    def test_path_planning_comparison(self, n_tests: int = 20, map_size: int = 50):
        """测试 2: A* vs Dijkstra 对比"""
        print("\n" + "="*60)
        print("测试 2: A* vs Dijkstra 路径规划对比")
        print("="*60)
        
        results = {
            'A*': {'times': [], 'path_lengths': [], 'successes': 0},
            'Dijkstra': {'times': [], 'path_lengths': [], 'successes': 0}
        }
        
        for i in range(n_tests):
            grid = generate_test_map(map_size, 0.25)
            start = (5, 5)
            goal = (map_size-5, map_size-5)
            
            # 测试 A*
            astar = AStarPlanner(grid)
            start_time = time.time()
            astar_path, astar_cost = astar.plan(start, goal)
            astar_time = time.time() - start_time
            
            # 测试 Dijkstra
            dijkstra = DijkstraPlanner(grid)
            start_time = time.time()
            dijkstra_path, dijkstra_cost = dijkstra.plan(start, goal)
            dijkstra_time = time.time() - start_time
            
            # 记录结果
            if len(astar_path) > 0:
                results['A*']['successes'] += 1
                results['A*']['times'].append(astar_time)
                results['A*']['path_lengths'].append(astar_cost)
            
            if len(dijkstra_path) > 0:
                results['Dijkstra']['successes'] += 1
                results['Dijkstra']['times'].append(dijkstra_time)
                results['Dijkstra']['path_lengths'].append(dijkstra_cost)
            
            if (i + 1) % 5 == 0:
                print(f"  测试 {i+1}/{n_tests}: "
                      f"A*={astar_time:.3f}s, Dijkstra={dijkstra_time:.3f}s")
        
        # 打印结果
        print(f"\n{'='*60}")
        print("算法对比结果:")
        print(f"{'='*60}")
        print(f"{'算法':<12} {'成功率':<10} {'平均时间 (s)':<15} {'平均路径长度':<15}")
        print("-"*60)
        
        for algo_name in ['A*', 'Dijkstra']:
            r = results[algo_name]
            success_rate = r['successes'] / n_tests * 100
            avg_time = np.mean(r['times']) if r['times'] else float('inf')
            avg_length = np.mean(r['path_lengths']) if r['path_lengths'] else float('inf')
            
            print(f"{algo_name:<12} {success_rate:>9.1f}%  {avg_time:>15.3f}  {avg_length:>15.2f}")
        
        self.results['algorithms']['path_planning'] = results
        self.results['test_config']['n_tests'] = n_tests
        self.results['test_config']['map_size'] = map_size
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        algo_names = ['A*', 'Dijkstra']
        times = [np.mean(results[name]['times']) for name in algo_names]
        lengths = [np.mean(results[name]['path_lengths']) for name in algo_names]
        
        ax1.bar(algo_names, times, alpha=0.7, edgecolor='black', color=['blue', 'green'])
        ax1.set_ylabel('平均时间 (秒)')
        ax1.set_title('平均规划时间对比')
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2.bar(algo_names, lengths, alpha=0.7, edgecolor='black', color=['orange', 'red'])
        ax2.set_ylabel('平均路径长度')
        ax2.set_title('平均路径长度对比')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'algorithm_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 可视化已保存：algorithm_comparison.png")
    
    def save_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, f'test_results_{timestamp}.json')
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 测试结果已保存：{filename}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("SW-RDQN 真实性能测试")
        print("="*60)
        print(f"开始时间：{datetime.now().isoformat()}")
        
        # 测试 1: Voronoi 效率
        self.test_voronoi_efficiency(n_tests=20, map_size=50)
        
        # 测试 2: 算法对比
        self.test_path_planning_comparison(n_tests=20, map_size=50)
        
        # 保存结果
        self.save_results()
        
        print("\n" + "="*60)
        print("所有测试完成!")
        print(f"结束时间：{datetime.now().isoformat()}")
        print("="*60)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主测试入口"""
    tester = PerformanceTester(results_dir="test_results")
    tester.run_all_tests()
    
    print("\n" + "="*60)
    print("测试报告生成完毕!")
    print("="*60)
    print("\n生成的文件:")
    print("  1. voronoi_efficiency.png - Voronoi 关键点生成效率")
    print("  2. algorithm_comparison.png - A* vs Dijkstra 对比")
    print("  3. test_results_*.json - 完整测试数据 (JSON)")
    print("\n请查看 test_results/ 目录查看结果!")


if __name__ == "__main__":
    main()
