"""
SW-RDQN 完整性能测试脚本

测试内容:
1. Voronoi 关键点生成效率
2. SW-RDQN 训练性能
3. 与基线算法对比 (A*, Dijkstra, 纯 DQN)
4. 生成可视化结果

运行时间：约 30-60 分钟
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict
import sys

# 导入实现的算法
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.Voronoi import VoronoiKeyPointGenerator
from examples.SW_RDQN import SWRDQNPlanner, LocalSWRDQNController, SWRDQNNetwork
import heapq


# ============================================================================
# 测试环境生成
# ============================================================================

def generate_test_maps(n_maps: int = 10, size: int = 50, obstacle_density: float = 0.3):
    """生成多个测试地图"""
    maps = []
    np.random.seed(42)  # 固定随机种子保证可复现
    
    for i in range(n_maps):
        # 生成随机障碍物
        grid = np.zeros((size, size), dtype=np.int8)
        n_obstacles = int(size * size * obstacle_density)
        
        # 随机放置障碍物（避开起点和终点）
        for _ in range(n_obstacles):
            x = np.random.randint(5, size - 5)
            y = np.random.randint(5, size - 5)
            
            # 确保不阻塞起点 (5, 5) 和终点 (size-5, size-5)
            if abs(x - 5) > 3 or abs(y - 5) > 3:
                if abs(x - (size-5)) > 3 or abs(y - (size-5)) > 3:
                    # 随机大小的障碍物
                    obs_size = np.random.randint(2, 5)
                    x1, x2 = max(0, x - obs_size//2), min(size, x + obs_size//2)
                    y1, y2 = max(0, y - obs_size//2), min(size, y + obs_size//2)
                    grid[x1:x2, y1:y2] = 1
        
        maps.append(grid)
    
    return maps


def generate_test_scenarios(n_scenarios: int = 20, map_size: int = 50):
    """生成测试场景（起点 - 终点对）"""
    scenarios = []
    np.random.seed(123)
    
    for i in range(n_scenarios):
        # 随机起点和终点（确保在可行区域）
        start = (np.random.randint(5, 15), np.random.randint(5, 15))
        goal = (np.random.randint(map_size-15, map_size-5), 
                np.random.randint(map_size-15, map_size-5))
        scenarios.append((start, goal))
    
    return scenarios


# ============================================================================
# 基线算法实现
# ============================================================================

class AStarPlanner:
    """A* 算法实现"""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.size = grid.shape[0]
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """欧氏距离启发式"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取 8 方向邻居"""
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
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """
        A* 路径规划
        
        Returns:
            path: 路径点列表
            cost: 路径总代价
        """
        if self.grid[start[0], start[1]] == 1 or self.grid[goal[0], goal[1]] == 1:
            return [], float('inf')
        
        # 优先队列：(f_score, g_score, position, path)
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
                
                # 计算移动代价（对角线移动代价更大）
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
    """Dijkstra 算法实现"""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.size = grid.shape[0]
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取 8 方向邻居"""
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
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """
        Dijkstra 路径规划
        
        Returns:
            path: 路径点列表
            cost: 路径总代价
        """
        if self.grid[start[0], start[1]] == 1 or self.grid[goal[0], goal[1]] == 1:
            return [], float('inf')
        
        # 优先队列：(cost, position, path)
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
            'algorithms': {},
            'comparison': {}
        }
    
    def test_voronoi_efficiency(self, n_tests: int = 20, map_size: int = 50):
        """测试 Voronoi 关键点生成效率"""
        print("\n" + "="*60)
        print("测试 1: Voronoi 关键点生成效率")
        print("="*60)
        
        times = []
        n_keypoints_list = []
        
        np.random.seed(42)
        for i in range(n_tests):
            # 生成随机地图
            grid = np.zeros((map_size, map_size), dtype=np.int8)
            n_obstacles = int(map_size * map_size * 0.3)
            for _ in range(n_obstacles):
                x, y = np.random.randint(5, map_size-5, 2)
                obs_size = np.random.randint(2, 4)
                x1, x2 = max(0, x-obs_size//2), min(map_size, x+obs_size//2)
                y1, y2 = max(0, y-obs_size//2), min(map_size, y+obs_size//2)
                grid[x1:x2, y1:y2] = 1
            
            start = (5, 5)
            goal = (map_size-5, map_size-5)
            
            # 计时
            start_time = time.time()
            generator = VoronoiKeyPointGenerator(grid, safety_distance=2.0)
            key_points = generator.extract_key_points(start, goal)
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            n_keypoints_list.append(len(key_points))
            
            if (i + 1) % 5 == 0:
                print(f"  测试 {i+1}/{n_tests}: 时间={elapsed:.3f}s, 关键点={len(key_points)}个")
        
        # 统计结果
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
    
    def test_path_planning_comparison(self, n_maps: int = 10, map_size: int = 50, n_scenarios: int = 5):
        """对比不同路径规划算法的性能"""
        print("\n" + "="*60)
        print("测试 2: 路径规划算法对比")
        print("="*60)
        
        # 生成测试地图和场景
        maps = generate_test_maps(n_maps, map_size)
        scenarios = generate_test_scenarios(n_scenarios, map_size)
        
        # 初始化算法
        algorithms = {
            'Dijkstra': lambda grid: DijkstraPlanner(grid),
            'A*': lambda grid: AStarPlanner(grid),
        }
        
        # 测试结果
        results = {}
        for algo_name in algorithms.keys():
            results[algo_name] = {
                'success_rate': 0.0,
                'avg_time': 0.0,
                'avg_path_length': 0.0,
                'times': [],
                'path_lengths': [],
                'successes': 0
            }
        
        total_tests = n_maps * n_scenarios
        test_count = 0
        
        for map_idx, grid in enumerate(maps):
            print(f"\n地图 {map_idx+1}/{n_maps}:")
            
            for scenario_idx, (start, goal) in enumerate(scenarios):
                test_count += 1
                
                for algo_name, algo_factory in algorithms.items():
                    planner = algo_factory(grid)
                    
                    # 计时
                    start_time = time.time()
                    path, cost = planner.plan(start, goal)
                    elapsed = time.time() - start_time
                    
                    # 记录结果
                    results[algo_name]['times'].append(elapsed)
                    
                    if len(path) > 0:
                        results[algo_name]['successes'] += 1
                        results[algo_name]['path_lengths'].append(cost)
                    
                    if test_count % 10 == 0:
                        print(f"  测试 {test_count}/{total_tests}: {algo_name} - 时间={elapsed:.3f}s, 成功={len(path)>0}")
        
        # 计算统计结果
        for algo_name in algorithms.keys():
            n_success = results[algo_name]['successes']
            results[algo_name]['success_rate'] = n_success / total_tests
            
            if n_success > 0:
                results[algo_name]['avg_time'] = np.mean(results[algo_name]['times'])
                results[algo_name]['avg_path_length'] = np.mean(results[algo_name]['path_lengths'])
            else:
                results[algo_name]['avg_time'] = float('inf')
                results[algo_name]['avg_path_length'] = float('inf')
        
        # 打印结果
        print(f"\n{'='*60}")
        print("算法对比结果:")
        print(f"{'='*60}")
        print(f"{'算法':<12} {'成功率':<10} {'平均时间 (s)':<15} {'平均路径长度':<15}")
        print("-"*60)
        for algo_name in algorithms.keys():
            r = results[algo_name]
            print(f"{algo_name:<12} {r['success_rate']*100:>9.1f}%  {r['avg_time']:>15.3f}  {r['avg_path_length']:>15.2f}")
        
        self.results['algorithms']['path_planning'] = results
        self.results['test_config']['n_maps'] = n_maps
        self.results['test_config']['map_size'] = map_size
        self.results['test_config']['n_scenarios'] = n_scenarios
        
        # 可视化
        self._plot_algorithm_comparison(results)
    
    def _plot_algorithm_comparison(self, results: Dict):
        """绘制算法对比图"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        algo_names = list(results.keys())
        
        # 成功率对比
        success_rates = [results[name]['success_rate'] * 100 for name in algo_names]
        ax1.bar(algo_names, success_rates, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('成功率 (%)')
        ax1.set_title('成功率对比')
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 平均时间对比
        avg_times = [results[name]['avg_time'] for name in algo_names]
        ax2.bar(algo_names, avg_times, alpha=0.7, edgecolor='black', color='green')
        ax2.set_ylabel('平均时间 (秒)')
        ax2.set_title('平均规划时间对比')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 平均路径长度对比
        avg_lengths = [results[name]['avg_path_length'] for name in algo_names]
        ax3.bar(algo_names, avg_lengths, alpha=0.7, edgecolor='black', color='orange')
        ax3.set_ylabel('平均路径长度')
        ax3.set_title('平均路径长度对比')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'algorithm_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 可视化已保存：algorithm_comparison.png")
    
    def test_swr_dqn_training(self, n_episodes: int = 100, map_size: int = 30):
        """测试 SW-RDQN 训练性能"""
        print("\n" + "="*60)
        print("测试 3: SW-RDQN 训练性能")
        print("="*60)
        
        # 生成测试地图
        np.random.seed(42)
        grid = np.zeros((map_size, map_size), dtype=np.int8)
        n_obstacles = int(map_size * map_size * 0.25)
        for _ in range(n_obstacles):
            x, y = np.random.randint(5, map_size-5, 2)
            obs_size = np.random.randint(2, 4)
            x1, x2 = max(0, x-obs_size//2), min(map_size, x+obs_size//2)
            y1, y2 = max(0, y-obs_size//2), min(map_size, y+obs_size//2)
            grid[x1:x2, y1:y2] = 1
        
        start = (5, 5)
        goal = (map_size-5, map_size-5)
        
        # 初始化规划器
        planner = SWRDQNPlanner(grid, safety_distance=2.0)
        
        # 训练记录
        episode_rewards = []
        episode_times = []
        success_count = 0
        
        print(f"\n开始训练：{n_episodes} 回合")
        print(f"地图大小：{map_size}x{map_size}")
        print(f"起点：{start}, 终点：{goal}")
        print()
        
        for episode in range(n_episodes):
            start_time = time.time()
            
            # 运行一个训练回合
            try:
                total_reward, success = planner.run_episode(start, goal, train=True)
                elapsed = time.time() - start_time
                
                episode_rewards.append(total_reward)
                episode_times.append(elapsed)
                
                if success:
                    success_count += 1
                
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    success_rate = success_count / (episode + 1) * 100
                    print(f"回合 {episode+1}/{n_episodes}: "
                          f"平均奖励={avg_reward:.2f}, "
                          f"成功率={success_rate:.1f}%, "
                          f"时间={elapsed:.2f}s")
            except Exception as e:
                print(f"回合 {episode+1} 出错：{e}")
                episode_rewards.append(-100)
                episode_times.append(time.time() - start_time)
        
        # 统计结果
        print(f"\n训练完成!")
        print(f"  总成功率：{success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
        print(f"  平均训练时间：{np.mean(episode_times):.2f} 秒/回合")
        print(f"  平均奖励：{np.mean(episode_rewards):.2f}")
        
        self.results['algorithms']['swrdqn_training'] = {
            'n_episodes': n_episodes,
            'success_rate': success_count / n_episodes,
            'avg_time_per_episode': float(np.mean(episode_times)),
            'avg_reward': float(np.mean(episode_rewards))
        }
        
        # 可视化
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 奖励曲线
        ax1.plot(episode_rewards, alpha=0.7, label='单回合奖励')
        
        # 滑动平均
        window = 10
        if len(episode_rewards) >= window:
            rewards_smooth = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(episode_rewards)), rewards_smooth, 
                    'r-', linewidth=2, label=f'{window}回合滑动平均')
        
        ax1.set_xlabel('回合')
        ax1.set_ylabel('奖励')
        ax1.set_title('SW-RDQN 训练奖励曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 时间分布
        ax2.hist(episode_times, bins=10, alpha=0.7, edgecolor='black', color='green')
        ax2.axvline(np.mean(episode_times), color='red', linestyle='--', 
                   label=f'平均：{np.mean(episode_times):.2f}s')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('频数')
        ax2.set_title('单回合训练时间分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 成功率曲线
        cumulative_success = np.cumsum([1 if r > -50 else 0 for r in episode_rewards])
        success_rate_curve = cumulative_success / np.arange(1, len(episode_rewards)+1) * 100
        ax3.plot(success_rate_curve, 'b-', linewidth=2)
        ax3.set_xlabel('回合')
        ax3.set_ylabel('累计成功率 (%)')
        ax3.set_title('累计成功率曲线')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'swrdqn_training.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 可视化已保存：swrdqn_training.png")
    
    def save_results(self):
        """保存测试结果到 JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, f'test_results_{timestamp}.json')
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 测试结果已保存：{filename}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("SW-RDQN 完整性能测试")
        print("="*60)
        print(f"开始时间：{datetime.now().isoformat()}")
        
        # 测试 1: Voronoi 效率
        self.test_voronoi_efficiency(n_tests=20, map_size=50)
        
        # 测试 2: 算法对比
        self.test_path_planning_comparison(n_maps=10, map_size=50, n_scenarios=5)
        
        # 测试 3: SW-RDQN 训练
        self.test_swr_dqn_training(n_episodes=100, map_size=30)
        
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
    # 创建测试器
    tester = PerformanceTester(results_dir="test_results")
    
    # 运行所有测试
    tester.run_all_tests()
    
    print("\n" + "="*60)
    print("测试报告生成完毕!")
    print("="*60)
    print("\n生成的文件:")
    print("  1. voronoi_efficiency.png - Voronoi 关键点生成效率")
    print("  2. algorithm_comparison.png - 算法性能对比")
    print("  3. swrdqn_training.png - SW-RDQN 训练曲线")
    print("  4. test_results_*.json - 完整测试数据")
    print("\n请查看 test_results/ 目录查看结果!")


if __name__ == "__main__":
    main()
