"""
搜索算法 - A* 搜索 (A-Star Search)

问题描述:
使用 A* 算法求解带启发式的最短路径问题，应用于线缆布线路径规划

算法原理:
1. 初始化：开放列表 OpenSet = {起点}, 关闭列表 ClosedSet = {}
2. 当 OpenSet 非空:
   a. 选择 f(n) = g(n) + h(n) 最小的节点 n
   b. 如果 n 是终点，返回路径
   c. 将 n 从 OpenSet 移到 ClosedSet
   d. 对于 n 的每个邻居 m:
      - 如果 m 在 ClosedSet 中，跳过
      - 计算 tentative_g = g(n) + cost(n, m)
      - 如果 m 不在 OpenSet 或 tentative_g 更小:
        * 更新 m 的父节点为 n
        * 更新 g(m) = tentative_g
        * 更新 f(m) = g(m) + h(m)
        * 如果 m 不在 OpenSet，加入

评估函数:
- g(n): 从起点到 n 的实际代价
- h(n): 从 n 到终点的启发式估计（必须可采纳，不高估）
- f(n) = g(n) + h(n): 总估计代价

常用启发式:
- 欧氏距离：h(n) = ||n - goal||
- 曼哈顿距离：h(n) = |x1-x2| + |y1-y2|
- 切比雪夫距离：h(n) = max(|x1-x2|, |y1-y2|)
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Node:
    """A* 搜索节点"""
    position: Tuple[int, int]
    g: float = 0  # 从起点到当前节点的实际代价
    h: float = 0  # 启发式估计代价
    f: float = 0  # 总代价 f = g + h
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        return self.f < other.f


class AStarSearch:
    """A* 搜索求解器"""
    
    def __init__(self, 
                 heuristic: str = 'euclidean',
                 allow_diagonal: bool = True):
        self.heuristic = heuristic
        self.allow_diagonal = allow_diagonal
        
        # 搜索方向
        if allow_diagonal:
            self.directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),  # 四方向
                (-1, -1), (-1, 1), (1, -1), (1, 1)  # 对角线
            ]
            self.move_costs = {
                (0, 1): 1.0, (0, -1): 1.0, (1, 0): 1.0, (-1, 0): 1.0,
                (1, 1): 1.414, (1, -1): 1.414, (-1, 1): 1.414, (-1, -1): 1.414
            }
        else:
            self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            self.move_costs = {(0, 1): 1.0, (0, -1): 1.0, (1, 0): 1.0, (-1, 0): 1.0}
        
        self.open_set: List[Node] = []
        self.closed_set: Set[Tuple[int, int]] = set()
        self.g_scores: Dict[Tuple[int, int], float] = {}
        self.came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        self.searched_nodes = 0
        self.expanded_nodes = 0
    
    def heuristic_function(self, 
                          pos: Tuple[int, int], 
                          goal: Tuple[int, int]) -> float:
        """计算启发式值"""
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        
        if self.heuristic == 'euclidean':
            return np.sqrt(dx**2 + dy**2)
        elif self.heuristic == 'manhattan':
            return dx + dy
        elif self.heuristic == 'chebyshev':
            return max(dx, dy)
        elif self.heuristic == 'octile':
            # 八方向移动的精确启发式
            return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)
        else:
            return np.sqrt(dx**2 + dy**2)
    
    def get_neighbors(self, 
                     pos: Tuple[int, int], 
                     grid: np.ndarray) -> List[Tuple[Tuple[int, int], float]]:
        """获取可行邻居节点及移动代价"""
        rows, cols = grid.shape
        neighbors = []
        
        for direction in self.directions:
            new_pos = (pos[0] + direction[0], pos[1] + direction[1])
            
            # 边界检查
            if 0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols:
                # 障碍物检查（grid 值为 1 表示障碍）
                if grid[new_pos] == 0:
                    move_cost = self.move_costs.get(direction, 1.0)
                    
                    # 对角线移动时检查是否穿过障碍
                    if abs(direction[0]) + abs(direction[1]) == 2:
                        # 检查两个相邻方向是否有障碍
                        adj1 = (pos[0] + direction[0], pos[1])
                        adj2 = (pos[0], pos[1] + direction[1])
                        if grid[adj1] == 0 or grid[adj2] == 0:
                            continue
                    
                    neighbors.append((new_pos, move_cost))
        
        return neighbors
    
    def search(self, 
               grid: np.ndarray,
               start: Tuple[int, int],
               goal: Tuple[int, int],
               verbose: bool = False) -> Optional[List[Tuple[int, int]]]:
        """
        执行 A* 搜索
        
        Args:
            grid: 网格地图（0=可通行，1=障碍）
            start: 起点坐标
            goal: 终点坐标
            verbose: 是否打印搜索信息
        
        Returns:
            路径列表 [(x1,y1), (x2,y2), ...]，如果无路径返回 None
        """
        # 重置状态
        self.open_set = []
        self.closed_set = set()
        self.g_scores = defaultdict(lambda: float('inf'))
        self.came_from = {}
        self.searched_nodes = 0
        self.expanded_nodes = 0
        
        # 起点和终点检查
        if grid[start] == 1 or grid[goal] == 1:
            print("起点或终点在障碍物中!")
            return None
        
        # 初始化起点
        start_h = self.heuristic_function(start, goal)
        start_node = Node(position=start, g=0, h=start_h, f=start_h)
        
        heapq.heappush(self.open_set, start_node)
        self.g_scores[start] = 0
        
        while self.open_set:
            # 取出 f 值最小的节点
            current = heapq.heappop(self.open_set)
            current_pos = current.position
            
            self.searched_nodes += 1
            
            # 到达终点
            if current_pos == goal:
                return self.reconstruct_path(current_pos)
            
            # 已访问过（有更优路径）
            if current_pos in self.closed_set:
                continue
            
            self.closed_set.add(current_pos)
            self.expanded_nodes += 1
            
            # 扩展邻居
            for neighbor_pos, move_cost in self.get_neighbors(current_pos, grid):
                if neighbor_pos in self.closed_set:
                    continue
                
                tentative_g = self.g_scores[current_pos] + move_cost
                
                if tentative_g < self.g_scores[neighbor_pos]:
                    # 找到更优路径
                    self.came_from[neighbor_pos] = current_pos
                    self.g_scores[neighbor_pos] = tentative_g
                    
                    h = self.heuristic_function(neighbor_pos, goal)
                    f = tentative_g + h
                    
                    neighbor_node = Node(
                        position=neighbor_pos,
                        g=tentative_g,
                        h=h,
                        f=f,
                        parent=current
                    )
                    heapq.heappush(self.open_set, neighbor_node)
            
            if verbose and self.searched_nodes % 500 == 0:
                print(f"搜索中... 已访问 {self.searched_nodes} 节点，"
                      f"OpenSet 大小：{len(self.open_set)}")
        
        # 无路径
        return None
    
    def reconstruct_path(self, goal_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重构路径"""
        path = [goal_pos]
        current = goal_pos
        
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def get_search_stats(self) -> Dict:
        """获取搜索统计信息"""
        return {
            'searched_nodes': self.searched_nodes,
            'expanded_nodes': self.expanded_nodes,
            'closed_set_size': len(self.closed_set)
        }


def test_astar_simple():
    """简单的 A* 测试"""
    print("=" * 60)
    print("A* 测试 - 简单网格")
    print("=" * 60)
    
    # 创建简单网格
    grid = np.zeros((10, 10), dtype=int)
    
    # 添加一些障碍
    grid[2, 2:5] = 1
    grid[5, 5:8] = 1
    grid[7, 3] = 1
    
    start = (0, 0)
    goal = (9, 9)
    
    # 执行搜索
    astar = AStarSearch(heuristic='euclidean', allow_diagonal=True)
    path = astar.search(grid, start, goal, verbose=True)
    
    if path:
        print(f"\n找到路径! 长度：{len(path)} 步")
        print(f"搜索节点数：{astar.searched_nodes}")
        print(f"扩展节点数：{astar.expanded_nodes}")
        
        # 可视化
        plt.figure(figsize=(8, 8))
        
        # 绘制网格
        plt.imshow(grid, cmap='binary', alpha=0.3)
        
        # 绘制路径
        path_array = np.array(path)
        plt.plot(path_array[:, 1], path_array[:, 0], 
                'b-o', linewidth=2, markersize=6, label='A* 路径')
        
        # 标记起点和终点
        plt.plot(start[1], start[0], 'gs', markersize=15, label='起点')
        plt.plot(goal[1], goal[0], 'r^', markersize=15, label='终点')
        
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('A* 搜索路径', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/root/.openclaw/workspace/cable-optimization/examples/astar_simple.png', dpi=150)
        print("结果图已保存到：astar_simple.png")
        plt.show()
    else:
        print("未找到路径!")
    
    return astar, grid, path


def test_astar_heuristics():
    """比较不同启发式函数的效果"""
    print("\n" + "=" * 60)
    print("A* 测试 - 启发式函数对比")
    print("=" * 60)
    
    # 创建测试网格
    grid = np.zeros((20, 20), dtype=int)
    
    # 添加随机障碍
    np.random.seed(42)
    n_obstacles = 50
    for _ in range(n_obstacles):
        x, y = np.random.randint(0, 20, 2)
        grid[x, y] = 1
    
    start = (0, 0)
    goal = (19, 19)
    
    heuristics = ['euclidean', 'manhattan', 'chebyshev', 'octile']
    results = []
    
    for h_name in heuristics:
        astar = AStarSearch(heuristic=h_name, allow_diagonal=True)
        path = astar.search(grid, start, goal)
        
        if path:
            path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) 
                            for i in range(len(path)-1))
            stats = astar.get_search_stats()
            
            results.append({
                'heuristic': h_name,
                'path_length': path_length,
                'searched': stats['searched_nodes'],
                'expanded': stats['expanded_nodes']
            })
            
            print(f"\n{h_name}:")
            print(f"  路径长度：{path_length:.2f}")
            print(f"  搜索节点：{stats['searched_nodes']}")
            print(f"  扩展节点：{stats['expanded_nodes']}")
        else:
            print(f"\n{h_name}: 未找到路径")
    
    # 可视化对比
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    h_names = [r['heuristic'] for r in results]
    searched = [r['searched'] for r in results]
    plt.bar(h_names, searched, color='steelblue', alpha=0.7)
    plt.xlabel('启发式函数', fontsize=12)
    plt.ylabel('搜索节点数', fontsize=12)
    plt.title('不同启发式的搜索效率对比', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    expanded = [r['expanded'] for r in results]
    plt.bar(h_names, expanded, color='coral', alpha=0.7)
    plt.xlabel('启发式函数', fontsize=12)
    plt.ylabel('扩展节点数', fontsize=12)
    plt.title('不同启发式的扩展节点对比', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/root/.openclaw/workspace/cable-optimization/examples/astar_heuristics_compare.png', dpi=150)
    print("\n对比图已保存到：astar_heuristics_compare.png")
    plt.show()
    
    return results


def apply_astar_to_cable_routing():
    """
    A* 在线缆布线中的应用
    
    问题：在复杂环境中规划线缆路径，避开障碍物
    """
    print("\n" + "=" * 60)
    print("A* 应用 - 复杂环境线缆布线")
    print("=" * 60)
    
    # 创建复杂环境（模拟机房/工厂布局）
    grid_size = 30
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # 添加设备/障碍区域
    # 设备 1
    grid[8:12, 5:15] = 1
    # 设备 2
    grid[18:25, 10:20] = 1
    # 设备 3
    grid[5:10, 20:28] = 1
    # 墙壁
    grid[15, 5:18] = 1
    
    # 起点和终点（接线端子位置）
    start = (2, 2)
    goal = (27, 27)
    
    print(f"网格大小：{grid_size}x{grid_size}")
    print(f"起点：{start}, 终点：{goal}")
    print(f"障碍物占比：{grid.sum() / (grid_size*grid_size) * 100:.1f}%")
    
    # 执行 A* 搜索
    astar = AStarSearch(heuristic='octile', allow_diagonal=True)
    path = astar.search(grid, start, goal, verbose=True)
    
    if path:
        stats = astar.get_search_stats()
        
        # 计算路径长度
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) 
                        for i in range(len(path)-1))
        
        print(f"\n✅ 找到最优路径!")
        print(f"路径长度：{path_length:.2f} 单位")
        print(f"路径点数：{len(path)}")
        print(f"搜索节点：{stats['searched_nodes']}")
        print(f"扩展节点：{stats['expanded_nodes']}")
        print(f"搜索效率：{stats['expanded_nodes']/stats['searched_nodes']*100:.1f}%")
        
        # 可视化
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制网格背景
        ax.imshow(grid, cmap='Greys', alpha=0.5)
        
        # 绘制路径
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 
               'b-', linewidth=3, label='A* 规划路径')
        ax.plot(path_array[:, 1], path_array[:, 0], 
               'bo', markersize=4, alpha=0.5)
        
        # 标记关键点
        ax.plot(start[1], start[0], 'gs', markersize=20, label='起点 (接线端 A)')
        ax.plot(goal[1], goal[0], 'r^', markersize=20, label='终点 (接线端 B)')
        
        # 标注设备区域
        ax.add_patch(plt.Rectangle((5, 8), 10, 4, fill=True, 
                                   color='orange', alpha=0.3, label='设备区 1'))
        ax.add_patch(plt.Rectangle((10, 18), 10, 7, fill=True, 
                                   color='orange', alpha=0.3))
        ax.add_patch(plt.Rectangle((20, 5), 8, 5, fill=True, 
                                   color='orange', alpha=0.3))
        
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        ax.set_title('A* 算法 - 复杂环境线缆路径规划', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, grid_size)
        ax.set_ylim(-1, grid_size)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('/root/.openclaw/workspace/cable-optimization/examples/astar_cable_routing.png', dpi=150)
        print("\n路径规划图已保存到：astar_cable_routing.png")
        plt.show()
        
        return astar, grid, path
    else:
        print("❌ 未找到可行路径!")
        return astar, grid, None


def visualize_astar_process():
    """可视化 A* 搜索过程"""
    print("\n" + "=" * 60)
    print("A* 可视化 - 搜索过程演示")
    print("=" * 60)
    
    # 小型网格用于演示
    grid = np.zeros((15, 15), dtype=int)
    grid[5:8, 3:10] = 1  # 横向障碍
    grid[10:13, 8:13] = 1  # 右下障碍
    
    start = (2, 2)
    goal = (13, 13)
    
    astar = AStarSearch(heuristic='euclidean', allow_diagonal=True)
    path = astar.search(grid, start, goal)
    
    if path:
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：最终路径
        ax1 = axes[0]
        ax1.imshow(grid, cmap='binary', alpha=0.3)
        
        path_array = np.array(path)
        ax1.plot(path_array[:, 1], path_array[:, 0], 'b-o', linewidth=2, markersize=6)
        ax1.plot(start[1], start[0], 'gs', markersize=15)
        ax1.plot(goal[1], goal[0], 'r^', markersize=15)
        
        ax1.set_title(f'最终路径 (长度={len(path)}步)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 右图：搜索过的节点
        ax2 = axes[1]
        ax2.imshow(grid, cmap='binary', alpha=0.3)
        
        # 绘制所有搜索过的节点
        closed_array = np.array(list(astar.closed_set))
        if len(closed_array) > 0:
            ax2.scatter(closed_array[:, 1], closed_array[:, 0], 
                       c='yellow', s=30, alpha=0.6, label='已搜索节点')
        
        ax2.plot(path_array[:, 1], path_array[:, 0], 'b-o', linewidth=2, markersize=6, label='最终路径')
        ax2.plot(start[1], start[0], 'gs', markersize=15, label='起点')
        ax2.plot(goal[1], goal[0], 'r^', markersize=15, label='终点')
        
        ax2.set_title(f'搜索过程 (扩展{astar.expanded_nodes}节点)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/root/.openclaw/workspace/cable-optimization/examples/astar_process.png', dpi=150)
        print("搜索过程图已保存到：astar_process.png")
        plt.show()
    
    return astar


if __name__ == "__main__":
    print("\n" + "⭐" * 30)
    print("A* 搜索算法 - 线缆布线优化")
    print("⭐" * 30 + "\n")
    
    # 测试 1: 简单网格
    astar1, grid1, path1 = test_astar_simple()
    
    # 测试 2: 启发式对比
    results = test_astar_heuristics()
    
    # 测试 3: 复杂环境布线
    astar3, grid3, path3 = apply_astar_to_cable_routing()
    
    # 测试 4: 搜索过程可视化
    astar4 = visualize_astar_process()
    
    print("\n" + "=" * 60)
    print("A* 算法学习完成!")
    print("=" * 60)
    print("\n关键收获:")
    print("1. A* 通过启发式函数指导搜索方向")
    print("2. f(n) = g(n) + h(n) 是核心评估函数")
    print("3. 启发式必须可采纳（不高估）才能保证最优")
    print("4. 不同启发式影响搜索效率")
    print("5. 适合网格环境的路径规划")
    print("\n本周算法完成：PSO ✅, SA ✅, A* ✅")
    print("明日继续：最小生成树算法 (Prim/Kruskal)")
