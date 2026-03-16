"""
Voronoi 关键点生成器 - 优化版

修复问题:
1. 关键点去重
2. 关键点数量控制
3. OMP 库冲突处理

作者：智子 (Sophon)
日期：2026-03-11
"""

import numpy as np
from scipy.spatial import Voronoi
import heapq
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


# 解决 OMP 库冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class VoronoiKeyPointGenerator:
    """Voronoi 关键点生成器（优化版）"""
    
    def __init__(self, grid_map: np.ndarray, safety_distance: float = 2.0):
        """
        初始化
        
        Args:
            grid_map: 栅格地图 (0=可行，1=障碍物)
            safety_distance: 安全距离阈值
        """
        self.grid_map = grid_map
        self.safety_distance = safety_distance
        self.size = grid_map.shape[0]
        self.key_points = []
    
    def extract_obstacle_centers(self) -> List[Tuple[float, float]]:
        """
        提取障碍物中心坐标
        
        Returns:
            障碍物中心坐标列表
        """
        obstacle_coords = np.argwhere(self.grid_map == 1)
        centers = []
        
        # 简单处理：每个障碍物格子作为一个点
        # 论文方法：提取连通区域中心
        for coord in obstacle_coords:
            # 加 0.5 得到中心坐标
            center = (coord[0] + 0.5, coord[1] + 0.5)
            centers.append(center)
        
        return centers
    
    def build_voronoi(self, points: List[Tuple[float, float]]) -> Voronoi:
        """
        构建 Voronoi 图
        
        Args:
            points: 输入点集
        
        Returns:
            Voronoi 对象
        """
        points_array = np.array(points)
        vor = Voronoi(points_array)
        return vor
    
    def filter_vertices(self, vor: Voronoi) -> List[int]:
        """
        过滤障碍物内部的顶点
        
        Args:
            vor: Voronoi 对象
        
        Returns:
            有效顶点索引列表
        """
        valid_indices = []
        
        for i, vertex in enumerate(vor.vertices):
            # 检查顶点是否在地图范围内
            if not (0 <= vertex[0] < self.size and 0 <= vertex[1] < self.size):
                continue
            
            # 检查是否距离障碍物足够远
            grid_x, grid_y = int(vertex[0]), int(vertex[1])
            
            if 0 <= grid_x < self.size and 0 <= grid_y < self.size:
                if self.grid_map[grid_x, grid_y] == 0:
                    valid_indices.append(i)
        
        return valid_indices
    
    def filter_edges(self, vor: Voronoi, valid_indices: List[int]) -> List[Tuple[int, int]]:
        """
        过滤与障碍物相交的边
        
        Args:
            vor: Voronoi 对象
            valid_indices: 有效顶点索引
        
        Returns:
            有效边列表
        """
        valid_edges = []
        valid_set = set(valid_indices)
        
        for ridge in vor.ridge_vertices:
            if -1 in ridge:
                continue
            
            # 检查边的两个端点是否都有效
            if ridge[0] in valid_set and ridge[1] in valid_set:
                valid_edges.append((ridge[0], ridge[1]))
        
        return valid_edges
    
    def remove_redundant_nodes(self, edges: List[Tuple[int, int]], 
                               vertices: np.ndarray,
                               tolerance: float = 1e-5) -> List[Tuple[int, int]]:
        """
        移除共线的冗余节点
        
        Args:
            edges: 边列表
            vertices: 顶点坐标
            tolerance: 容差
        
        Returns:
            简化后的边列表
        """
        # 构建邻接表
        adjacency = {}
        for i, j in edges:
            if i not in adjacency:
                adjacency[i] = []
            if j not in adjacency:
                adjacency[j] = []
            adjacency[i].append(j)
            adjacency[j].append(i)
        
        # 查找并移除共线点
        removed = set()
        
        for i in adjacency:
            if i in removed:
                continue
            
            neighbors = adjacency[i]
            if len(neighbors) != 2:
                continue
            
            j, k = neighbors
            
            # 计算向量
            v1 = vertices[j] - vertices[i]
            v2 = vertices[k] - vertices[i]
            
            # 计算叉积
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            
            # 如果共线（叉积接近 0）
            if abs(cross) < tolerance:
                # 移除节点 i，连接 j 和 k
                removed.add(i)
                adjacency[j].remove(i)
                adjacency[k].remove(i)
                
                if k not in adjacency[j]:
                    adjacency[j].append(k)
                if j not in adjacency[k]:
                    adjacency[k].append(j)
        
        # 重建边列表
        simplified_edges = []
        seen = set()
        
        for i in adjacency:
            if i in removed:
                continue
            for j in adjacency[i]:
                if j in removed:
                    continue
                edge = tuple(sorted([i, j]))
                if edge not in seen:
                    simplified_edges.append(edge)
                    seen.add(edge)
        
        return simplified_edges
    
    def extract_key_points_from_path(self, path: List[Tuple[int, int]], 
                                     d_threshold: float = None) -> List[Tuple[int, int]]:
        """
        从路径中提取关键点（论文算法 1）
        
        Args:
            path: A* 生成的路径点列表
            d_threshold: 安全距离阈值
        
        Returns:
            关键点列表
        """
        if d_threshold is None:
            d_threshold = self.safety_distance
        
        if not path:
            return []
        
        key_points = [path[0]]  # 起点必为关键点
        i = 0
        
        while i < len(path) - 1:
            S = np.array(path[i])
            j = i + 1
            
            while j < len(path):
                jv = np.array(path[j])
                
                # 计算路径段 S→jv 与障碍物的最小距离
                d_min = self._calculate_min_distance_to_obstacles(S, jv)
                
                if d_min < d_threshold:
                    # 距离不足，将前一个点设为关键点
                    if j > i + 1:
                        key_points.append(path[j - 1])
                    else:
                        key_points.append(path[i])
                    i = j - 1
                    break
                
                j += 1
            
            if j >= len(path) - 1:
                # 到达终点
                break
            
            i += 1
        
        # 终点必为关键点
        if path[-1] not in key_points:
            key_points.append(path[-1])
        
        # 去重
        unique_key_points = []
        for kp in key_points:
            if kp not in unique_key_points:
                unique_key_points.append(kp)
        
        return unique_key_points
    
    def _calculate_min_distance_to_obstacles(self, S: np.ndarray, 
                                             E: np.ndarray) -> float:
        """
        计算路径段 S→E 与障碍物的最小垂直距离（论文式 6）
        
        Args:
            S: 起点
            E: 终点
        
        Returns:
            最小距离
        """
        obstacle_coords = np.argwhere(self.grid_map == 1)
        
        if len(obstacle_coords) == 0:
            return float('inf')
        
        min_dist = float('inf')
        
        # 路径段向量
        SE = E - S
        seg_len = np.linalg.norm(SE)
        
        if seg_len < 1e-6:
            return np.min(np.linalg.norm(obstacle_coords - S, axis=1))
        
        for obs in obstacle_coords:
            # 计算点到直线的垂直距离
            # dist = |det(S, E, P)| / |S - E|
            det = np.abs((E[0] - S[0]) * (obs[1] - S[1]) - 
                        (E[1] - S[1]) * (obs[0] - S[0]))
            dist = det / seg_len
            
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def extract_key_points(self, start: Tuple[int, int], 
                          goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        提取关键点（完整流程）
        
        Args:
            start: 起点
            goal: 终点
        
        Returns:
            关键点列表
        """
        # 1. 提取障碍物中心
        obstacle_centers = self.extract_obstacle_centers()
        
        if len(obstacle_centers) < 2:
            # 障碍物太少，直接返回起点终点
            return [start, goal]
        
        # 2. 构建 Voronoi 图
        vor = self.build_voronoi(obstacle_centers)
        
        # 3. 过滤顶点
        valid_indices = self.filter_vertices(vor)
        
        if len(valid_indices) < 2:
            return [start, goal]
        
        # 4. 过滤边
        valid_edges = self.filter_edges(vor, valid_indices)
        
        # 5. 移除冗余节点
        simplified_edges = self.remove_redundant_nodes(
            valid_edges, vor.vertices
        )
        
        # 6. 构建邻接矩阵
        n_vertices = len(vor.vertices)
        adj_matrix = np.full((n_vertices, n_vertices), np.inf)
        
        for i, j in simplified_edges:
            dist = np.linalg.norm(vor.vertices[i] - vor.vertices[j])
            adj_matrix[i, j] = dist
            adj_matrix[j, i] = dist
        
        # 7. 找到最近的顶点
        start_vertex = self._find_nearest_vertex(vor.vertices, start, valid_indices)
        goal_vertex = self._find_nearest_vertex(vor.vertices, goal, valid_indices)
        
        if start_vertex is None or goal_vertex is None:
            return [start, goal]
        
        # 8. 使用 Dijkstra 找最短路径
        path_indices = self._dijkstra_path(adj_matrix, start_vertex, goal_vertex)
        
        if not path_indices:
            return [start, goal]
        
        # 9. 转换为网格坐标
        path = []
        for idx in path_indices:
            if 0 <= idx < len(vor.vertices):
                vertex = vor.vertices[idx]
                grid_point = (int(vertex[0]), int(vertex[1]))
                path.append(grid_point)
        
        # 10. 添加起点和终点
        if start not in path:
            path.insert(0, start)
        if goal not in path:
            path.append(goal)
        
        # 11. 提取关键点
        key_points = self.extract_key_points_from_path(path)
        
        self.key_points = key_points
        return key_points
    
    def _find_nearest_vertex(self, vertices: np.ndarray, 
                            point: Tuple[int, int],
                            valid_indices: List[int]) -> Optional[int]:
        """找到最近的顶点"""
        min_dist = float('inf')
        nearest_idx = None
        
        point_array = np.array(point)
        
        for idx in valid_indices:
            dist = np.linalg.norm(vertices[idx] - point_array)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        return nearest_idx
    
    def _dijkstra_path(self, adj_matrix: np.ndarray, 
                      start: int, goal: int) -> List[int]:
        """Dijkstra 最短路径"""
        n = adj_matrix.shape[0]
        dist = np.full(n, np.inf)
        prev = [None] * n
        dist[start] = 0
        
        pq = [(0, start)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            
            visited.add(u)
            
            if u == goal:
                # 重构路径
                path = []
                curr = goal
                while curr is not None:
                    path.append(curr)
                    curr = prev[curr]
                return path[::-1]
            
            for v in range(n):
                if adj_matrix[u, v] < np.inf and v not in visited:
                    alt = dist[u] + adj_matrix[u, v]
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
                        heapq.heappush(pq, (alt, v))
        
        return []
    
    def visualize(self, save_path: str = None):
        """可视化 Voronoi 图和关键点"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制地图
        ax.imshow(self.grid_map, cmap='binary', origin='lower')
        
        # 绘制关键点
        if self.key_points:
            kp_x = [kp[1] for kp in self.key_points]
            kp_y = [kp[0] for kp in self.key_points]
            ax.scatter(kp_x, kp_y, c='red', s=50, marker='o', 
                      label=f'关键点 ({len(self.key_points)}个)', zorder=5)
            
            # 连接关键点
            if len(self.key_points) > 1:
                kp_x_line = [kp[1] for kp in self.key_points]
                kp_y_line = [kp[0] for kp in self.key_points]
                ax.plot(kp_x_line, kp_y_line, 'r--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Voronoi 关键点生成 (共{len(self.key_points)}个关键点)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 可视化已保存：{save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """测试 Voronoi 关键点生成"""
    print("="*60)
    print("Voronoi 关键点生成器测试（优化版）")
    print("="*60)
    
    # 创建测试地图
    size = 80
    np.random.seed(42)
    grid = np.zeros((size, size), dtype=np.int8)
    
    # 随机障碍物
    n_obstacles = int(size * size * 0.2)
    for _ in range(n_obstacles):
        x, y = np.random.randint(10, size-10, 2)
        obs_size = np.random.randint(3, 6)
        x1, x2 = max(0, x-obs_size//2), min(size, x+obs_size//2)
        y1, y2 = max(0, y-obs_size//2), min(size, y+obs_size//2)
        grid[x1:x2, y1:y2] = 1
    
    start = (12, 12)
    goal = (75, 130) if size > 130 else (size-5, size-5)
    
    print(f"\n地图大小：{size}x{size}")
    print(f"障碍物数量：{n_obstacles}")
    print(f"起点：{start}")
    print(f"终点：{goal}")
    
    # 生成关键点
    generator = VoronoiKeyPointGenerator(grid, safety_distance=2.0)
    key_points = generator.extract_key_points(start, goal)
    
    print(f"\n✓ 生成 {len(key_points)} 个关键点")
    for i, kp in enumerate(key_points):
        print(f"  {i}: {kp}")
    
    # 可视化
    generator.visualize(save_path="voronoi_keypoints.png")
    
    print("\n✓ 测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
