import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import List, Tuple, Optional, Dict
from scipy.spatial import Voronoi

# 配置字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class VoronoiKeyPointGenerator:
    """Voronoi 关键点生成器 - 完全对齐文献所有公式与伪代码"""

    def __init__(self, grid_map: np.ndarray, safety_distance: float = 2.0, remove_collinear: bool = True):
        self.grid_map = grid_map.astype(int)
        self.safety_distance = safety_distance
        self.remove_collinear = remove_collinear  # 👈 新增：冗余节点移除开关
        self.height, self.width = grid_map.shape

        # 提取全部障碍物/边界栅格点作为 Voronoi 生成基准，确保骨架平滑
        self.obstacle_centers = np.argwhere(self.grid_map == 1) + 0.5

        # 内部状态变量
        self.raw_safe_vertices: np.ndarray = np.array([])  # 过滤安全距离后的原始密点
        self.raw_edges: List[Tuple[np.ndarray, np.ndarray]] = []  # 过滤安全距离后的原始边
        self.edges: List[Tuple[np.ndarray, np.ndarray]] = []  # 最终边 (可能被化简)
        self.vertex_coords: List[np.ndarray] = []  # 最终放入A*的顶点
        self.path_pts: List[np.ndarray] = []
        self.key_points: List[Tuple[int, int]] = []

    def extract_key_points(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        start_arr = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)

        if len(self.obstacle_centers) < 2:
            return [start, goal]

        # === Step 1: 构建带边界的 Voronoi 图 ===
        boundary_pts = np.array([
            [-5, -5], [-5, self.width + 5],
            [self.height + 5, -5], [self.height + 5, self.width + 5],
            [self.height // 2, -5], [self.height // 2, self.width + 5],
            [-5, self.width // 2], [self.height + 5, self.width // 2]
        ])
        all_pts = np.vstack([self.obstacle_centers, boundary_pts])
        self.vor = Voronoi(all_pts)

        # === Step 2: 过滤安全顶点 ===
        valid_vertex_indices = []
        safe_vertices = []
        for i, v in enumerate(self.vor.vertices):
            x, y = v[0], v[1]
            if not (0 <= x <= self.height and 0 <= y <= self.width):
                continue
            min_dist = np.min(np.linalg.norm(self.obstacle_centers - v, axis=1))
            if min_dist >= self.safety_distance:
                safe_vertices.append(v)
                valid_vertex_indices.append(i)

        if not safe_vertices:
            return [start, goal]

        self.raw_safe_vertices = np.array(safe_vertices)
        valid_set = set(valid_vertex_indices)

        # === Step 3: 构建安全边 ===
        skeleton_edges = []
        for ridge_vertices in self.vor.ridge_vertices:
            if -1 in ridge_vertices:
                continue
            i1, i2 = ridge_vertices
            if i1 in valid_set and i2 in valid_set:
                pt1 = self.vor.vertices[i1]
                pt2 = self.vor.vertices[i2]
                skeleton_edges.append((pt1, pt2))

        self.raw_edges = skeleton_edges.copy()

        # === 文献 2.2 节 第(4)步：移除冗余节点 ===
        # 使用文献公式 (4) (5) 将共线的多个线段合并为一条直线，大幅压缩 A* 搜索空间
        if self.remove_collinear:
            skeleton_edges = self._remove_collinear_nodes(skeleton_edges)

        self.edges = skeleton_edges

        # === 构建精简后的真实顶点集合 ===
        unique_pts = {}
        for p1, p2 in skeleton_edges:
            unique_pts[tuple(np.round(p1, 5))] = p1
            unique_pts[tuple(np.round(p2, 5))] = p2

        self.vertex_coords = list(unique_pts.values())
        n = len(self.vertex_coords)

        if n == 0:
            return [start, goal]

        start_idx = n
        goal_idx = n + 1
        self.vertex_coords.extend([start_arr, goal_arr])

        # === Step 4: 构建 A* 邻接图 ===
        adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(self.vertex_coords))}

        # 此时的 skeleton_edges 已经是剔除了冗余节点的高效长边
        for pt1, pt2 in skeleton_edges:
            i1 = next(i for i, v in enumerate(self.vertex_coords[:n]) if np.allclose(v, pt1, atol=1e-5))
            i2 = next(i for i, v in enumerate(self.vertex_coords[:n]) if np.allclose(v, pt2, atol=1e-5))
            dist = np.linalg.norm(pt1 - pt2)
            adj[i1].append((i2, dist))
            adj[i2].append((i1, dist))

        def connect_to_closest(idx: int, point: np.ndarray, k: int = 3):
            dists = [np.linalg.norm(v - point) for v in self.vertex_coords[:n]]
            closest = np.argsort(dists)[:k]
            for j in closest:
                d = dists[j]
                adj[idx].append((j, d))
                adj[j].append((idx, d))

        connect_to_closest(start_idx, start_arr)
        connect_to_closest(goal_idx, goal_arr)

        # === Step 5: A* 搜索 ===
        path_pts = self._astar_path(adj, start_idx, goal_idx)
        if path_pts is None:
            return [start, goal]

        # === Step 6: 算法1剪枝提取关键点 ===
        key_points = self._extract_key_points_from_path(path_pts)
        self.key_points = [tuple(p.astype(int)) for p in key_points]
        return self.key_points

    def _remove_collinear_nodes(self, edges: List[Tuple[np.ndarray, np.ndarray]]) -> List[
        Tuple[np.ndarray, np.ndarray]]:
        """文献 2.2 节 第(4)步: 基于公式(4)与(5)的共线冗余节点移除"""
        # 1. 建立节点坐标字典映射，降低图遍历的时间复杂度
        pt_to_id = {}
        id_to_pt = {}
        curr_id = 0
        for p1, p2 in edges:
            for pt in (p1, p2):
                k = tuple(np.round(pt, 5))
                if k not in pt_to_id:
                    pt_to_id[k] = curr_id
                    id_to_pt[curr_id] = pt
                    curr_id += 1

        # 2. 构建图的邻接表
        adj_list = {i: set() for i in range(curr_id)}
        for p1, p2 in edges:
            u, v = pt_to_id[tuple(np.round(p1, 5))], pt_to_id[tuple(np.round(p2, 5))]
            adj_list[u].add(v)
            adj_list[v].add(u)

        # 3. 循环剔除度数为 2 且共线的节点 (公式4, 5)
        removed_any = True
        while removed_any:
            removed_any = False
            for j, neighbors in list(adj_list.items()):
                if j not in adj_list:
                    continue
                if len(neighbors) == 2:  # 只有夹在中间的节点(度数为2)才可能冗余
                    nbs = list(neighbors)
                    i, k = nbs[0], nbs[1]
                    vi, vj, vk = id_to_pt[i], id_to_pt[j], id_to_pt[k]

                    # 【文献公式 4】：计算向量 vivj 与 vivk 的叉积
                    # (xj - xi)*(yk - yi) - (xk - xi)*(yj - yi)
                    cross_prod = (vj[0] - vi[0]) * (vk[1] - vi[1]) - (vk[0] - vi[0]) * (vj[1] - vi[1])

                    # 【文献公式 5】：若叉积为0(即三点共线)，则节点 j 为冗余节点
                    if abs(cross_prod) < 1e-4:  # 浮点数比较采用阈值容差
                        # 移除 j，将 i 和 k 直接相连！
                        adj_list[i].remove(j)
                        adj_list[k].remove(j)
                        adj_list[i].add(k)
                        adj_list[k].add(i)
                        del adj_list[j]
                        removed_any = True

        # 4. 从剔除后的精简邻接表中重构全新的安全长边
        new_edges = []
        for u, neighbors in adj_list.items():
            for v in neighbors:
                if u < v:
                    new_edges.append((id_to_pt[u], id_to_pt[v]))

        return new_edges

    def _astar_path(self, adj, start_idx, goal_idx):
        def heuristic(a, b):
            return np.linalg.norm(self.vertex_coords[a] - self.vertex_coords[b])

        heap = [(0.0, start_idx, [start_idx])]
        visited = set()
        while heap:
            f, curr, path = heapq.heappop(heap)
            if curr in visited: continue
            visited.add(curr)
            if curr == goal_idx:
                self.path_pts = [self.vertex_coords[i] for i in path]
                return self.path_pts
            g = f - heuristic(curr, goal_idx)
            for nb, dist in adj[curr]:
                if nb not in visited:
                    heapq.heappush(heap, (g + dist + heuristic(nb, goal_idx), nb, path + [nb]))
        return None

    def _extract_key_points_from_path(self, path_pts):
        """【算法 1】：从路径中提取关键点 """
        if len(path_pts) <= 2: return path_pts
        t_path = [path_pts[0]]
        i = 0
        n = len(path_pts)
        while i < n - 1:
            S = path_pts[i]
            j = i + 1
            safe_j = j
            while j < n:
                jv = path_pts[j]
                segment_vec = jv - S
                seg_len = np.linalg.norm(segment_vec)
                if seg_len < 1e-6:
                    d_min = np.min(np.linalg.norm(self.obstacle_centers - S, axis=1))
                else:
                    # 【文献公式 6】：利用叉积(即平行四边形面积)除以底边长，精确求线段到所有障碍物的垂直距离
                    cross_prods = np.abs(
                        segment_vec[0] * (self.obstacle_centers[:, 1] - S[1]) -
                        segment_vec[1] * (self.obstacle_centers[:, 0] - S[0])
                    )
                    d_min = np.min(cross_prods) / seg_len
                if d_min < self.safety_distance: break
                safe_j = j
                j += 1
            if safe_j == i: safe_j = i + 1
            t_path.append(path_pts[safe_j])
            i = safe_j
        result = []
        for p in t_path:
            if not result or not np.allclose(result[-1], p, atol=1e-5):
                result.append(p)
        return result

    def visualize(self, save_path: str = None):
        """完全按照文献截图风格 1:1 复刻的绘图配置"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'wspace': 0.1, 'hspace': 0.2},dpi=300)

        titles = ['a)网格坐标转换，Voronoi图建立', 'b)过滤障碍物顶点及边',
                  'c)移除冗余顶点', 'd)关键路径点提取']

        for idx, ax in enumerate(axes.flatten()):
            ax.imshow(1 - self.grid_map, cmap='gray', origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[idx], y=-0.12, fontsize=14)
            ax.set_xlim([0, self.width])
            ax.set_ylim([0, self.height])

            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(3)

            # ======== a) 图：原始全体 Voronoi ========
            if idx == 0:
                ax.scatter(self.obstacle_centers[:, 1], self.obstacle_centers[:, 0], c='#2d8a39', s=0.5)
                all_lines = []
                for ridge in self.vor.ridge_vertices:
                    if -1 not in ridge:
                        p1, p2 = self.vor.vertices[ridge[0]], self.vor.vertices[ridge[1]]
                        all_lines.append([(p1[1], p1[0]), (p2[1], p2[0])])
                ax.add_collection(LineCollection(all_lines, colors='#6a7ebf', linewidths=0.5))

            # ======== b) 图：过滤安全距离后的图 (依然包含大量共线冗余节点) ========
            elif idx == 1:
                raw_edge_lines = [[(p1[1], p1[0]), (p2[1], p2[0])] for p1, p2 in self.raw_edges]
                ax.add_collection(LineCollection(raw_edge_lines, colors='#4b65ba', linewidths=0.6))
                if len(self.raw_safe_vertices) > 0:
                    ax.scatter(self.raw_safe_vertices[:, 1], self.raw_safe_vertices[:, 0], c='#2d8a39', s=2, zorder=3)

            # ======== c) 图：移除冗余顶点后的图 (应用文献 2.2 第 4 步后) ========
            elif idx == 2:
                final_edge_lines = [[(p1[1], p1[0]), (p2[1], p2[0])] for p1, p2 in self.edges]
                ax.add_collection(LineCollection(final_edge_lines, colors='#4b65ba', linewidths=0.6))
                if len(self.vertex_coords) > 2:  # 去掉最终追加的起止点，真实展示图网络节点
                    coords_arr = np.array(self.vertex_coords[:-2])
                    ax.scatter(coords_arr[:, 1], coords_arr[:, 0], c='#2d8a39', s=8, marker='D', zorder=3)

            # ======== d) 图：最终提取关键点 ========
            elif idx == 3:
                final_edge_lines = [[(p1[1], p1[0]), (p2[1], p2[0])] for p1, p2 in self.edges]
                ax.add_collection(LineCollection(final_edge_lines, colors='#4b65ba', linewidths=0.6, alpha=0.5))
                if len(self.vertex_coords) > 2:
                    coords_arr = np.array(self.vertex_coords[:-2])
                    ax.scatter(coords_arr[:, 1], coords_arr[:, 0], c='#2d8a39', s=2, alpha=0.5, zorder=2)

                if self.key_points:
                    kpx = [kp[1] for kp in self.key_points]
                    kpy = [kp[0] for kp in self.key_points]
                    ax.plot(kpx, kpy, color='#c23531', linewidth=1.5, zorder=4)
                    ax.scatter(kpx, kpy, c='#c23531', s=10, zorder=5)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_paper_map():
    grid = np.zeros((100, 160), dtype=int)
    grid[0:6, :] = 1
    grid[-6:, :] = 1
    grid[:, 0:6] = 1
    grid[:, -6:] = 1
    grid[15:45, 15:55] = 1
    grid[25:35, 25:45] = 0
    grid[45:60, 65:80] = 1
    grid[15:70, 110:135] = 1
    grid[80:92, 105:145] = 1
    grid[60:92, 15:35] = 1
    grid[78:92, 35:65] = 1
    grid[-18:, -20:] = 1
    grid[-25:-18, -10:] = 1
    return grid


if __name__ == "__main__":
    grid_map = create_paper_map()

    # 初始化时打开 remove_collinear=True 开关，即可看到图c的完美剔除效果
    generator = VoronoiKeyPointGenerator(grid_map, safety_distance=4.5, remove_collinear=True)

    start_pos = (12, 12)
    goal_pos = (75, 130)
    key_points = generator.extract_key_points(start_pos, goal_pos)

    generator.visualize(save_path="voronoi_result_paper_style.png")