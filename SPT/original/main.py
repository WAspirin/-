import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict, Set, Optional, Any
import heapq
from collections import defaultdict
import math

from scipy.interpolate import splprep, splev
from scipy.signal import convolve2d
from matplotlib.collections import LineCollection
from ortools.linear_solver import pywraplp
import time
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from skimage.morphology import skeletonize
import sknw
from itertools import combinations

# --- Matplotlib 全局配置，用于正确显示中文 ---
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 英文用Times New Roman，中文用SimSun
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# [Function] draw_background (no changes)
def draw_background(ax, router):
    """改进的背景绘制函数 - 更清晰的颜色映射"""
    H, W = router.grid_rows, router.grid_cols
    bg_img = np.ones((H, W, 3))
    cost = router.cost_grid
    inf_mask = np.isinf(cost)
    bg_img[inf_mask] = [0, 0, 0]
    low_cost_mask = (~inf_mask) & (cost < 1.0)
    bg_img[low_cost_mask] = [0.7, 1, 0.7]
    normal_cost_mask = (~inf_mask) & (cost == 1.0)
    bg_img[normal_cost_mask] = [1, 1, 1]
    medium_cost_mask = (~inf_mask) & (cost > 1.0) & (cost < 5.0)
    bg_img[medium_cost_mask] = [1, 0.7, 0.7]
    high_cost_mask = (~inf_mask) & (cost >= 5.0)
    bg_img[high_cost_mask] = [0.8, 0, 0]
    bg_img_flipped = np.flipud(bg_img)
    ax.imshow(
        bg_img_flipped,
        extent=[0, router.physical_width, 0, router.physical_height],
        origin='lower',
        interpolation='none',
        alpha=0.6
    )


# [Class] GridCableRouter (no changes)
class GridCableRouter:
    """二维离散化网格电缆布线器（优化版）"""

    def __init__(self, physical_size: Tuple[float, float], grid_size: Tuple[int, int],
                 obstacles: List[Tuple[float, float]] = None,
                 cost_zones: List[Dict] = None,
                 enable_diagonal: bool = True):
        self.physical_width, self.physical_height = physical_size
        self.grid_cols, self.grid_rows = grid_size
        self.enable_diagonal = enable_diagonal
        self.cell_width = self.physical_width / self.grid_cols
        self.cell_height = self.physical_height / self.grid_rows
        self.cost_grid = np.ones((self.grid_rows, self.grid_cols))
        if obstacles:
            for x, y in obstacles:
                gx, gy = self.physical_to_grid_coords(x, y)
                if 0 <= gx < self.grid_cols and 0 <= gy < self.grid_rows:
                    row, col = self.grid_coords_to_array_index(gx, gy)
                    self.cost_grid[row, col] = float('inf')
        if cost_zones:
            for zone in cost_zones:
                (x1, y1), (x2, y2) = zone['coords']
                gx1, gy1 = self.physical_to_grid_coords(x1, y1)
                gx2, gy2 = self.physical_to_grid_coords(x2, y2)
                gx1, gx2 = sorted([gx1, gx2])
                gy1, gy2 = sorted([gy1, gy2])
                for gx in range(max(0, gx1), min(self.grid_cols, gx2 + 1)):
                    for gy in range(max(0, gy1), min(self.grid_rows, gy2 + 1)):
                        row, col = self.grid_coords_to_array_index(gx, gy)
                        self.cost_grid[row, col] *= zone.get('cost_multiplier', 1.0)

    def _rasterize_line(self, p1_phys: Tuple[float, float], p2_phys: Tuple[float, float]) -> List[Tuple[int, int]]:
        """
        [NEW] 使用Bresenham算法将两个物理点之间的直线转换为栅格坐标列表。
        """
        g1_x, g1_y = self.physical_to_grid_coords(*p1_phys)
        g2_x, g2_y = self.physical_to_grid_coords(*p2_phys)

        points = []
        dx = abs(g2_x - g1_x)
        dy = -abs(g2_y - g1_y)
        sx = 1 if g1_x < g2_x else -1
        sy = 1 if g1_y < g2_y else -1
        err = dx + dy  # error value e_xy

        x, y = g1_x, g1_y
        while True:
            points.append((x, y))
            if x == g2_x and y == g2_y:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    def physical_to_grid_coords(self, x: float, y: float) -> Tuple[int, int]:
        gx = min(self.grid_cols - 1, max(0, int(x // self.cell_width)))
        gy = min(self.grid_rows - 1, max(0, int(y // self.cell_height)))
        return gx, gy

    def grid_coords_to_array_index(self, gx: int, gy: int) -> Tuple[int, int]:
        return self.grid_rows - 1 - gy, gx

    def array_index_to_grid_coords(self, row: int, col: int) -> Tuple[int, int]:
        return col, self.grid_rows - 1 - row

    def grid_coords_to_physical(self, gx: int, gy: int) -> Tuple[float, float]:
        x = (gx + 0.5) * self.cell_width
        y = (gy + 0.5) * self.cell_height
        return x, y

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = node
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if self.enable_diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                row, col = self.grid_coords_to_array_index(nx, ny)
                if not np.isinf(self.cost_grid[row, col]):
                    neighbors.append((nx, ny))
        return neighbors

    def _calculate_turn_penalty_angle_based(self, parent, current, neighbor, penalty_for_90_deg):
        """
        根据90度转弯的惩罚值，计算与角度成比例的转弯惩罚。
        """
        if not parent or penalty_for_90_deg == 0:
            return 0.0

        v_in = (current[0] - parent[0], current[1] - parent[1])
        v_out = (neighbor[0] - current[0], neighbor[1] - current[1])

        len_in = math.hypot(v_in[0], v_in[1])
        len_out = math.hypot(v_out[0], v_out[1])

        if len_in == 0 or len_out == 0:
            return 0.0

        dot_product = v_in[0] * v_out[0] + v_in[1] * v_out[1]
        cos_theta = max(-1.0, min(1.0, dot_product / (len_in * len_out)))
        angle_deg = math.degrees(math.acos(cos_theta))

        #
        if angle_deg > 0:  # 避免浮点数误差导致对几乎直线的路径施加小惩罚
            return penalty_for_90_deg * (angle_deg / 90.0)

        return 0.0

    def find_path_single_cable(self, start_physical: Tuple[float, float],
                               end_physical: Tuple[float, float],
                               shared_edges: Optional[Dict[Tuple, int]] = None,
                               shared_bonus: float = 0.5,
                               turn_penalty: float = 0.0) -> List[
        Tuple[float, float]]:
        start_gx, start_gy = self.physical_to_grid_coords(*start_physical)
        end_gx, end_gy = self.physical_to_grid_coords(*end_physical)
        start_node, end_node = (start_gx, start_gy), (end_gx, end_gy)

        start_row, start_col = self.grid_coords_to_array_index(*start_node)
        if np.isinf(self.cost_grid[start_row, start_col]): return []
        end_row, end_col = self.grid_coords_to_array_index(*end_node)
        if np.isinf(self.cost_grid[end_row, end_col]): return []

        open_set = [(0, start_node)]
        came_from, g_score = {}, {start_node: 0}
        f_score = {start_node: self.heuristic(start_node, end_node)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end_node:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                path.reverse()
                return [self.grid_coords_to_physical(gx, gy) for gx, gy in path]

            parent = came_from.get(current)

            for neighbor in self.get_neighbors(current):
                move_dist = math.sqrt(2) if abs(neighbor[0] - current[0]) == 1 and abs(
                    neighbor[1] - current[1]) == 1 else 1.0
                row, col = self.grid_coords_to_array_index(*neighbor)
                cost_multiplier = self.cost_grid[row, col]
                raw_cost = move_dist * cost_multiplier

                if shared_edges:
                    edge_key = tuple(
                        sorted((self.grid_coords_to_physical(*current), self.grid_coords_to_physical(*neighbor))))
                    if edge_key in shared_edges:
                        raw_cost *= (1.0 - shared_bonus)

                penalty = self._calculate_turn_penalty_angle_based(parent, current, neighbor, turn_penalty)

                tentative_g = g_score[current] + raw_cost + penalty

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor], g_score[neighbor] = current, tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, end_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []
    def find_path_single_cable_box(self, start_physical: Tuple[float, float],
                               end_physical: Tuple[float, float],
                               shared_edges: Optional[Dict[Tuple, int]] = None,
                               shared_bonus: float = 0.5,
                               turn_penalty: float = 0.5) -> List[Tuple[float, float]]:
        """
        使用带状态的A*算法寻找平滑的单根电缆路径。

        此版本经过彻底重构，集成了所有高级特性：
        1.  **带状态的节点**: 核心算法现在使用 `(x, y, last_direction_index)` 作为节点状态，
            赋予了寻路过程“惯性”，从根本上消除了Z字形抖动，保证路径平滑。
        2.  **鲁棒的共享奖励**: `shared_edges` 字典的键使用整数栅格坐标，
            完全避免了物理坐标的浮点数精度问题。
        3.  **精确的成本模型**: 移动成本基于边的两个端点的平均成本，转弯惩罚则基于角度。

        Args:
            start_physical: 起点的物理坐标。
            end_physical: 终点的物理坐标。
            shared_edges: 一个字典，记录了已被占用的 **栅格边** 及其被使用次数。
                          例如: {((gx1, gy1), (gx2, gy2)): count}。
            shared_bonus: 对于共享边，提供的成本折扣因子。
            turn_penalty: 90度转弯的基础惩罚值。

        Returns:
            从起点到终点的物理坐标路径列表，如果找不到则返回空列表。
        """
        start_gx, start_gy = self.physical_to_grid_coords(*start_physical)
        end_gx, end_gy = self.physical_to_grid_coords(*end_physical)
        start_node, end_node = (start_gx, start_gy), (end_gx, end_gy)

        # 检查起点/终点是否有效
        start_row, start_col = self.grid_coords_to_array_index(*start_node)
        if np.isinf(self.cost_grid[start_row, start_col]): return []
        end_row, end_col = self.grid_coords_to_array_index(*end_node)
        if np.isinf(self.cost_grid[end_row, end_col]): return []

        # 定义所有可能的移动方向
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] + (
            [(1, 1), (1, -1), (-1, 1), (-1, -1)] if self.enable_diagonal else [])

        # A* 核心数据结构 (基于状态)
        # 优先队列: (f_score, x, y, last_direction_index)
        open_set = [(0, start_node[0], start_node[1], -1)]
        # g_score 和 came_from 的键是状态元组: (x, y, last_direction_index)
        g_score = defaultdict(lambda: float('inf'))
        came_from = {}
        g_score[(start_node[0], start_node[1], -1)] = 0

        while open_set:
            f, cx, cy, last_d_idx = heapq.heappop(open_set)

            # 优化：如果队列中取出的节点的f_score已经比记录的g_score还差，则跳过
            if f > g_score[(cx, cy, last_d_idx)] + self.heuristic((cx, cy), end_node):
                continue

            # 到达终点，重构路径
            if (cx, cy) == end_node:
                path_grid = []
                curr_state = (cx, cy, last_d_idx)
                while curr_state in came_from:
                    gx, gy, _ = curr_state
                    path_grid.append((gx, gy))
                    curr_state = came_from[curr_state]
                path_grid.append(start_node)
                path_grid.reverse()
                return [self.grid_coords_to_physical(gx, gy) for gx, gy in path_grid]

            # 遍历所有邻居
            for d_idx, (dx, dy) in enumerate(directions):
                nx, ny = cx + dx, cy + dy

                if not (0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows):
                    continue

                row, col = self.grid_coords_to_array_index(nx, ny)
                if np.isinf(self.cost_grid[row, col]):
                    continue

                # --- 成本计算 ---
                move_dist = math.hypot(dx, dy)
                cost_multiplier = (self.cost_grid[self.grid_coords_to_array_index(cx, cy)] +
                                   self.cost_grid[row, col]) / 2.0
                move_cost = move_dist * cost_multiplier

                # 应用共享路段奖励
                if shared_edges:
                    edge_key = tuple(sorted(((cx, cy), (nx, ny))))
                    if edge_key in shared_edges:
                        move_cost *= (1.0 - shared_bonus)

                # 应用转弯惩罚
                penalty = 0.0
                if last_d_idx != -1 and turn_penalty > 0:
                    parent_node_gx = cx - directions[last_d_idx][0]
                    parent_node_gy = cy - directions[last_d_idx][1]
                    penalty = self._calculate_turn_penalty_angle_based(
                        parent=(parent_node_gx, parent_node_gy),
                        current=(cx, cy),
                        neighbor=(nx, ny),
                        penalty_for_90_deg=turn_penalty
                    )

                total_move_cost = move_cost + penalty

                # --- A* 更新逻辑 (基于状态) ---
                tentative_g = g_score[(cx, cy, last_d_idx)] + total_move_cost
                next_state = (nx, ny, d_idx)

                if tentative_g < g_score[next_state]:
                    g_score[next_state] = tentative_g
                    came_from[next_state] = (cx, cy, last_d_idx)
                    h = self.heuristic((nx, ny), end_node)
                    heapq.heappush(open_set, (tentative_g + h, nx, ny, d_idx))

        return []


class TopologyExtractor:
    def __init__(self, router: GridCableRouter, routes: Dict[int, List[Tuple[float, float]]]):
        self.router = router
        self.routes = routes

    def calculate_segment_cost(self, segment: List[Tuple[float, float]]) -> float:
        total_cost = 0.0
        if not segment or len(segment) < 2: return 0.0
        for i in range(len(segment) - 1):
            p1, p2 = segment[i], segment[i + 1]
            gx1, gy1 = self.router.physical_to_grid_coords(*p1)
            gx2, gy2 = self.router.physical_to_grid_coords(*p2)
            r1, c1 = self.router.grid_coords_to_array_index(gx1, gy1)
            r2, c2 = self.router.grid_coords_to_array_index(gx2, gy2)
            cost1, cost2 = self.router.cost_grid[r1, c1], self.router.cost_grid[r2, c2]
            if np.isinf(cost1) or np.isinf(cost2): return float('inf')
            avg_cost = (cost1 + cost2) / 2.0
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            total_cost += length * avg_cost
        return total_cost

    def visualize_skeleton(self, keypoints: Set[Tuple[float, float]], path_segments: Dict,
                           title: str = '拓扑骨架', save_path=None):
        fig, ax = plt.subplots(figsize=(20, 10))
        draw_background(ax, self.router)
        lines = [segment['path'] for segment in path_segments.values()]
        lc = LineCollection(lines, colors='red', linewidths=3, alpha=0.8, zorder=4)
        ax.add_collection(lc)
        if keypoints:
            kx, ky = zip(*keypoints)
            ax.scatter(kx, ky, color='yellow', marker='*', s=150, zorder=10, label='Keypoints')
        endpoints = self._get_all_endpoints()
        ep_in_kp = [ep for ep in endpoints if ep in keypoints]
        if ep_in_kp:
            ex, ey = zip(*ep_in_kp)
            ax.scatter(ex, ey, color='green', marker='s', s=80, zorder=11, label='Endpoints')
        ax.set_xlim(0, self.router.physical_width)
        ax.set_ylim(0, self.router.physical_height)
        ax.set_title(title)
        ax.legend()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    def _extract_skeleton_from_routes(self, target_routes: Dict) -> Tuple[Set, Set]:
        edges, nodes = set(), set()
        for route in target_routes.values():
            if not route: continue
            for j in range(len(route) - 1):
                p1, p2 = route[j], route[j + 1]
                edge = tuple(sorted((p1, p2)))
                edges.add(edge)
                nodes.add(p1)
                nodes.add(p2)
        return nodes, edges

    def identify_keypoints(self, nodes: Set[Tuple[float, float]]) -> Set[Tuple[float, float]]:
        G = nx.Graph()
        if not self.routes: return set()
        for route in self.routes.values():
            if route and len(route) >= 2: nx.add_path(G, route)
        if not G: return set()
        keypoints = set()
        endpoints = self._get_all_endpoints()
        keypoints.update(endpoints)
        for node in G.nodes():
            degree = G.degree(node)
            if degree >= 3 or (degree == 1 and node not in endpoints):
                keypoints.add(node)
        return keypoints

    def build_simplified_graph(self) -> Tuple[Set, Dict]:
        initial_nodes, _ = self._extract_skeleton_from_routes(self.routes)
        print(f"--- 拓扑骨架提取前---\n  初始节点数量: {len(initial_nodes)}")
        keypoints = self.identify_keypoints(initial_nodes)
        path_segments = {}
        for route in self.routes.values():
            if not route: continue
            kp_indices = [i for i, pt in enumerate(route) if pt in keypoints]
            if len(kp_indices) < 2:
                kp_indices = sorted(list(set([0, len(route) - 1])))

            for j in range(len(kp_indices) - 1):
                start_idx, end_idx = kp_indices[j], kp_indices[j + 1]
                if start_idx == end_idx: continue
                segment = route[start_idx: end_idx + 1]
                edge_key = tuple(sorted((segment[0], segment[-1])))
                current_cost = self.calculate_segment_cost(segment)

                # Use a dictionary to store both path and cost
                if edge_key not in path_segments or current_cost < path_segments[edge_key]['cost']:
                    path_segments[edge_key] = {'path': segment, 'cost': current_cost}

        print(
            f"--- 拓扑骨架提取后 ---\n  关键节点数量: {len(keypoints)}\n  关键边 (路径段) 数量: {len(path_segments)}\n------------------------")
        return keypoints, path_segments

    def _get_all_endpoints(self) -> Set[Tuple[float, float]]:
        return {p for route in self.routes.values() if route for p in (route[0], route[-1])}

    def _merge_nearby_keypoints(self, keypoints: Set[Tuple[float, float]], path_segments: Dict,
                                endpoint_merge_distance: float, internal_merge_distance: float
                                ) -> Tuple[
        Set[Tuple[float, float]], Set[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        """基于路径距离的拓扑感知关键点合并算法。"""
        print("  正在执行拓扑感知的关键点合并...")
        G = nx.Graph()
        for (u, v), segment_info in path_segments.items():
            cost = segment_info['cost']
            if math.isfinite(cost) and cost > 0:
                G.add_edge(u, v, weight=cost)

        endpoints_set = self._get_all_endpoints()
        keypoint_mapping = {p: p for p in keypoints}
        merged = set()

        # 阶段 1: 端点吸收（将附近的交叉点合并到端点上）
        for ep in endpoints_set:
            if ep not in G: continue
            for kp in keypoints:
                if kp == ep or kp in merged or kp in endpoints_set or kp not in G: continue
                try:
                    dist = nx.shortest_path_length(G, source=ep, target=kp, weight='weight')
                    if dist <= endpoint_merge_distance:
                        keypoint_mapping[kp] = ep
                        merged.add(kp)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        # 阶段 2: 内部合并（将内部的交叉点合并在一起）
        remaining_kps = [kp for kp in keypoints if kp not in endpoints_set and keypoint_mapping[kp] == kp]
        layout_center = (self.router.physical_width / 2, self.router.physical_height / 2)
        remaining_kps.sort(key=lambda p: math.hypot(p[0] - layout_center[0], p[1] - layout_center[1]))

        for center_kp in remaining_kps:
            if center_kp in merged: continue
            for other_kp in remaining_kps:
                if center_kp == other_kp or other_kp in merged: continue
                try:
                    dist = nx.shortest_path_length(G, source=center_kp, target=other_kp, weight='weight')
                    if dist <= internal_merge_distance:
                        keypoint_mapping[other_kp] = center_kp
                        merged.add(other_kp)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        # 阶段 3: 生成新的拓扑结构（新的关键点和它们之间的连接关系）
        new_edges = set()
        for u, v in path_segments:
            new_u, new_v = keypoint_mapping[u], keypoint_mapping[v]
            if new_u != new_v:
                new_edges.add(tuple(sorted((new_u, new_v))))

        new_keypoints = set(keypoint_mapping.values())
        print(f"  合并完成。关键点数量从 {len(keypoints)} 减少到 {len(new_keypoints)}。")
        return new_keypoints, new_edges

    def _find_smooth_path_between_keypoints(self,
                                            start_physical: Tuple[float, float],
                                            end_physical: Tuple[float, float],
                                            turn_penalty: float,
                                            shared_grid_edges: Optional[Dict[Tuple, int]] = None,
                                            shared_bonus: float = 0.0):
        """
        使用带状态的A*算法为关键点之间寻找平滑路径。
        支持对共享路段进行成本奖励，以鼓励路径捆绑。
        """
        start_gx, start_gy = self.router.physical_to_grid_coords(*start_physical)
        end_gx, end_gy = self.router.physical_to_grid_coords(*end_physical)
        start_node, end_node = (start_gx, start_gy), (end_gx, end_gy)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] + (
            [(1, 1), (1, -1), (-1, 1), (-1, -1)] if self.router.enable_diagonal else [])
        dir_to_idx = {d: i for i, d in enumerate(directions)}

        # 优先队列: (f_score, x, y, last_direction_index)
        open_set = [(0, start_node[0], start_node[1], -1)]
        g_score = defaultdict(lambda: float('inf'))
        came_from = {}
        g_score[(start_node[0], start_node[1], -1)] = 0

        while open_set:
            f, cx, cy, last_d_idx = heapq.heappop(open_set)

            # 如果g_score已经有更优解，则跳过（A*的常见优化）
            if f > g_score[(cx, cy, last_d_idx)] + self.router.heuristic((cx, cy), end_node):
                continue

            if (cx, cy) == end_node:
                path_grid = []
                curr = (cx, cy, last_d_idx)
                while curr in came_from:
                    gx, gy, _ = curr
                    path_grid.append((gx, gy))
                    curr = came_from[curr]
                path_grid.append(start_node)
                path_grid.reverse()
                return [self.router.grid_coords_to_physical(gx, gy) for gx, gy in path_grid]

            # 遍历邻居
            for d_idx, (dx, dy) in enumerate(directions):
                nx, ny = cx + dx, cy + dy

                if not (0 <= nx < self.router.grid_cols and 0 <= ny < self.router.grid_rows):
                    continue

                row, col = self.router.grid_coords_to_array_index(nx, ny)
                if np.isinf(self.router.cost_grid[row, col]):
                    continue

                # --- 成本计算 (核心修改部分) ---
                move_dist = math.hypot(dx, dy)
                cost_multiplier = (self.router.cost_grid[self.router.grid_coords_to_array_index(cx, cy)] +
                                   self.router.cost_grid[row, col]) / 2.0
                move_cost = move_dist * cost_multiplier

                # 新增: 应用共享路段奖励
                if shared_grid_edges and shared_bonus > 0:
                    # 注意：shared_grid_edges 的键必须是 grid 坐标
                    edge_key = tuple(sorted(((cx, cy), (nx, ny))))
                    if edge_key in shared_grid_edges:
                        # 降低成本，鼓励走这条路
                        move_cost *= (1.0 - shared_bonus)

                # 应用转向惩罚
                if last_d_idx != -1 and turn_penalty > 0:
                    last_dx, last_dy = directions[last_d_idx]
                    v_in = (last_dx, last_dy)
                    v_out = (dx, dy)

                    dot_product = v_in[0] * v_out[0] + v_in[1] * v_out[1]
                    len_in = math.hypot(v_in[0], v_in[1])
                    len_out = math.hypot(v_out[0], v_out[1])

                    if len_in > 0 and len_out > 0:
                        cos_theta = max(-1.0, min(1.0, dot_product / (len_in * len_out)))
                        angle_deg = math.degrees(math.acos(cos_theta))
                        if angle_deg > 0:
                            penalty = turn_penalty * (angle_deg / 90.0)
                            move_cost += penalty

                tentative_g = g_score[(cx, cy, last_d_idx)] + move_cost
                current_state = (nx, ny, d_idx)

                if tentative_g < g_score[current_state]:
                    g_score[current_state] = tentative_g
                    came_from[current_state] = (cx, cy, last_d_idx)
                    h = self.router.heuristic((nx, ny), end_node)
                    heapq.heappush(open_set, (tentative_g + h, nx, ny, d_idx))
        return []

    def optimize_and_rewire_skeleton(self, keypoints, path_segments, merge_distances, turn_penalty,
                                     shared_bonus: float = 0.3):
        """
        协调骨架简化和几何路径重构的主函数。
        (新增) 支持协同布线，鼓励路径捆绑。
        """
        print("\n--- [骨架优化与重构] ---")
        merged_keypoints, new_topology_edges = self._merge_nearby_keypoints(
            keypoints, path_segments, merge_distances[0], merge_distances[1]
        )

        print(
            f"  正在为 {len(new_topology_edges)} 条新的拓扑边进行协同布线 (转向惩罚: {turn_penalty}, 共享奖励: {shared_bonus})...")

        rewired_segments = {}
        # **核心: 用于记录所有已布线路径经过的栅格路段**
        global_shared_grid_edges = {}

        # (可选但推荐) 对边进行排序，优先布线较长的边，有助于形成稳定的主干
        sorted_edges = sorted(
            list(new_topology_edges),
            key=lambda e: math.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]),
            reverse=True
        )

        for u, v in sorted_edges:
            # **核心: 将当前的共享路段信息传入寻路函数**
            new_path = self._find_smooth_path_between_keypoints(
                u, v,
                turn_penalty=turn_penalty,
                shared_grid_edges=global_shared_grid_edges,
                shared_bonus=shared_bonus
            )

            # 如果寻路失败，可以尝试不带协同机制的备用方法
            if not new_path:
                print(f"    [警告] 协同寻路失败: {u} -> {v}。尝试独立寻路。")
                # new_path = self.router.find_path_single_cable(u, v) # 您的备用方法

            if new_path:
                cost = self.calculate_segment_cost(new_path)
                rewired_segments[tuple(sorted((u, v)))] = {'path': new_path, 'cost': cost}

                # **核心: 更新全局共享路段字典**
                # 将新路径的物理坐标转回栅格坐标
                path_grid = [self.router.physical_to_grid_coords(*p) for p in new_path]
                for i in range(len(path_grid) - 1):
                    # 使用排序后的元组作为键，确保 (a,b) 和 (b,a) 是同一个路段
                    grid_edge = tuple(sorted((path_grid[i], path_grid[i + 1])))
                    # 增加该路段的被共享次数
                    global_shared_grid_edges[grid_edge] = global_shared_grid_edges.get(grid_edge, 0) + 1
            else:
                print(f"    [错误] 无法为边 {u} <-> {v} 找到任何路径。")

        print("  骨架重构完成。")
        return merged_keypoints, rewired_segments

class HRHRouter:
    def __init__(self, router: 'GridCableRouter', cables: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        self.router = router
        self.cables = cables
        self.routes: Dict[int, List[Tuple[float, float]]] = {}
        self.shared_edges: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}

    def run_hrh(self, shared_bonus: float = 0.5, max_iterations: int = 3,
                use_shared_bonus_mask: Optional[List[bool]] = None):
        n_cables = len(self.cables)
        if use_shared_bonus_mask is None: use_shared_bonus_mask = [True] * n_cables
        for i, (start, end) in enumerate(self.cables):
            self.routes[i] = self.router.find_path_single_cable(start, end) or []
        for it in range(max_iterations):
            print(f"  HRH Iteration {it + 1}/{max_iterations}...")
            improved = False
            for i in range(n_cables):
                if not self.routes[i]: continue
                start, end = self.cables[i]
                fixed_edges = self._build_fixed_edges(exclude_index=i)
                shared_edges = fixed_edges if use_shared_bonus_mask[i] else None
                new_route = self.router.find_path_single_cable(start, end, shared_edges=shared_edges,
                                                               shared_bonus=shared_bonus)
                if not new_route: continue
                old_cost, _ = self._route_cost(self.routes[i], fixed_edges, shared_bonus)
                new_cost, _ = self._route_cost(new_route, fixed_edges, shared_bonus)
                if new_cost < old_cost:
                    self.routes[i] = new_route
                    improved = True
            self._update_shared_edges()
            if not improved:
                print("  No further improvements found. Stopping HRH early.")
                break

    def _build_fixed_edges(self, exclude_index: int) -> Dict[Tuple, int]:
        fixed_edges = defaultdict(int)
        for j, route in self.routes.items():
            if j == exclude_index or not route: continue
            for k in range(len(route) - 1):
                edge = tuple(sorted((route[k], route[k + 1])))
                fixed_edges[edge] += 1
        return fixed_edges

    def _route_cost(self, route: List[Tuple[float, float]], fixed_edges: Dict[Tuple, int], shared_bonus: float) -> \
            Tuple[float, float]:
        if not route: return float('inf'), 0.0
        cost, length = 0.0, 0.0
        for k in range(len(route) - 1):
            a, b = route[k], route[k + 1]
            base_length = math.hypot(b[0] - a[0], b[1] - a[1])
            gx1, gy1 = self.router.physical_to_grid_coords(*a)
            gx2, gy2 = self.router.physical_to_grid_coords(*b)
            r1, c1 = self.router.grid_coords_to_array_index(gx1, gy1)
            r2, c2 = self.router.grid_coords_to_array_index(gx2, gy2)
            avg_mult = (self.router.cost_grid[r1, c1] + self.router.cost_grid[r2, c2]) / 2.0
            weighted_cost = base_length * avg_mult
            if tuple(sorted((a, b))) in fixed_edges:
                weighted_cost *= (1.0 - shared_bonus)
            cost += weighted_cost
            length += base_length
        return cost, length

    def _update_shared_edges(self):
        self.shared_edges.clear()
        for route in self.routes.values():
            if route:
                for i in range(len(route) - 1):
                    edge = tuple(sorted((route[i], route[i + 1])))
                    self.shared_edges[edge] = self.shared_edges.get(edge, 0) + 1

    def visualize(self, save_path: Optional[str] = None, title: str = 'HRH Routing Result'):
        fig, ax = plt.subplots(figsize=(30, 15))
        draw_background(ax, self.router)
        colors = plt.cm.tab10.colors
        for i, route in self.routes.items():
            if route:
                xs, ys = zip(*route)
                ax.plot(xs, ys, color=colors[i % len(colors)], lw=2, alpha=0.9, zorder=4)
                ax.scatter(xs[0], ys[0], color=colors[i % len(colors)], marker='o', s=60, zorder=5, edgecolor='k')
                ax.scatter(xs[-1], ys[-1], color=colors[i % len(colors)], marker='s', s=60, zorder=5, edgecolor='k')
        self._update_shared_edges()
        shared_segments = [edge for edge, cnt in self.shared_edges.items() if cnt > 1]
        if shared_segments:
            shared_lc = LineCollection(shared_segments, colors='red', linewidths=4, alpha=0.8, zorder=5)
            ax.add_collection(shared_lc)
        ax.set_xlim(0, self.router.physical_width)
        ax.set_ylim(0, self.router.physical_height)
        ax.set_title(f'{title}\nShared Edges: {len(shared_segments)}')
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(save_path, dpi=300, bbox_inches='tight') if save_path else plt.show()


class BSplineSmoother:
    """
    B-样条路径平滑处理器 (增强版：支持最小转弯半径约束)
    """

    @staticmethod
    def simplify_path(path: List[Tuple[float, float]], min_dist: float = 1.0) -> List[Tuple[float, float]]:
        if len(path) < 3:
            return path
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            dist = math.hypot(path[i][0] - simplified[-1][0], path[i][1] - simplified[-1][1])
            if dist >= min_dist:
                simplified.append(path[i])
        if simplified[-1] != path[-1]:
            simplified.append(path[-1])
        return simplified

    @staticmethod
    def calculate_curvature(x_pts, y_pts):
        """
        计算离散点的曲率 (Kappa)
        Kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        """
        dx = np.gradient(x_pts)
        dy = np.gradient(y_pts)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # 避免除以零
        denominator = (dx ** 2 + dy ** 2) ** 1.5
        denominator[denominator < 1e-6] = 1e-6

        curvature = np.abs(dx * ddy - dy * ddx) / denominator
        return curvature

    @staticmethod
    def smooth_path(path: List[Tuple[float, float]],
                    router,
                    smoothness: float = 0.5,
                    degree: int = 3,
                    sample_density: int = 5,
                    min_turning_radius: float = 0.0) -> List[Tuple[float, float]]:
        """
        Args:
            min_turning_radius: 明确的物理转弯半径约束 (单位与坐标系一致)
        """
        if not path or len(path) < degree + 1:
            return path

        clean_path = BSplineSmoother.simplify_path(path, min_dist=router.cell_width * 1.5)
        if len(clean_path) <= degree:
            return path

        try:
            x = [p[0] for p in clean_path]
            y = [p[1] for p in clean_path]

            # --- 迭代优化以满足半径约束 ---
            current_s = smoothness
            best_path = None
            max_iterations = 5  # 避免死循环

            for _ in range(max_iterations):
                # 1. 生成样条
                tck, u = splprep([x, y], s=current_s, k=degree, per=False)

                # 2. 采样
                total_len = sum(
                    math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))
                num_points = int(total_len / (router.cell_width / float(sample_density)))
                if num_points < len(path): num_points = len(path) * 2

                u_new = np.linspace(0, 1, num_points)
                x_new, y_new = splev(u_new, tck)

                # 强制固定端点
                x_new[0], y_new[0] = path[0]
                x_new[-1], y_new[-1] = path[-1]

                # 3. 检查转弯半径
                if min_turning_radius > 0:
                    curvature = BSplineSmoother.calculate_curvature(x_new, y_new)
                    # 半径 R = 1 / curvature
                    # 找到最小的物理半径
                    valid_mask = curvature > 1e-6
                    if np.any(valid_mask):
                        min_R_found = np.min(1.0 / curvature[valid_mask])
                    else:
                        min_R_found = float('inf')  # 直线

                    # 如果当前最小半径小于要求的半径，说明太急了，需要增大平滑度
                    if min_R_found < min_turning_radius:
                        # print(f"  [BSpline] Radius {min_R_found:.1f} < Limit {min_turning_radius}. Increasing s...")
                        current_s *= 2.0  # 增大平滑因子，让曲线更直
                        continue  # 重试

                # 4. 检查碰撞
                temp_path = list(zip(x_new, y_new))
                if BSplineSmoother._is_path_safe(temp_path, router):
                    best_path = temp_path
                    break  # 成功找到
                else:
                    # 如果撞墙了，通常是因为平滑过度导致切角太严重
                    # 此时陷入两难：要半径大就要切角，切角就撞墙。
                    # 策略：如果为了满足半径而撞墙，则优先保证安全，回退到上一个可行解或放弃
                    # 这里简单的逻辑是：如果第一次就撞，直接放弃；如果是后续迭代撞，也许保持原来的好
                    pass

            return best_path if best_path else path

        except Exception as e:
            print(f"[BSpline] Error: {e}")
            return path

    @staticmethod
    def _is_path_safe(path: List[Tuple[float, float]], router) -> bool:
        for px, py in path:
            gx, gy = router.physical_to_grid_coords(px, py)
            if not (0 <= gx < router.grid_cols and 0 <= gy < router.grid_rows):
                return False
            row, col = router.grid_coords_to_array_index(gx, gy)
            if np.isinf(router.cost_grid[row, col]):
                return False
        return True

class GraphTheory:
    """
    一个工具类，用于从几何布局中直接生成布线骨架，并在此骨架上规划电缆路径。
    """

    @staticmethod
    def _calculate_plug_position(comp):
        direction = comp.get('connector')
        if not direction: return None
        x, y, l, w = comp['x'], comp['y'], comp['l'], comp['w']
        if direction == '左': return (x - w / 2, y)
        if direction == '右': return (x + w / 2, y)
        if direction == '上': return (x, y + l / 2)
        if direction == '下': return (x, y - l / 2)
        return None

    @staticmethod
    def _satisfies_direction_constraint(plug_pos, point, direction):
        plug_x, plug_y = plug_pos
        point_x, point_y = point
        if direction == '右': return point_x >= plug_x
        if direction == '左': return point_x <= plug_x
        if direction == '上': return point_y >= plug_y
        if direction == '下': return point_y <= plug_y
        return True

    @staticmethod
    def _find_closest_point_on_edges(plug_pos, edges, direction):
        min_dist, closest_pt, closest_edge_idx = float('inf'), None, None
        for edge_idx, edge in enumerate(edges):
            for point in edge:
                if GraphTheory._satisfies_direction_constraint(plug_pos, point, direction):
                    dist = math.hypot(point[0] - plug_pos[0], point[1] - plug_pos[1])
                    if dist < min_dist:
                        min_dist, closest_pt, closest_edge_idx = dist, tuple(point), edge_idx
        return closest_pt, closest_edge_idx

    @staticmethod
    def generate_skeleton_from_geometry(router: GridCableRouter, components: List[Dict], offset: Tuple):
        """
        直接从布局几何信息创建布线骨架网络，并包含精确的插头连接逻辑。
        返回: (keypoints, path_segments)
        """
        print("--- [1B.1] 从几何布局创建二值图像 ---")
        binary_img = np.isinf(router.cost_grid) == False
        binary_img = binary_img.astype(np.uint8)

        print("--- [1B.2] 提取拓扑骨架 ---")
        skeleton = skeletonize(binary_img)

        print("--- [1B.3] 从骨架像素构建初始图 ---")
        graph_sknw = sknw.build_sknw(skeleton)

        raw_nodes, raw_edges = set(), []
        for node_id in graph_sknw.nodes():
            r, c = graph_sknw.nodes[node_id]['o']
            raw_nodes.add(router.grid_coords_to_physical(*router.array_index_to_grid_coords(r, c)))

        for s, e in graph_sknw.edges():
            pts = graph_sknw[s][e]['pts']
            path = [router.grid_coords_to_physical(*router.array_index_to_grid_coords(r, c)) for r, c in pts]
            raw_edges.append(np.array(path))

        print(f"  初始图构建完成。节点: {len(raw_nodes)}, 边: {len(raw_edges)}")

        print("--- [1B.4] 精确连接插头到骨架网络 ---")
        final_nodes = list(raw_nodes)
        plugs_info = []
        for comp in components:
            plug_pos_local = GraphTheory._calculate_plug_position(comp)
            if plug_pos_local:
                plug_pos_global = (plug_pos_local[0] + offset[0], plug_pos_local[1] + offset[1])
                plugs_info.append({
                    'name': comp.get('name', 'N/A'),
                    'plug_pos': plug_pos_global,
                    'direction': comp.get('connector')
                })

        edge_splits = defaultdict(list)
        plug_connections = []

        for plug in plugs_info:
            pos, direction = plug['plug_pos'], plug['direction']
            closest_pt, edge_idx = GraphTheory._find_closest_point_on_edges(pos, raw_edges, direction)
            if closest_pt:
                final_nodes.append(pos)
                plug_connections.append((pos, closest_pt))
                edge_to_split = raw_edges[edge_idx]
                is_endpoint = (closest_pt == tuple(edge_to_split[0])) or (closest_pt == tuple(edge_to_split[-1]))
                if not is_endpoint:
                    edge_splits[edge_idx].append(closest_pt)

        final_edges = []
        for i, edge in enumerate(raw_edges):
            if i not in edge_splits:
                final_edges.append(edge)
            else:
                split_points = sorted(list(set(edge_splits[i])), key=lambda p: np.linalg.norm(p - edge[0]))
                all_pts = [tuple(edge[0])] + split_points + [tuple(edge[-1])]
                for j in range(len(all_pts) - 1):
                    start_idx = np.where((edge == all_pts[j]).all(axis=1))[0][0]
                    end_idx = np.where((edge == all_pts[j + 1]).all(axis=1))[0][0]
                    if start_idx < end_idx:
                        final_edges.append(edge[start_idx:end_idx + 1])
                    else:
                        final_edges.append(np.flip(edge[end_idx:start_idx + 1], axis=0))

        for plug_pos, closest_pt in plug_connections:
            final_edges.append(np.array([plug_pos, closest_pt]))
            if closest_pt not in final_nodes:
                final_nodes.append(closest_pt)

        keypoints = set(final_nodes)
        extractor = TopologyExtractor(router, {})
        path_segments = {}
        for edge in final_edges:
            segment = [tuple(p) for p in edge]
            cost = extractor.calculate_segment_cost(segment)
            path_segments[tuple(sorted((segment[0], segment[-1])))] = {'path': segment, 'cost': cost}

        print(f"  插头连接完成。最终网络节点: {len(keypoints)}, 路径段: {len(path_segments)}")

        return keypoints, path_segments, raw_nodes, raw_edges


    @staticmethod
    def generate_complete_skeleton(router: GridCableRouter, components: List[Dict], offset: Tuple):
        """
        (已重构) 从布局几何信息创建完整的布线骨架网络。
        此函数现在只负责构建和返回包含所有潜在路径的骨架。

        返回:
            (keypoints, path_segments, raw_nodes, raw_edges)
            - keypoints: 包含所有接插件和交叉点的节点集。
            - path_segments: 描述节点间连接的路径段字典。
            - raw_nodes, raw_edges: 用于可视化基础路网的原始数据。
        """
        print("--- [1A] 从几何布局创建基础路网 ---")
        binary_img = np.isinf(router.cost_grid) == False
        binary_img = binary_img.astype(np.uint8)
        skeleton = skeletonize(binary_img)
        graph_sknw = sknw.build_sknw(skeleton)

        raw_nodes, raw_edges = set(), []
        for node_id in graph_sknw.nodes():
            r, c = graph_sknw.nodes[node_id]['o']
            raw_nodes.add(router.grid_coords_to_physical(*router.array_index_to_grid_coords(r, c)))
        for s, e in graph_sknw.edges():
            pts = graph_sknw[s][e]['pts']
            path = [router.grid_coords_to_physical(*router.array_index_to_grid_coords(r, c)) for r, c in pts]
            raw_edges.append(np.array(path))
        print(f"  -> 基础路网构建完成。节点: {len(raw_nodes)}, 边: {len(raw_edges)}")

        print("--- [1B] 连接接插件，构建完整骨架 ---")
        final_nodes = list(raw_nodes)
        plugs_info = []
        for comp in components:
            plug_pos_local = GraphTheory._calculate_plug_position(comp)
            if plug_pos_local:
                plug_pos_global = (plug_pos_local[0] + offset[0], plug_pos_local[1] + offset[1])
                plugs_info.append({
                    'name': comp.get('name', 'N/A'),
                    'plug_pos': plug_pos_global,
                    'direction': comp.get('connector')
                })

        edge_splits = defaultdict(list)
        plug_connections = []

        for plug in plugs_info:
            pos, direction = plug['plug_pos'], plug['direction']
            closest_pt, edge_idx = GraphTheory._find_closest_point_on_edges(pos, raw_edges, direction)
            if closest_pt:
                final_nodes.append(pos)
                plug_connections.append((pos, closest_pt))
                edge_to_split = raw_edges[edge_idx]
                is_endpoint = (closest_pt == tuple(edge_to_split[0])) or (closest_pt == tuple(edge_to_split[-1]))
                if not is_endpoint:
                    edge_splits[edge_idx].append(closest_pt)

        final_edges = []
        for i, edge in enumerate(raw_edges):
            if i not in edge_splits:
                final_edges.append(edge)
            else:
                split_points = sorted(list(set(edge_splits[i])), key=lambda p: np.linalg.norm(p - edge[0]))
                all_pts = [tuple(edge[0])] + split_points + [tuple(edge[-1])]
                for j in range(len(all_pts) - 1):
                    start_idx = np.where((edge == all_pts[j]).all(axis=1))[0][0]
                    end_idx = np.where((edge == all_pts[j + 1]).all(axis=1))[0][0]
                    if start_idx < end_idx:
                        final_edges.append(edge[start_idx:end_idx + 1])
                    else:
                        final_edges.append(np.flip(edge[end_idx:start_idx + 1], axis=0))

        for plug_pos, closest_pt in plug_connections:
            final_edges.append(np.array([plug_pos, closest_pt]))
            if closest_pt not in final_nodes:
                final_nodes.append(closest_pt)

        keypoints = set(final_nodes)
        extractor = TopologyExtractor(router, {})
        path_segments = {}
        for edge in final_edges:
            segment = [tuple(p) for p in edge]
            if len(segment) < 2: continue  # 忽略长度为1的无效段
            cost = extractor.calculate_segment_cost(segment)
            edge_key = tuple(sorted((segment[0], segment[-1])))
            if edge_key[0] == edge_key[1]: continue  # 忽略自环
            path_segments[edge_key] = {'path': segment, 'cost': cost}

        print(f"  -> 完整骨架构建完成。关键节点: {len(keypoints)}, 路径段: {len(path_segments)}")

        return keypoints, path_segments, raw_nodes, raw_edges

    @staticmethod
    def route_cables_on_skeleton(
            router: GridCableRouter,
            keypoints: Set[Tuple[float, float]],
            path_segments: Dict,
            demands: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> Dict[int, List[Tuple[float, float]]]:
        """
        (Dijkstra 方法) 在预先计算的拓扑骨架上为一组电缆需求规划路径。
        对每个电缆需求，在骨架图上执行最短路径搜索。
        """
        print("\n--- [1C] 在几何骨架上规划电缆需求路径 (Dijkstra) ---")
        G = nx.Graph()
        for (u, v), segment_info in path_segments.items():
            cost = segment_info['cost']
            if math.isfinite(cost) and cost > 0:
                G.add_edge(u, v, weight=cost, path=segment_info['path'])

        if not G.nodes():
            print("  错误: 骨架图为空，无法规划路径。")
            return {i: [] for i in range(len(demands))}

        graph_nodes = list(G.nodes())
        terminal_map = {}
        all_terminals = {pt for demand in demands for pt in demand}

        # 将每个电缆端点“吸附”到骨架上最近的节点
        for term in all_terminals:
            closest_node = min(graph_nodes, key=lambda n: math.hypot(n[0] - term[0], n[1] - term[1]))
            terminal_map[term] = closest_node

        routes = {}
        for i, (start_demand, end_demand) in enumerate(demands):
            snapped_start = terminal_map.get(start_demand)
            snapped_end = terminal_map.get(end_demand)

            if not snapped_start or not snapped_end:
                print(f"  警告: 无法将需求 {i} 映射到骨架。回退到全局A*算法。")
                routes[i] = router.find_path_single_cable(start_demand, end_demand)
                continue

            try:
                # 在骨架图上找到节点间的最短路径
                path_nodes = nx.shortest_path(G, source=snapped_start, target=snapped_end, weight='weight')
                full_path = []

                # 拼接：实际起点 -> 吸附点
                if start_demand != snapped_start:
                    path_to_snap = router.find_path_single_cable(start_demand, snapped_start)
                    full_path.extend(path_to_snap or [start_demand, snapped_start])

                # 拼接：骨架内的核心路径
                core_path = [path_nodes[0]]
                for j in range(len(path_nodes) - 1):
                    u, v = path_nodes[j], path_nodes[j + 1]
                    segment = G.get_edge_data(u, v)['path']
                    # 确保路径段方向正确
                    if segment[0] != u:
                        segment = segment[::-1]
                    core_path.extend(segment[1:])

                # 合并路径，避免重复点
                if full_path and core_path and full_path[-1] == core_path[0]:
                    full_path.extend(core_path[1:])
                else:
                    full_path.extend(core_path)

                # 拼接：吸附点 -> 实际终点
                if end_demand != snapped_end:
                    path_from_snap = router.find_path_single_cable(snapped_end, end_demand)
                    if full_path and path_from_snap and full_path[-1] == path_from_snap[0]:
                        full_path.extend(path_from_snap[1:])
                    else:
                        full_path.extend(path_from_snap)

                routes[i] = full_path

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"  警告: 无法在骨架上为需求 {i} 找到路径。回退到全局A*算法。")
                routes[i] = router.find_path_single_cable(start_demand, end_demand)

        print(f"  成功在骨架上规划了 {len(routes)} 条电缆路径。")
        return routes

    @staticmethod
    def milp_route_on_skeleton(
            router: GridCableRouter,
            keypoints: Set[Tuple[float, float]],
            path_segments: Dict,
            demands: List[Tuple[Tuple[float, float], Tuple[float, float]]],
            time_limit_seconds: int
    ) -> Dict[int, List[Tuple[float, float]]]:
        """
        (MILP 方法) 在给定的完整骨架上，使用MILP多商品流模型为所有电缆规划最短路径。
        """
        print("\n--- [1C] 在完整骨架上执行MILP初始寻路 (非树形) ---")

        # 直接调用通用的MILP寻路函数
        paths_list = milp_shortest_paths_for_all_cables(
            keypoints, path_segments, demands, router, time_limit_seconds
        )

        # 将结果转换为字典格式以保持一致性
        initial_routes = {i: p for i, p in enumerate(paths_list)}

        # 检查是否有失败的路径，并尝试用A*回退
        for i, path in initial_routes.items():
            if not path:
                print(f"  [警告] MILP 未能为电缆 {i} 找到路径。尝试使用 A* 回退。")
                start_demand, end_demand = demands[i]
                fallback_path = router.find_path_single_cable(start_demand, end_demand)
                initial_routes[i] = fallback_path or []
                if not fallback_path:
                    print(f"  [错误] A* 回退也未能为电缆 {i} 找到路径。")

        print(
            f"  -> MILP 初始寻路完成。成功规划了 {len([p for p in initial_routes.values() if p])}/{len(demands)} 条电缆。")
        return initial_routes

    @staticmethod
    def run(router, components, demands, offset):
        """
        执行基于几何的布线全流程。
        返回值也包括原始骨架数据。
        """
        print("=== [1] 基于几何布局的初始布线 ===")
        keypoints, path_segments, raw_nodes, raw_edges = GraphTheory.generate_skeleton_from_geometry(router, components, offset
        )
        initial_routes = GraphTheory.route_cables_on_skeleton(router, keypoints, path_segments, demands)

        return initial_routes, raw_nodes, raw_edges

    @staticmethod
    def visualize_raw_skeleton(router: GridCableRouter, raw_nodes: Set, raw_edges: List,
                               save_path: str, title: str):
        """
        [新增] 独立的可视化函数，用于绘制从图像生成的原始路网骨架。
        """
        print(f"  -> 正在可视化原始路网骨架...")
        fig, ax = plt.subplots(figsize=(20, 10))
        draw_background(ax, router)
        lc = LineCollection(raw_edges, colors='purple', linewidths=2, alpha=0.9, zorder=4)
        ax.add_collection(lc)
        if raw_nodes:
            kx, ky = zip(*raw_nodes)
            ax.scatter(kx, ky, color='blue', marker='o', s=50, zorder=10, label='骨架节点')
        ax.set_xlim(0, router.physical_width)
        ax.set_ylim(0, router.physical_height)
        ax.set_title(title)
        ax.legend()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"     已保存图像到: {save_path}")

def mst_based_routing(keypoints: Set[Tuple[float, float]], path_segments: Dict, demands: List,
                      router: 'GridCableRouter') -> List:
    G = nx.Graph()
    extractor = TopologyExtractor(router, {})
    for (u, v), segment_info in path_segments.items():
        cost = segment_info['cost']
        if math.isfinite(cost) and cost > 0:
            G.add_edge(u, v, weight=cost, path=segment_info['path'])

    terminals = {pt for demand in demands for pt in demand}
    snapped_terminals, terminal_map = set(), {}
    for term in terminals:
        gx, gy = router.physical_to_grid_coords(*term)
        snapped_term = router.grid_coords_to_physical(gx, gy)
        snapped_terminals.add(snapped_term)
        terminal_map[term] = snapped_term
        if snapped_term not in G:
            nearest_kp = min((kp for kp in keypoints if kp in G),
                             key=lambda kp: math.hypot(snapped_term[0] - kp[0], snapped_term[1] - kp[1]),
                             default=None)
            if nearest_kp:
                path_to_kp = router.find_path_single_cable(snapped_term, nearest_kp)
                if path_to_kp:
                    cost_to_kp = extractor.calculate_segment_cost(path_to_kp)
                    G.add_edge(snapped_term, nearest_kp, weight=cost_to_kp, path=path_to_kp)

    try:
        connected_terminals = [t for t in snapped_terminals if t in G]
        if len(connected_terminals) < 2: return []
        steiner = steiner_tree(G, connected_terminals, weight='weight')
    except Exception as e:
        # 回退到MST方法
        print(f"Steiner tree approximation failed ({e}), falling back to MST.")
        valid_terminals_in_G = [t for t in connected_terminals if t in G]
        subgraph = G.subgraph(valid_terminals_in_G).copy()
        largest_cc = max(nx.connected_components(subgraph), key=len)
        steiner = nx.minimum_spanning_tree(subgraph.subgraph(largest_cc), weight='weight')

    optimized_paths = []
    for src, tgt in demands:
        snapped_src, snapped_tgt = terminal_map.get(src), terminal_map.get(tgt)
        try:
            if snapped_src not in steiner or snapped_tgt not in steiner: raise nx.NetworkXNoPath
            path_nodes = nx.shortest_path(steiner, snapped_src, snapped_tgt)
            full_path = [path_nodes[0]]
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                if G.has_edge(u, v):
                    segment = G.get_edge_data(u, v).get('path', [u, v])
                    full_path.extend(segment[1:] if segment[0] == u else reversed(segment[:-1]))
                else:
                    full_path.append(v)
            final_path = [p for i, p in enumerate(full_path) if i == 0 or p != full_path[i - 1]]
            optimized_paths.append(final_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            optimized_paths.append(router.find_path_single_cable(src, tgt) or [src, tgt])
    return optimized_paths


def spt_heuristic_mrct_robust(G: nx.Graph, terminals: list, demands_snapped: list, weight='weight') -> nx.Graph:
    """
    (REVISED - DEMAND AWARE VERSION)
    使用最短路径树 (SPT) 启发式算法来近似求解路由问题。

    改进点：
    1. 评估标准变更：不再计算所有终端的全连接 (All-Pairs) 成本，而是只计算 demands 列表中
       指定的端点对之间的距离之和。这使得优化目标与实际布线需求一致。
    2. 复杂度降低：计算成本的复杂度从 O(K^2 * V log V) 降低到 O(|D| * V log V)。
    """
    best_tree = None
    min_total_routing_cost = float('inf')

    if len(terminals) < 2:
        return nx.Graph()

    # 为了快速查找，将终端列表转为集合（虽然这里主要依赖 Dijkstra 结果）
    terminals_set = set(terminals)

    # 1. 遍历每个终端，将其作为候选根
    for root_candidate in terminals:

        # --- 步骤 2: 使用 Dijkstra 算法一次性计算从根出发的最短路径 ---
        try:
            paths_from_root = nx.single_source_dijkstra_path(G, source=root_candidate, weight=weight)
        except nx.NodeNotFound:
            continue

        # --- 步骤 3: 检查此根是否能连接到所有其他终端 ---
        all_terminals_connected = True
        for target in terminals:
            if target not in paths_from_root:
                all_terminals_connected = False
                break

        if not all_terminals_connected:
            continue

        # --- 步骤 4: "约束式构建" SPT ---
        current_spt = nx.Graph()
        for target in terminals:
            path_to_target = paths_from_root[target]
            nx.add_path(current_spt, path_to_target)

        # --- 步骤 5: [关键修改] 基于 Demands 计算总路由成本 ---
        # 此时的 current_spt 已被保证为一棵连接所有终端的树
        total_cost_for_this_spt = 0
        valid_tree_for_demands = True

        for u, v in demands_snapped:
            # 防御性检查：确保需求点都在当前构建的树中
            if u not in current_spt or v not in current_spt:
                # 这种情况理论上不应发生（因为前面已经检查了 all_terminals_connected），
                # 但为了鲁棒性，如果发生则视为此树不可用
                valid_tree_for_demands = False
                break

            try:
                # 在我们构建的树上计算特定需求 u->v 的距离
                # 注意：这里只计算实际需要的路径，而不是所有组合
                cost = nx.shortest_path_length(current_spt, source=u, target=v, weight=weight)
                total_cost_for_this_spt += cost
            except nx.NetworkXNoPath:
                valid_tree_for_demands = False
                break

        if not valid_tree_for_demands:
            continue

        # --- 步骤 6: 如果当前树更优（基于实际需求的总线长更短），则保存它 ---
        if total_cost_for_this_spt < min_total_routing_cost:
            min_total_routing_cost = total_cost_for_this_spt

            # 构建最终树对象
            final_tree = nx.Graph()
            final_tree.add_nodes_from(current_spt.nodes())
            for u, v in current_spt.edges():
                final_tree.add_edge(u, v, **G.get_edge_data(u, v, default={}))
            best_tree = final_tree

    return best_tree


def spt_based_routing(keypoints: Set[Tuple[float, float]], path_segments: Dict, demands: List,
                      router: 'GridCableRouter') -> List:
    """
    此函数使用 SPT 启发式算法，已更新为将 'demands' 传递给核心算法以进行针对性优化。
    """
    G = nx.Graph()
    extractor = TopologyExtractor(router, {})
    for (u, v), segment_info in path_segments.items():
        cost = segment_info['cost']
        if math.isfinite(cost) and cost > 0:
            G.add_edge(u, v, weight=cost, path=segment_info['path'])

    terminals = {pt for demand in demands for pt in demand}
    snapped_terminals, terminal_map = set(), {}

    # 构建 terminal_map 以及 snapped_terminals 集合
    for term in terminals:
        gx, gy = router.physical_to_grid_coords(*term)
        snapped_term = router.grid_coords_to_physical(gx, gy)
        terminal_map[term] = snapped_term  # 记录原始坐标到吸附坐标的映射

        snapped_terminals.add(snapped_term)

        if snapped_term not in G:
            nearest_kp = min((kp for kp in keypoints if kp in G),
                             key=lambda kp: math.hypot(snapped_term[0] - kp[0], snapped_term[1] - kp[1]),
                             default=None)
            if nearest_kp:
                path_to_kp = router.find_path_single_cable(snapped_term, nearest_kp)
                if path_to_kp:
                    cost_to_kp = extractor.calculate_segment_cost(path_to_kp)
                    G.add_edge(snapped_term, nearest_kp, weight=cost_to_kp, path=path_to_kp)

    # [新增步骤] 预处理 snapped_demands
    # 将原始需求的坐标转换为图 G 中的节点坐标 (snapped coordinates)
    # 这将传递给启发式算法用于计算成本
    snapped_demands = []
    for src, tgt in demands:
        s_src = terminal_map.get(src)
        s_tgt = terminal_map.get(tgt)
        if s_src and s_tgt:
            snapped_demands.append((s_src, s_tgt))

    # 步骤 3: 尝试使用 SPT 启发式构建树
    routing_tree = None
    connected_terminals = sorted(list(t for t in snapped_terminals if t in G))

    try:
        if len(connected_terminals) < 2: return []
        print("Attempting to build routing tree using 'SPT Heuristic (Demand-Aware)' strategy...")

        # [关键修改] 传入 snapped_demands
        routing_tree = spt_heuristic_mrct_robust(G, connected_terminals, snapped_demands, weight='weight')

        if routing_tree is None:
            raise ValueError("SPT heuristic failed to find a spanning tree connecting all terminals.")

    except Exception as e:
        # **回退到MST方法**
        print("\n" + "=" * 60)
        print(f"[WARNING] SPT Heuristic failed: {e}.")
        print("          Falling back to Minimum Spanning Tree (MST) strategy.")
        print("=" * 60 + "\n")

        valid_terminals_in_G = [t for t in connected_terminals if t in G]
        subgraph = G.subgraph(valid_terminals_in_G).copy()
        largest_cc = max(nx.connected_components(subgraph), key=len)
        cc_subgraph = subgraph.subgraph(largest_cc)
        routing_tree = nx.minimum_spanning_tree(cc_subgraph, weight='weight')

    if routing_tree and routing_tree.number_of_nodes() > 0:
        num_nodes = routing_tree.number_of_nodes()
        num_edges = routing_tree.number_of_edges()
        print(f"\n  [Planning Tree Stats]")
        print(f"  - Nodes in the planning tree: {num_nodes}")
        print(f"  - Edges in the planning tree: {num_edges}")
        if num_edges == num_nodes - 1:
            print("  - Verification: Success, |E| = |V| - 1. The planning topology is a tree.")
        else:
            print(f"  - Verification: FAILED, |E| != |V| - 1. The planning topology is NOT a tree.")
        print("-" * 25)

    # 路径重建部分 (保持不变)
    optimized_paths = []
    for src, tgt in demands:
        snapped_src, snapped_tgt = terminal_map.get(src), terminal_map.get(tgt)
        try:
            if snapped_src not in routing_tree or snapped_tgt not in routing_tree: raise nx.NetworkXNoPath
            path_nodes = nx.shortest_path(routing_tree, snapped_src, snapped_tgt)
            full_path = [path_nodes[0]]
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                if G.has_edge(u, v):
                    segment = G.get_edge_data(u, v).get('path', [u, v])
                    full_path.extend(segment[1:] if segment[0] == u else reversed(segment[:-1]))
                else:
                    full_path.append(v)
            final_path = [p for i, p in enumerate(full_path) if i == 0 or p != full_path[i - 1]]
            optimized_paths.append(final_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            optimized_paths.append(router.find_path_single_cable(src, tgt) or [src, tgt])
    return optimized_paths

# =================================================================================
# [已完善] MILP 优化函数
# =================================================================================
def milp_scip_based_routing(keypoints: Set[Tuple[float, float]],
                            path_segments: Dict, demands: List,
                            router: 'GridCableRouter',
                            time_limit_seconds: int = 300) -> List[List[Tuple[float, float]]]:
    """
    使用 MILP 找到最优的树形线束，并返回每条电缆的详细几何路径。
    (已更新为使用显式树约束和最小化总线缆长度的目标)
    """
    print("\n--- 3. 开始 MILP 优化过程 ---")
    start_time = time.time()

    # --- 图的构建与增强 (无变化) ---
    G = nx.Graph()

    extractor = TopologyExtractor(router, {})
    for (u, v), segment_info in path_segments.items():
        cost = segment_info['cost']
        if math.isfinite(cost) and cost > 0:
            G.add_edge(u, v, weight=cost, path=segment_info['path'])

    terminals = {pt for demand in demands for pt in demand}
    terminal_map_to_snapped = {}
    for term in terminals:
        gx, gy = router.physical_to_grid_coords(*term)
        snapped_term = router.grid_coords_to_physical(gx, gy)
        terminal_map_to_snapped[term] = snapped_term
        if snapped_term not in G:
            nearest_kp = min((kp for kp in keypoints if kp in G),
                             key=lambda kp: math.hypot(snapped_term[0] - kp[0], snapped_term[1] - kp[1]),
                             default=None)
            if nearest_kp:
                path_to_kp = router.find_path_single_cable(snapped_term, nearest_kp)
                if path_to_kp:
                    # cost_to_kp = extractor.calculate_segment_cost(path_to_kp)
                    cost_to_kp = sum(
                        math.hypot(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in zip(path_to_kp, path_to_kp[1:]))
                    G.add_edge(snapped_term, nearest_kp, weight=cost_to_kp, path=path_to_kp)

    # --- 步骤 A: 数据预处理 (无变化) ---
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)
    num_demands = len(demands)
    snapped_demands = []
    for start_demand, end_demand in demands:
        snapped_start_node = terminal_map_to_snapped[start_demand]
        snapped_end_node = terminal_map_to_snapped[end_demand]
        snapped_demands.append((node_to_idx[snapped_start_node], node_to_idx[snapped_end_node]))

    # --- 步骤 B: 初始化求解器并创建变量 (无变化) ---
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver not available.")
    x_k = {}
    x = {}
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_key = tuple(sorted((u_idx, v_idx)))
        x[edge_key] = solver.BoolVar(f'x_{edge_key[0]}_{edge_key[1]}')
        for k in range(num_demands):
            x_k[k, u_idx, v_idx] = solver.BoolVar(f'xk_{k}_{u_idx}_{v_idx}')
            x_k[k, v_idx, u_idx] = solver.BoolVar(f'xk_{k}_{v_idx}_{u_idx}')
    node_used = {i: solver.BoolVar(f'node_used_{i}') for i in range(num_nodes)}

    # --- 步骤 C: 添加约束 ---
    # 1. 需求路径约束 (流量守恒)
    for k in range(num_demands):
        start_idx, end_idx = snapped_demands[k]
        for i in range(num_nodes):
            node = node_list[i]
            in_flow = solver.Sum(x_k.get((k, node_to_idx[j], i), 0) for j in G.neighbors(node))
            out_flow = solver.Sum(x_k.get((k, i, node_to_idx[j]), 0) for j in G.neighbors(node))
            if i == start_idx:
                solver.Add(out_flow - in_flow == 1)
            elif i == end_idx:
                solver.Add(in_flow - out_flow == 1)
            else:
                solver.Add(in_flow == out_flow)

    # 2. 关联 x_k 和 x (无变化)
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_key = tuple(sorted((u_idx, v_idx)))
        for k in range(num_demands):
            solver.Add(x[edge_key] >= x_k[k, u_idx, v_idx])
            solver.Add(x[edge_key] >= x_k[k, v_idx, u_idx])

    # 3. 关联 x 和 node_used (无变化)
    for i in range(num_nodes):
        node = node_list[i]
        incident_edges = solver.Sum(x[tuple(sorted((i, node_to_idx[j])))] for j in G.neighbors(node))
        solver.Add(node_used[i] <= incident_edges)
        solver.Add(incident_edges <= G.degree(node) * node_used[i])

    # *** 4.显式的树约束 ***
    # 强制要求被选中的边的总数等于被使用节点的总数减一。
    # 这与下面的连通性约束相结合，确保最终的拓扑结构是一棵树。
    total_selected_edges = solver.Sum(x.values())
    total_used_nodes = solver.Sum(node_used.values())
    solver.Add(total_selected_edges == total_used_nodes - 1, "TreeConstraint_EdgesNodes")

    # 5. 连通性约束 (使用流量模型，无变化)
    root_node_idx = snapped_demands[0][0]
    max_flow = num_nodes - 1
    flow = {}
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_key = tuple(sorted((u_idx, v_idx)))
        flow[u_idx, v_idx] = solver.IntVar(0, max_flow, f'flow_{u_idx}_{v_idx}')
        flow[v_idx, u_idx] = solver.IntVar(0, max_flow, f'flow_{v_idx}_{u_idx}')
        solver.Add(flow[u_idx, v_idx] + flow[v_idx, u_idx] <= max_flow * x[edge_key])

    total_flow_demand = solver.Sum(node_used[i] for i in range(num_nodes) if i != root_node_idx)
    out_flow_root = solver.Sum(flow[root_node_idx, node_to_idx[j]] for j in G.neighbors(node_list[root_node_idx]))
    in_flow_root = solver.Sum(flow[node_to_idx[j], root_node_idx] for j in G.neighbors(node_list[root_node_idx]))
    solver.Add(out_flow_root - in_flow_root == total_flow_demand)

    for i in range(num_nodes):
        if i != root_node_idx:
            node = node_list[i]
            in_flow_node = solver.Sum(flow[node_to_idx[j], i] for j in G.neighbors(node))
            out_flow_node = solver.Sum(flow[i, node_to_idx[j]] for j in G.neighbors(node))
            solver.Add(in_flow_node - out_flow_node == node_used[i])

    # --- 步骤 D: 定义目标函数 (*** 此处已修改 ***) ---
    # objective_expr = solver.Sum(
    #     G.edges[u, v]['weight'] * x[tuple(sorted((node_to_idx[u], node_to_idx[v])))] for u, v in G.edges())
    # solver.Minimize(objective_expr)
    # 新目标: 最小化所有电缆的总长度。
    # 因为我们已经通过显式约束强制了树形结构，所以可以安全地使用这个目标。
    total_wire_length = solver.Sum(
        G.edges[u, v]['weight'] * (solver.Sum(
            x_k[k, node_to_idx[u], node_to_idx[v]] + x_k[k, node_to_idx[v], node_to_idx[u]] for k in
            range(num_demands)))
        for u, v in G.edges()
    )
    solver.Minimize(total_wire_length)

    # --- 步骤 E: 求解模型 (无变化) ---
    print("  d) 开始求解 MILP 模型...")
    solver.EnableOutput()

    solver.SetTimeLimit(time_limit_seconds * 1000)
    status = solver.Solve()


    # --- 步骤 F: 结果解析 (无变化) ---
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        print("求解失败，未找到解决方案。")
        return [[] for _ in demands]

    print(f"  e) 求解成功！所有电缆总长度: {solver.Objective().Value():.2f}")

    # (结果解析部分与之前完全相同)
    H = nx.Graph()
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_key = tuple(sorted((u_idx, v_idx)))
        if x[edge_key].solution_value() > 0.5:
            H.add_edge(u, v, **G.edges[u, v])

    if H.number_of_nodes() > 0:
        num_nodes = H.number_of_nodes()
        num_edges = H.number_of_edges()
        print(f"\n  [MILP Solution Graph Stats]")
        print(f"  - Nodes in the solution graph: {num_nodes}")
        print(f"  - Edges in the solution graph: {num_edges}")
        # 这个函数不强制树结构，所以不进行 |E|=|V|-1 的验证
        print("-" * 25)

    optimized_paths = []
    for start_demand, end_demand in demands:
        snapped_start = terminal_map_to_snapped[start_demand]
        snapped_end = terminal_map_to_snapped[end_demand]
        try:
            path_nodes = nx.shortest_path(H, source=snapped_start, target=snapped_end, weight='weight')
            full_geometric_path = []
            if start_demand != snapped_start:
                path_to_snap = router.find_path_single_cable(start_demand, snapped_start)
                if path_to_snap:
                    full_geometric_path.extend(path_to_snap[:-1])
                else:
                    full_geometric_path.append(start_demand)
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                segment = H.get_edge_data(u, v)['path']
                if segment[0] != u: segment = segment[::-1]
                if full_geometric_path:
                    full_geometric_path.extend(segment[1:])
                else:
                    full_geometric_path.extend(segment)
            if end_demand != snapped_end:
                path_from_snap = router.find_path_single_cable(snapped_end, end_demand)
                if path_from_snap:
                    if full_geometric_path:
                        full_geometric_path.extend(path_from_snap[1:])
                    else:
                        full_geometric_path.extend(path_from_snap)
                elif not full_geometric_path:
                    full_geometric_path.append(end_demand)
            optimized_paths.append(full_geometric_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"  警告: 无法在线束子图上为需求 {start_demand} -> {end_demand} 找到路径。")
            optimized_paths.append([])

    print(f"MILP 优化及路径重建总耗时: {time.time() - start_time:.2f} 秒。")
    return optimized_paths


def milp_shortest_paths_for_all_cables(keypoints: Set[Tuple[float, float]],
                            path_segments: Dict, demands: List,
                            router: 'GridCableRouter',
                            time_limit_seconds: int = 300) -> List[List[Tuple[float, float]]]:
    """
    使用 MILP 找到最优的树形线束，并返回每条电缆的详细几何路径。
    (已更新为使用显式树约束和最小化总线缆长度的目标)
    """
    print("\n--- 3. 开始 MILP 优化过程 ---")
    start_time = time.time()

    # --- 图的构建与增强 (无变化) ---
    G = nx.Graph()
    # 假设 TopologyExtractor 是一个有效的类
    # extractor = TopologyExtractor(router, {})
    for (u, v), segment_info in path_segments.items():
        cost = segment_info['cost']
        if math.isfinite(cost) and cost > 0:
            G.add_edge(u, v, weight=cost, path=segment_info['path'])

    terminals = {pt for demand in demands for pt in demand}
    terminal_map_to_snapped = {}
    for term in terminals:
        gx, gy = router.physical_to_grid_coords(*term)
        snapped_term = router.grid_coords_to_physical(gx, gy)
        terminal_map_to_snapped[term] = snapped_term
        if snapped_term not in G:
            nearest_kp = min((kp for kp in keypoints if kp in G),
                             key=lambda kp: math.hypot(snapped_term[0] - kp[0], snapped_term[1] - kp[1]),
                             default=None)
            if nearest_kp:
                path_to_kp = router.find_path_single_cable(snapped_term, nearest_kp)
                if path_to_kp:
                    # cost_to_kp = extractor.calculate_segment_cost(path_to_kp)
                    cost_to_kp = sum(
                        math.hypot(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in zip(path_to_kp, path_to_kp[1:]))
                    G.add_edge(snapped_term, nearest_kp, weight=cost_to_kp, path=path_to_kp)

    # --- 步骤 A: 数据预处理 (无变化) ---
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)
    num_demands = len(demands)
    snapped_demands = []
    for start_demand, end_demand in demands:
        snapped_start_node = terminal_map_to_snapped[start_demand]
        snapped_end_node = terminal_map_to_snapped[end_demand]
        snapped_demands.append((node_to_idx[snapped_start_node], node_to_idx[snapped_end_node]))

    # --- 步骤 B: 初始化求解器并创建变量 (无变化) ---
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver not available.")
    x_k = {}
    x = {}
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_key = tuple(sorted((u_idx, v_idx)))
        x[edge_key] = solver.BoolVar(f'x_{edge_key[0]}_{edge_key[1]}')
        for k in range(num_demands):
            x_k[k, u_idx, v_idx] = solver.BoolVar(f'xk_{k}_{u_idx}_{v_idx}')
            x_k[k, v_idx, u_idx] = solver.BoolVar(f'xk_{k}_{v_idx}_{u_idx}')
    node_used = {i: solver.BoolVar(f'node_used_{i}') for i in range(num_nodes)}

    # --- 步骤 C: 添加约束 ---
    # 1. 需求路径约束 (流量守恒)
    for k in range(num_demands):
        start_idx, end_idx = snapped_demands[k]
        for i in range(num_nodes):
            node = node_list[i]
            in_flow = solver.Sum(x_k.get((k, node_to_idx[j], i), 0) for j in G.neighbors(node))
            out_flow = solver.Sum(x_k.get((k, i, node_to_idx[j]), 0) for j in G.neighbors(node))
            if i == start_idx:
                solver.Add(out_flow - in_flow == 1)
            elif i == end_idx:
                solver.Add(in_flow - out_flow == 1)
            else:
                solver.Add(in_flow == out_flow)

    # 2. 关联 x_k 和 x (无变化)
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_key = tuple(sorted((u_idx, v_idx)))
        for k in range(num_demands):
            solver.Add(x[edge_key] >= x_k[k, u_idx, v_idx])
            solver.Add(x[edge_key] >= x_k[k, v_idx, u_idx])

    # 3. 关联 x 和 node_used (无变化)
    for i in range(num_nodes):
        node = node_list[i]
        incident_edges = solver.Sum(x[tuple(sorted((i, node_to_idx[j])))] for j in G.neighbors(node))
        solver.Add(node_used[i] <= incident_edges)
        solver.Add(incident_edges <= G.degree(node) * node_used[i])

    # 5. 连通性约束 (使用流量模型，无变化)
    root_node_idx = snapped_demands[0][0]
    max_flow = num_nodes - 1
    flow = {}
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_key = tuple(sorted((u_idx, v_idx)))
        flow[u_idx, v_idx] = solver.IntVar(0, max_flow, f'flow_{u_idx}_{v_idx}')
        flow[v_idx, u_idx] = solver.IntVar(0, max_flow, f'flow_{v_idx}_{u_idx}')
        solver.Add(flow[u_idx, v_idx] + flow[v_idx, u_idx] <= max_flow * x[edge_key])

    total_flow_demand = solver.Sum(node_used[i] for i in range(num_nodes) if i != root_node_idx)
    out_flow_root = solver.Sum(flow[root_node_idx, node_to_idx[j]] for j in G.neighbors(node_list[root_node_idx]))
    in_flow_root = solver.Sum(flow[node_to_idx[j], root_node_idx] for j in G.neighbors(node_list[root_node_idx]))
    solver.Add(out_flow_root - in_flow_root == total_flow_demand)

    for i in range(num_nodes):
        if i != root_node_idx:
            node = node_list[i]
            in_flow_node = solver.Sum(flow[node_to_idx[j], i] for j in G.neighbors(node))
            out_flow_node = solver.Sum(flow[i, node_to_idx[j]] for j in G.neighbors(node))
            solver.Add(in_flow_node - out_flow_node == node_used[i])


    total_wire_length = solver.Sum(
        G.edges[u, v]['weight'] * (solver.Sum(
            x_k[k, node_to_idx[u], node_to_idx[v]] + x_k[k, node_to_idx[v], node_to_idx[u]] for k in
            range(num_demands)))
        for u, v in G.edges()
    )
    solver.Minimize(total_wire_length)

    # --- 步骤 E: 求解模型 (无变化) ---
    print("  d) 开始求解 MILP 模型...")
    solver.EnableOutput()

    solver.SetTimeLimit(time_limit_seconds * 1000)
    status = solver.Solve()

    # --- 步骤 F: 结果解析 (无变化) ---
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        print("求解失败，未找到解决方案。")
        return [[] for _ in demands]

    print(f"  e) 求解成功！所有电缆总长度: {solver.Objective().Value():.2f}")

    # (结果解析部分与之前完全相同)
    H = nx.Graph()
    for u, v in G.edges():
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_key = tuple(sorted((u_idx, v_idx)))
        if x[edge_key].solution_value() > 0.5:
            H.add_edge(u, v, **G.edges[u, v])

    if H.number_of_nodes() > 0:
        num_nodes = H.number_of_nodes()
        num_edges = H.number_of_edges()
        print(f"\n  [MILP Solution Graph Stats]")
        print(f"  - Nodes in the solution graph: {num_nodes}")
        print(f"  - Edges in the solution graph: {num_edges}")
        # 这个函数不强制树结构，所以不进行 |E|=|V|-1 的验证
        print("-" * 25)

    optimized_paths = []
    for start_demand, end_demand in demands:
        snapped_start = terminal_map_to_snapped[start_demand]
        snapped_end = terminal_map_to_snapped[end_demand]
        try:
            path_nodes = nx.shortest_path(H, source=snapped_start, target=snapped_end, weight='weight')
            full_geometric_path = []
            if start_demand != snapped_start:
                path_to_snap = router.find_path_single_cable(start_demand, snapped_start)
                if path_to_snap:
                    full_geometric_path.extend(path_to_snap[:-1])
                else:
                    full_geometric_path.append(start_demand)
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                segment = H.get_edge_data(u, v)['path']
                if segment[0] != u: segment = segment[::-1]
                if full_geometric_path:
                    full_geometric_path.extend(segment[1:])
                else:
                    full_geometric_path.extend(segment)
            if end_demand != snapped_end:
                path_from_snap = router.find_path_single_cable(snapped_end, end_demand)
                if path_from_snap:
                    if full_geometric_path:
                        full_geometric_path.extend(path_from_snap[1:])
                    else:
                        full_geometric_path.extend(path_from_snap)
                elif not full_geometric_path:
                    full_geometric_path.append(end_demand)
            optimized_paths.append(full_geometric_path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"  警告: 无法在线束子图上为需求 {start_demand} -> {end_demand} 找到路径。")
            optimized_paths.append([])

    print(f"MILP 优化及路径重建总耗时: {time.time() - start_time:.2f} 秒。")
    return optimized_paths


def refine_harness_geometry(
        router: 'GridCableRouter',
        optimized_paths: List[List[Tuple[float, float]]],
        original_demands: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        turn_penalty: float = 0.5,
        shared_bonus: float = 0.15
) -> List[List[Tuple[float, float]]]:
    """
    一个先进的几何精细化函数，用于后处理优化结果。

    Args:
        router: GridCableRouter 实例。
        optimized_paths: 来自上一阶段（如MILP）的优化路径列表。
        original_demands: 原始电缆需求列表。
        turn_penalty: 用于平滑路径的转向惩罚值。
        shared_bonus: 在协同布线中，给予共享路段的成本折扣，鼓励捆绑。

    Returns:
        经过几何精细化后的最终电缆路径列表。
    """
    print("\n=== [4] 高级几何路径精细化重构 ===")
    start_time = time.time()

    if not any(optimized_paths):
        print("  输入路径为空，跳过精细化。")
        return optimized_paths

    # --- 步骤 1: 从优化结果中提取无环拓扑和路段使用频率 ---
    print("  a) 提取当前线束拓扑并计算路段使用频率...")
    routes_for_extraction = {i: path for i, path in enumerate(optimized_paths) if path}
    if not routes_for_extraction: return optimized_paths

    extractor = TopologyExtractor(router, routes_for_extraction)
    keypoints, path_segments = extractor.build_simplified_graph()

    # 创建一个临时的图来计算每段被共享的次数
    G_temp = nx.Graph()
    for (u, v), segment_info in path_segments.items():
        # 我们只关心拓扑连接，成本暂时不重要
        G_temp.add_edge(u, v, usage_count=0, length=extractor.calculate_segment_cost(segment_info['path']))

    # 遍历每条电缆路径，在其经过的拓扑段上增加使用计数
    for path in optimized_paths:
        if not path: continue
        # 找出路径上的关键点
        path_keypoints_indices = [i for i, p in enumerate(path) if p in keypoints]
        if len(path_keypoints_indices) < 2:
            # 如果路径不经过任何内部关键点，则只考虑其端点
            path_keypoints_indices = sorted(list({0, len(path) - 1}))

        for i in range(len(path_keypoints_indices) - 1):
            start_idx, end_idx = path_keypoints_indices[i], path_keypoints_indices[i + 1]
            if start_idx == end_idx: continue

            u, v = path[start_idx], path[end_idx]
            edge_key = tuple(sorted((u, v)))

            if G_temp.has_edge(edge_key[0], edge_key[1]):
                G_temp.edges[edge_key]['usage_count'] += 1
            else:
                # 理论上不应发生，但作为健壮性检查
                print(f"  [警告] 路径段 {edge_key} 在拓扑图中未找到。")

    # --- 步骤 2: 优先重路由 (协同布线) ---
    print("  b) 按重要性排序并进行协同重路由...")

    # 计算每个路段的“影响因子”：使用次数 * 长度
    segments_to_reroute = []
    for u, v, data in G_temp.edges(data=True):
        impact_score = data['usage_count'] * data['length']
        segments_to_reroute.append(((u, v), impact_score))

    # 按影响因子从高到低排序
    segments_to_reroute.sort(key=lambda x: x[1], reverse=True)

    # `global_shared_grid_edges` 用于记录所有已布线路径经过的栅格路段
    # 这使得后布线的路径可以“看到”先布线的路径，并选择与之捆绑
    global_shared_grid_edges = {}

    # G_refined 将存储我们最终的、几何优化的线束网络
    G_refined = nx.Graph()

    for (u, v), _ in segments_to_reroute:
        # 使用带状态的A*算法寻找平滑路径，并利用共享路段信息
        new_path = extractor._find_smooth_path_between_keypoints(
            u, v,
            turn_penalty=turn_penalty,
            shared_grid_edges=global_shared_grid_edges,
            shared_bonus=shared_bonus
        )

        if new_path:
            cost = extractor.calculate_segment_cost(new_path)
            G_refined.add_edge(u, v, weight=cost, path=new_path)

            # 更新全局共享栅格路段字典
            path_grid = [router.physical_to_grid_coords(*p) for p in new_path]
            for i in range(len(path_grid) - 1):
                grid_edge = tuple(sorted((path_grid[i], path_grid[i + 1])))
                global_shared_grid_edges[grid_edge] = global_shared_grid_edges.get(grid_edge, 0) + 1
        else:
            # 如果寻路失败，保留原始路径段以保证连通性
            print(f"  [警告] 协同重路由失败: {u} -> {v}。保留原始路径。")
            original_segment = path_segments.get(tuple(sorted((u, v))))
            if original_segment:
                G_refined.add_edge(u, v, weight=original_segment['cost'], path=original_segment['path'])

    # --- 步骤 3: 在精细化的网络上重建所有电缆路径 ---
    print("  c) 在精细化后的无环网络上重建最终电缆路径...")

    # 扩展图，确保所有原始需求端点都能连接到图中
    all_graph_nodes = set(G_refined.nodes())
    for start_demand, end_demand in original_demands:
        for terminal in [start_demand, end_demand]:
            if terminal not in all_graph_nodes:
                # 找到拓扑图中最接近的节点
                nearest_node = min(all_graph_nodes, key=lambda n: math.hypot(n[0] - terminal[0], n[1] - terminal[1]))
                # 规划连接路径
                path_to_graph = router.find_path_single_cable(terminal, nearest_node, turn_penalty=turn_penalty)
                if path_to_graph:
                    cost = extractor.calculate_segment_cost(path_to_graph)
                    G_refined.add_edge(terminal, nearest_node, weight=cost, path=path_to_graph)
                else:
                    # 如果失败，直接添加一个直线连接作为备用
                    cost = math.hypot(terminal[0] - nearest_node[0], terminal[1] - nearest_node[1])
                    G_refined.add_edge(terminal, nearest_node, weight=cost, path=[terminal, nearest_node])

    final_paths = []
    for start_demand, end_demand in original_demands:
        try:
            # 在最终的、保证无环的图上寻找最短路径
            path_nodes = nx.shortest_path(G_refined, source=start_demand, target=end_demand, weight='weight')

            # 拼接几何路径
            full_geometric_path = [path_nodes[0]]
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                segment = G_refined.edges[u, v]['path']
                # 确保路径方向正确
                if segment[0] != u:
                    segment = segment[::-1]
                full_geometric_path.extend(segment[1:])
            final_paths.append(full_geometric_path)

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"  [错误] 最终路径重建失败: {start_demand} -> {end_demand}。回退到简单A*。")
            fallback_path = router.find_path_single_cable(start_demand, end_demand, turn_penalty=turn_penalty)
            final_paths.append(fallback_path or [start_demand, end_demand])

    print(f"  高级几何精细化完成。耗时: {time.time() - start_time:.2f} 秒。")
    return final_paths


def refine_topology_with_bundling(
        router: 'GridCableRouter',
        optimized_paths: List[List[Tuple[float, float]]],
        turn_penalty: float = 0.5,
        corridor_bonus: float = 0.5,
        corridor_width: int = 10,
        shared_bonus: float = 0.5
) -> List[List[Tuple[float, float]]]:
    """
    一个混合几何精细化函数，严格保持拓扑结构的同时鼓励路径捆绑。

    它结合了两种策略：
    1.  使用引导走廊和禁行区来保证每个拓扑段的重路由不会产生捷径。
    2.  按照线束主干（被共享次数多的路段）优先的顺序进行重路由，并使用
        `shared_bonus` 鼓励后布线的路径与先布好的主干路径对齐。

    Args:
        router: GridCableRouter 实例。
        optimized_paths: 保证为树形结构的路径列表。
        turn_penalty: 用于平滑路径的转向惩罚值。
        corridor_bonus: 为引导走廊内的栅格提供的成本折扣。
        corridor_width: 引导走廊矩形从其基本边界向外扩展的栅格数。
        shared_bonus: 为已布线的共享栅格路段提供的额外成本折扣，鼓励捆绑。

    Returns:
        经过几何精细化，拓扑结构不变且路径更紧凑的最终电缆路径列表。
    """
    print("\n=== [4 Hybrid] 拓扑保持与路径捆绑的几何精细化 ===")
    start_time = time.time()

    if not any(optimized_paths):
        return optimized_paths

    # --- 步骤 1: 提取拓扑骨架并计算路段影响因子 ---
    print("  a) 提取拓扑骨架并计算路段影响因子...")
    routes_for_extraction = {i: path for i, path in enumerate(optimized_paths) if path}
    if not routes_for_extraction: return optimized_paths

    extractor = TopologyExtractor(router, routes_for_extraction)
    keypoints, path_segments = extractor.build_simplified_graph()

    # 计算每个拓扑段被多少条电缆共享
    segment_usage = {edge: 0 for edge in path_segments.keys()}
    for path in optimized_paths:
        if not path: continue
        # 优化关键点提取，确保路径端点也被视为关键点进行分段
        path_kps_indices = sorted(list({0, len(path) - 1}.union(
            {i for i, p in enumerate(path) if p in keypoints}
        )))

        for i in range(len(path_kps_indices) - 1):
            u = path[path_kps_indices[i]]
            v = path[path_kps_indices[i + 1]]
            if u == v: continue
            edge = tuple(sorted((u, v)))
            if edge in segment_usage:
                segment_usage[edge] += 1

    # 按“影响因子” (使用次数 * 原始路径长度) 从高到低排序。
    segments_to_reroute = sorted(
        path_segments.keys(),
        key=lambda edge: segment_usage.get(edge, 0) * path_segments[edge]['cost'],
        reverse=True
    )
    print(f"  b) 将对 {len(segments_to_reroute)} 个拓扑段按“影响因子”进行协同重路由...")

    # --- 步骤 2: 按顺序进行有引导、有记忆的重路由 ---
    refined_segments = {}
    original_cost_grid = router.cost_grid.copy()
    # `shared_grid_mask` 用于记录已布线路径的位置，值为折扣
    shared_grid_mask = np.ones_like(original_cost_grid)

    for edge in segments_to_reroute:
        kp1, kp2 = edge
        segment_info = path_segments[edge]

        # --- 创建引导走廊与共享折扣相结合的临时成本地图 ---
        temp_cost_grid = original_cost_grid.copy()

        # 1. 定义主矩形区域 (同 refine_geometry_rect_topology)
        g_kp1_x, g_kp1_y = router.physical_to_grid_coords(*kp1)
        g_kp2_x, g_kp2_y = router.physical_to_grid_coords(*kp2)

        # 基础矩形（不含扩展）
        base_min_gx, base_max_gx = min(g_kp1_x, g_kp2_x), max(g_kp1_x, g_kp2_x)
        base_min_gy, base_max_gy = min(g_kp1_y, g_kp2_y), max(g_kp1_y, g_kp2_y)

        # 扩展后的走廊
        corridor_min_gx = max(0, base_min_gx - corridor_width)
        corridor_max_gx = min(router.grid_cols - 1, base_max_gx + corridor_width)
        corridor_min_gy = max(0, base_min_gy - corridor_width)
        corridor_max_gy = min(router.grid_rows - 1, base_max_gy + corridor_width)

        # 2. **混合折扣**: 在走廊内应用 `corridor_bonus` 和 `shared_bonus`
        for nx in range(corridor_min_gx, corridor_max_gx + 1):
            for ny in range(corridor_min_gy, corridor_max_gy + 1):
                row, col = router.grid_coords_to_array_index(nx, ny)
                if not np.isinf(temp_cost_grid[row, col]):
                    # 首先应用走廊折扣，然后应用已存在路径的共享折扣
                    temp_cost_grid[row, col] *= corridor_bonus * shared_grid_mask[row, col]

        # 3. “挖掉”禁行区 (同 refine_geometry_rect_topology)
        other_keypoints = keypoints - {kp1, kp2}
        corners = [(base_min_gx, base_min_gy), (base_min_gx, base_max_gy),
                   (base_max_gx, base_min_gy), (base_max_gx, base_max_gy)]

        for other_kp in other_keypoints:
            g_other_x, g_other_y = router.physical_to_grid_coords(*other_kp)
            if base_min_gx <= g_other_x <= base_max_gx and base_min_gy <= g_other_y <= base_max_gy:
                closest_corner = min(corners, key=lambda c: abs(c[0] - g_other_x) + abs(c[1] - g_other_y))
                sub_min_gx, sub_max_gx = min(g_other_x, closest_corner[0]), max(g_other_x, closest_corner[0])
                sub_min_gy, sub_max_gy = min(g_other_y, closest_corner[1]), max(g_other_y, closest_corner[1])
                for nx in range(sub_min_gx, sub_max_gx + 1):
                    for ny in range(sub_min_gy, sub_max_gy + 1):
                        row, col = router.grid_coords_to_array_index(nx, ny)
                        temp_cost_grid[row, col] = original_cost_grid[row, col]
                        # temp_cost_grid[row, col] = np.inf

        # --- 在引导地图上进行A*寻路 ---
        router.cost_grid = temp_cost_grid
        new_path = router.find_path_single_cable_box(kp1, kp2, turn_penalty=turn_penalty)
        router.cost_grid = original_cost_grid

        if new_path:
            refined_segments[edge] = {'path': new_path, 'cost': extractor.calculate_segment_cost(new_path)}
            # **关键**: 更新共享地图，为下一次布线提供“记忆”
            for p_phys in new_path:
                g_x, g_y = router.physical_to_grid_coords(*p_phys)
                row, col = router.grid_coords_to_array_index(g_x, g_y)
                shared_grid_mask[row, col] = 1 - shared_bonus
        else:
            print(f"    [警告] 协同重路由失败: {kp1} -> {kp2}。保留原始几何路径。")
            refined_segments[edge] = segment_info

    # --- 步骤 3: 使用新的几何路径段重建所有电缆 ---
    print("  c) 使用精细化后的几何路径段重建所有电缆...")
    final_paths = []
    for path in optimized_paths:
        if not path:
            final_paths.append([])
            continue
        path_keypoints_indices = [i for i, p in enumerate(path) if p in keypoints]
        if 0 not in path_keypoints_indices: path_keypoints_indices.insert(0, 0)
        if (len(path) - 1) not in path_keypoints_indices: path_keypoints_indices.append(len(path) - 1)
        path_keypoints_indices = sorted(list(set(path_keypoints_indices)))
        reconstructed_path = []
        for i in range(len(path_keypoints_indices) - 1):
            start_idx, end_idx = path_keypoints_indices[i], path_keypoints_indices[i + 1]
            u, v = path[start_idx], path[end_idx]
            edge_key = tuple(sorted((u, v)))
            if edge_key in refined_segments:
                segment_geom = refined_segments[edge_key]['path']
                if segment_geom[0] != u:
                    segment_geom = segment_geom[::-1]
                if not reconstructed_path:
                    reconstructed_path.extend(segment_geom)
                else:
                    reconstructed_path.extend(segment_geom[1:])
            else:
                print(f"    [警告] 未找到拓扑段 {edge_key} 的几何路径，使用原始路径。")
                reconstructed_path.extend(
                    path[start_idx: end_idx + 1] if not reconstructed_path else path[start_idx + 1: end_idx + 1])
        final_paths.append(reconstructed_path)

    print(f"  混合精细化完成。耗时: {time.time() - start_time:.2f} 秒。")
    return final_paths


def refine_topology_strict_topology(
        router: 'GridCableRouter',
        optimized_paths: List[List[Tuple[float, float]]],
        components_data: List[Dict],
        offsets: Tuple[float, float],
        global_buffer: float = 5.0,
        turn_penalty: float = 0.5,
        corridor_bonus: float = 0.5,
        corridor_width: int = 10,
        exclusion_padding: int = 1,
        visualize_steps: bool = False,
        output_prefix: str = "step_vis",
        enable_bspline: bool = True,
        bspline_smoothness: float = 5000.0,
        bspline_degree: int = 3,
        bspline_sample_density: int = 5,
        min_turning_radius: float = 0.0  # [新增]
) -> List[List[Tuple[float, float]]]:
    print("\n=== [4 Strict Topology Refinement] 严格拓扑保持 (物理增强版 + B-Spline) ===")
    start_time = time.time()

    if not any(optimized_paths):
        return optimized_paths

    # --- 辅助：预计算接插件的物理空隙 ---
    def _get_gap_blockages(router, components, off_x, off_y, buffer_dist):
        blockages = []
        for comp in components:
            cx, cy = comp['x'] + off_x, comp['y'] + off_y
            cw, cl = comp['w'], comp['l']
            direction = comp.get('connector')
            if not direction: continue
            overlap = 2.0
            rect = None
            if direction == '右':
                edge_x = cx + cw / 2
                rect = (edge_x - overlap, cy - cl / 2, buffer_dist + overlap, cl)
            elif direction == '左':
                edge_x = cx - cw / 2
                rect = (edge_x - buffer_dist, cy - cl / 2, buffer_dist + overlap, cl)
            elif direction == '上':
                edge_y = cy + cl / 2
                rect = (cx - cw / 2, edge_y - overlap, cw, buffer_dist + overlap)
            elif direction == '下':
                edge_y = cy - cl / 2
                rect = (cx - cw / 2, edge_y - buffer_dist, cw, buffer_dist + overlap)
            if rect:
                px, py, pw, ph = rect
                gx1, gy1 = router.physical_to_grid_coords(px, py)
                gx2, gy2 = router.physical_to_grid_coords(px + pw, py + ph)
                min_gx, max_gx = min(gx1, gx2), max(gx1, gx2)
                min_gy, max_gy = min(gy1, gy2), max(gy1, gy2)
                blockages.append((min_gx, max_gx, min_gy, max_gy))
        return blockages

    connector_gap_grids = _get_gap_blockages(router, components_data, offsets[0], offsets[1], global_buffer)
    print(f"  -> 已识别并封堵 {len(connector_gap_grids)} 个接插件物理空隙。")

    def snap_point(p):
        return (round(p[0], 4), round(p[1], 4))

    cleaned_paths = []
    for path in optimized_paths:
        if not path:
            cleaned_paths.append([])
            continue
        cleaned_paths.append([snap_point(p) for p in path])

    print("  a) 提取拓扑骨架并计算路段影响因子...")

    routes_for_extraction = {i: path for i, path in enumerate(cleaned_paths) if path}
    if not routes_for_extraction: return optimized_paths
    extractor = TopologyExtractor(router, routes_for_extraction)
    keypoints, path_segments = extractor.build_simplified_graph()

    segment_usage = {edge: 0 for edge in path_segments.keys()}
    for path in cleaned_paths:
        if not path: continue
        path_kps_indices = sorted(list({0, len(path) - 1}.union(
            {i for i, p in enumerate(path) if p in keypoints}
        )))
        for i in range(len(path_kps_indices) - 1):
            u, v = path[path_kps_indices[i]], path[path_kps_indices[i + 1]]
            if u == v: continue
            edge = tuple(sorted((u, v)))
            if edge in segment_usage:
                segment_usage[edge] += 1

    segments_to_reroute = sorted(
        path_segments.keys(),
        key=lambda edge: segment_usage.get(edge, 0) * path_segments[edge]['cost'],
        reverse=True
    )
    print(f"  b) 将对 {len(segments_to_reroute)} 个拓扑段按“影响因子”进行协同重路由...")

    refined_segments = {}
    original_cost_grid = router.cost_grid.copy()
    stats = {"success": 0, "failed": 0}

    if visualize_steps:
        import os
        vis_dir = f"{output_prefix}_strict_refinement"
        os.makedirs(vis_dir, exist_ok=True)

    for i, edge in enumerate(segments_to_reroute):
        kp1, kp2 = edge
        segment_info = path_segments[edge]

        g_kp1_x, g_kp1_y = router.physical_to_grid_coords(*kp1)
        g_kp2_x, g_kp2_y = router.physical_to_grid_coords(*kp2)

        base_min_gx, base_max_gx = min(g_kp1_x, g_kp2_x), max(g_kp1_x, g_kp2_x)
        base_min_gy, base_max_gy = min(g_kp1_y, g_kp2_y), max(g_kp1_y, g_kp2_y)

        corridor_min_gx = max(0, base_min_gx - corridor_width)
        corridor_max_gx = min(router.grid_cols - 1, base_max_gx + corridor_width)
        corridor_min_gy = max(0, base_min_gy - corridor_width)
        corridor_max_gy = min(router.grid_rows - 1, base_max_gy + corridor_width)

        temp_grid = original_cost_grid.copy()

        # A) 应用接插件空隙封堵
        for (bg_min_x, bg_max_x, bg_min_y, bg_max_y) in connector_gap_grids:
            if (bg_max_x < corridor_min_gx or bg_min_x > corridor_max_gx or
                    bg_max_y < corridor_min_gy or bg_min_y > corridor_max_gy):
                continue
            r_min = max(0, bg_min_y)
            r_max = min(router.grid_rows - 1, bg_max_y)
            c_min = max(0, bg_min_x)
            c_max = min(router.grid_cols - 1, bg_max_x)
            for gx in range(c_min, c_max + 1):
                for gy in range(r_min, r_max + 1):
                    r, c = router.grid_coords_to_array_index(gx, gy)
                    temp_grid[r, c] = np.inf

        # B) 动态障碍物识别
        current_obstacle_paths_for_vis = []
        for other_edge, other_info in path_segments.items():
            if other_edge == edge: continue
            if other_edge in refined_segments:
                obstacle_path = refined_segments[other_edge]['path']
            else:
                obstacle_path = other_info['path']

            if visualize_steps:
                current_obstacle_paths_for_vis.append(obstacle_path)

            obs_gx = [router.physical_to_grid_coords(p[0], p[1])[0] for p in obstacle_path]
            if not obs_gx: continue

            min_ox, max_ox = min(obs_gx), max(obs_gx)
            obs_gy = [router.physical_to_grid_coords(p[0], p[1])[1] for p in obstacle_path]
            min_oy, max_oy = min(obs_gy), max(obs_gy)

            if (max_ox < corridor_min_gx - exclusion_padding or min_ox > corridor_max_gx + exclusion_padding or
                    max_oy < corridor_min_gy - exclusion_padding or min_oy > corridor_max_gy + exclusion_padding):
                continue

            for p_phys in obstacle_path:
                gx, gy = router.physical_to_grid_coords(*p_phys)
                if (corridor_min_gx - exclusion_padding <= gx <= corridor_max_gx + exclusion_padding) and \
                        (corridor_min_gy - exclusion_padding <= gy <= corridor_max_gy + exclusion_padding):
                    p_min_x = max(0, gx - exclusion_padding)
                    p_max_x = min(router.grid_cols - 1, gx + exclusion_padding)
                    p_min_y = max(0, gy - exclusion_padding)
                    p_max_y = min(router.grid_rows - 1, gy + exclusion_padding)
                    for ox in range(p_min_x, p_max_x + 1):
                        for oy in range(p_min_y, p_max_y + 1):
                            r, c = router.grid_coords_to_array_index(ox, oy)
                            temp_grid[r, c] = np.inf

        # C) 走廊奖励
        for nx in range(corridor_min_gx, corridor_max_gx + 1):
            for ny in range(corridor_min_gy, corridor_max_gy + 1):
                row, col = router.grid_coords_to_array_index(nx, ny)
                if not np.isinf(temp_grid[row, col]):
                    temp_grid[row, col] *= corridor_bonus

        # D) 端点保护
        safe_radius = 3
        for kp in [kp1, kp2]:
            kpx, kpy = router.physical_to_grid_coords(*kp)
            for ox in range(max(0, kpx - safe_radius), min(router.grid_cols - 1, kpx + safe_radius + 1)):
                for oy in range(max(0, kpy - safe_radius), min(router.grid_rows - 1, kpy + safe_radius + 1)):
                    r, c = router.grid_coords_to_array_index(ox, oy)
                    if not np.isinf(original_cost_grid[r, c]):
                        temp_grid[r, c] = original_cost_grid[r, c] * corridor_bonus

        # --- A* Search ---
        router.cost_grid = temp_grid
        new_path = router.find_path_single_cable_box(kp1, kp2, turn_penalty=turn_penalty)

        # --- [NEW] B-Spline Smoothing Integration ---
        if new_path and enable_bspline:
            smoothed_path = BSplineSmoother.smooth_path(
                new_path,
                router,
                smoothness=bspline_smoothness,
                degree=bspline_degree,
                sample_density=bspline_sample_density,
                min_turning_radius=min_turning_radius  # [传递]
            )
            if smoothed_path and len(smoothed_path) > 2:
                new_path = smoothed_path

        router.cost_grid = original_cost_grid

        # --- 结果处理 ---
        if new_path:
            # 即使 smooth_path 强制了端点，这里再 snap 一次保证小数位一致
            snapped_new_path = [snap_point(p) for p in new_path]
            refined_segments[edge] = {'path': snapped_new_path}
            stats["success"] += 1
        else:
            refined_segments[edge] = segment_info
            stats["failed"] += 1

        # --- Visualization (Optional) ---
        if visualize_steps:
            status_str = "Success" if new_path else "FAILED"
            vis_title = (f"Step {i + 1}/{len(segments_to_reroute)}: {status_str}\n"
                         f"({int(kp1[0])},{int(kp1[1])}) -> ({int(kp2[0])},{int(kp2[1])})")
            save_path = f"{vis_dir}/step_{i + 1:03d}_{status_str}.png"
            visualize_segment_refinement_step(
                router, vis_title, save_path, kp1, kp2,
                original_path=segment_info['path'],
                new_path=new_path,
                corridor_coords=(corridor_min_gx, corridor_min_gy,
                                 corridor_max_gx - corridor_min_gx,
                                 corridor_max_gy - corridor_min_gy),
                obstacle_paths=current_obstacle_paths_for_vis,
                padding=exclusion_padding
            )

    print(f"  重构统计: 总计 {len(segments_to_reroute)} | 成功: {stats['success']} | 失败(保留原样): {stats['failed']}")

    # --- Step 3: Reconstruction (Robust Direction Check) ---
    print("  c) 重建路径 (使用距离判断方向)...")
    final_paths = []

    for path in cleaned_paths:
        if not path:
            final_paths.append([])
            continue

        path_kps_indices = [i for i, p in enumerate(path) if p in keypoints]
        if 0 not in path_kps_indices: path_kps_indices.insert(0, 0)
        if (len(path) - 1) not in path_kps_indices: path_kps_indices.append(len(path) - 1)
        path_kps_indices = sorted(list(set(path_kps_indices)))

        reconstructed_path = []

        for i in range(len(path_kps_indices) - 1):
            idx_a, idx_b = path_kps_indices[i], path_kps_indices[i + 1]
            u, v = path[idx_a], path[idx_b]  # u: 理论起点, v: 理论终点
            edge_key = tuple(sorted((u, v)))

            segment_geom = []
            if edge_key in refined_segments:
                segment_geom = refined_segments[edge_key]['path']
            else:
                segment_geom = path[idx_a: idx_b + 1]

            if not segment_geom: continue

            # --- [鲁棒性修复] 使用距离判断方向 ---
            # 判断 segment_geom 的起点离 u 近，还是终点离 u 近
            # 之前的 logic 是 if segment[0] != u，但浮点数误差会导致不相等
            d_start = (segment_geom[0][0] - u[0]) ** 2 + (segment_geom[0][1] - u[1]) ** 2
            d_end = (segment_geom[-1][0] - u[0]) ** 2 + (segment_geom[-1][1] - u[1]) ** 2

            # 如果尾部离理论起点更近，说明方向反了，翻转它
            if d_end < d_start:
                segment_geom = segment_geom[::-1]

            if not reconstructed_path:
                reconstructed_path.extend(segment_geom)
            else:
                # 拼接时避免重复添加连接点
                reconstructed_path.extend(segment_geom[1:])

        final_paths.append(reconstructed_path)

    print(f"  严格拓扑重构完成。耗时: {time.time() - start_time:.2f} 秒。")
    return final_paths


def post_optimization_mst_refinement(
        router: 'GridCableRouter',
        optimized_paths: List[List[Tuple[float, float]]],
        original_demands: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> List[List[Tuple[float, float]]]:
    """
    在全局优化后执行一个额外的精细化步骤。

    此函数接收一组已优化的电缆路径，从中提取一个新的拓扑骨架，然后
    利用基于最小生成树（MST）的方法在该新骨架上重新规划所有路径，
    旨在寻找一个可能更高效的最终线束拓扑结构。

    Args:
        router: GridCableRouter 实例。
        optimized_paths: 来自前一优化阶段的电缆路径列表。
        original_demands: 原始的电缆起点和终点需求列表。

    Returns:
        经过MST精细化后得到的新电缆路径列表。
    """
    print("\n=== [5] 后处理: 再次进行骨架提取与MST优化 ===")
    start_time = time.time()

    if not any(optimized_paths):
        print("  输入路径为空，跳过后处理。")
        return optimized_paths

    # 1. 将路径列表转换为 TopologyExtractor 所需的字典格式
    routes_for_extraction = {i: path for i, path in enumerate(optimized_paths) if path}

    # 2. 从优化后的路径中提取新的拓扑骨架
    print("  a) 从当前路径中提取新的拓扑骨架...")
    extractor = TopologyExtractor(router, routes_for_extraction)
    keypoints, path_segments = extractor.build_simplified_graph()

    # 3. 在新提取的骨架上运行基于MST的布线
    print("  b) 在新骨架上运行MST(SPT)优化...")
    final_paths = spt_based_routing(
        keypoints,
        path_segments,
        original_demands,
        router
    )

    print(f"  MST(SPT) 后处理完成。耗时: {time.time() - start_time:.2f} 秒。")
    return final_paths

# ==============================================================================
# 可视化函数
# ==============================================================================
def visualize_final_result(router, original_routes, optimized_paths, components_for_drawing, offset_x, offset_y,
                           save_path=None, title=None):
    """
    修改后的最终结果可视化函数。
    1. 所有线束路径段统一为红色。
    2. 路径段的粗细与被电缆使用的次数成正比。
    """
    fig, ax = plt.subplots(figsize=(30, 15))
    draw_background(ax, router)

    # 绘制组件布局 (无变化)
    for comp in components_for_drawing:
        l, w, x, y = comp['l'], comp['w'], comp['x'], comp['y']
        ax.add_patch(Rectangle((x - w / 2 + offset_x, y - l / 2 + offset_y), w, l,
                               edgecolor='blue', facecolor='lightblue', alpha=0.8, zorder=3))

    # 绘制原始 HRH/Graph 路由作为淡淡的背景参考 (无变化)
    colors = plt.cm.tab10.colors
    if isinstance(original_routes, dict):
        routes_to_draw = original_routes.values()
    else:
        routes_to_draw = original_routes

    for i, route in enumerate(routes_to_draw):
        if route:
            xs, ys = zip(*route)
            ax.plot(xs, ys, color=colors[i % len(colors)], lw=1.5, alpha=0.2, linestyle='--')

    # 绘制最终优化后的线束
    # 步骤 1: 计算每个独立线段被共享的次数 (无变化)
    shared_edges = defaultdict(int)
    for path in optimized_paths:
        if path:
            for k in range(len(path) - 1):
                # 使用排序后的元组作为键，确保 (p1, p2) 和 (p2, p1) 是同一个线段
                shared_edges[tuple(sorted((path[k], path[k + 1])))] += 1

    if not shared_edges:
        print("警告: 优化后的路径为空，无法生成最终可视化结果。")
        return

    all_lines = list(shared_edges.keys())

    # 步骤 2: [修改] 根据使用次数计算线宽，次数越多越粗
    base_linewidth = 1.2  # 单根电缆通过时的基础宽度
    width_increment = 0.6  # 每多一根电缆通过时增加的宽度
    max_linewidth = 12.0  # 设置最大线宽上限

    line_widths = [
        min(max_linewidth, base_linewidth + (shared_edges[edge] - 1) * width_increment)
        for edge in all_lines
    ]

    # 步骤 3: 创建 LineCollection，所有线段颜色固定为红色
    # 注意 'colors' 参数现在直接是 'red'
    lc = LineCollection(all_lines, colors='red', linewidths=line_widths, zorder=5, alpha=0.9)
    ax.add_collection(lc)


    # 绘制起点和终点 (无变化)
    start_points = {p[0] for p in optimized_paths if p}
    end_points = {p[-1] for p in optimized_paths if p}
    if start_points: ax.scatter(*zip(*start_points), c='cyan', marker='o', s=80, zorder=10, edgecolor='k',
                                label='Start Points')
    if end_points: ax.scatter(*zip(*end_points), c='magenta', marker='s', s=80, zorder=10, edgecolor='k',
                              label='End Points')

    # 设置图像属性
    ax.set_xlim(0, router.physical_width)
    ax.set_ylim(0, router.physical_height)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"最终可视化结果已保存至: {save_path}")
    else:
        plt.show()

def expand_obstacles_with_safety_distance(components):
    points = set()
    for comp in components:
        x, y, l, w, s = comp['x'], comp['y'], comp['l'], comp['w'], comp.get('safety_distance', 0)
        ex, ey = x - w / 2 - s, y - l / 2 - s
        for px in range(int(np.floor(ex)), int(np.ceil(ex + w + 2 * s))):
            for py in range(int(np.floor(ey)), int(np.ceil(ey + l + 2 * s))):
                points.add((px, py))
    return list(points)


def calculate_unique_edge_length(optimized_paths):
    unique_edges = set()
    for path in optimized_paths:
        if not path: continue
        for j in range(len(path) - 1):
            unique_edges.add(tuple(sorted((path[j], path[j + 1]))))
    return sum(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in unique_edges)


def visualize_paths_on_skeleton(
        router: 'GridCableRouter',
        keypoints: Set[Tuple[float, float]],
        path_segments: Dict,
        routes: Dict[int, List[Tuple[float, float]]],
        title: str,
        save_path: Optional[str] = None
):
    """
    可视化在一个背景骨架上规划出的多条路径。

    Args:
        router: GridCableRouter 实例。
        keypoints: 骨架的关键点。
        path_segments: 骨架的所有路径段，将作为灰色背景绘制。
        routes: 要高亮显示的电缆路径字典。
        title: 图像标题。
        save_path: 图像保存路径。
    """
    print(f"  -> 正在可视化: {title}...")
    fig, ax = plt.subplots(figsize=(30, 15))
    draw_background(ax, router)

    # --- 步骤 1: 绘制背景骨架 (灰色、较细) ---
    if path_segments:
        skeleton_lines = [segment['path'] for segment in path_segments.values()]
        lc_skeleton = LineCollection(skeleton_lines, colors='gray', linewidths=1.0, alpha=0.5, zorder=3,
                                     label='完整骨架')
        ax.add_collection(lc_skeleton)

    # --- 步骤 2: 绘制规划出的路径 (红色、根据共享次数变粗) ---
    if routes:
        shared_edges = defaultdict(int)
        for path in routes.values():
            if path:
                for k in range(len(path) - 1):
                    shared_edges[tuple(sorted((path[k], path[k + 1])))] += 1

        if shared_edges:
            all_lines = list(shared_edges.keys())
            base_linewidth = 1.5
            width_increment = 0.7
            max_linewidth = 12.0
            line_widths = [
                min(max_linewidth, base_linewidth + (shared_edges[edge] - 1) * width_increment)
                for edge in all_lines
            ]
            lc_routes = LineCollection(all_lines, colors='red', linewidths=line_widths, zorder=5, alpha=0.9,
                                       label='规划路径')
            ax.add_collection(lc_routes)

    # --- 步骤 3: 绘制关键点和端点 (可选) ---
    start_points = {p[0] for p in routes.values() if p}
    end_points = {p[-1] for p in routes.values() if p}
    if start_points: ax.scatter(*zip(*start_points), c='cyan', marker='o', s=80, zorder=10, edgecolor='k', label='起点')
    if end_points: ax.scatter(*zip(*end_points), c='magenta', marker='s', s=80, zorder=10, edgecolor='k', label='终点')

    # --- 设置图像属性 ---
    ax.set_xlim(0, router.physical_width)
    ax.set_ylim(0, router.physical_height)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"     可视化结果已保存至: {save_path}")
    else:
        plt.show()

def visualize_segment_refinement_step(
        router: 'GridCableRouter',
        title: str,
        save_path: str,
        kp1: Tuple[float, float],
        kp2: Tuple[float, float],
        original_path: List[Tuple[float, float]],
        new_path: List[Tuple[float, float]],
        corridor_coords: Tuple[int, int, int, int],
        obstacle_paths: List[List[Tuple[float, float]]],  # [修改] 传入作为障碍的其他路径列表
        padding: int = 1
):
    """
    Visualizes a single step of the strict topology refinement process.
    展示当前正在优化的线段、它的搜索范围（走廊），以及阻挡它的其他线段（障碍）。
    """
    fig, ax = plt.subplots(figsize=(16, 10))  # 调整一下比例
    ax.set_title(title, fontsize=14)

    # 1. 绘制底图 (Background)
    # 临时把 grid 设回去画背景，或者直接画（取决于 draw_background 实现）
    # 这里假设 draw_background 只读不写
    draw_background(ax, router)

    # 2. 绘制障碍路径 (Obstacles / Context) - [核心修改]
    # 在严格拓扑模式下，这些路径周围是禁区 (INF)
    if obstacle_paths:
        # 画两层：一层宽的半透明红色代表“避让区域(Padding)”，一层细深红代表“路径中心”
        # 估算 linewidth: 1个网格约等于 screen 上的若干 point，这里简单给个大概值
        # 更好的做法是画成 PathPatch，但 LineCollection 性能更好且够用

        # 宽线 (模拟 Padding 区域)
        lc_obs_zone = LineCollection(obstacle_paths, colors='red',
                                     linewidths=(padding * 4 + 2),  # 夸张一点显示的宽度
                                     alpha=0.2, zorder=3, label='拓扑障碍区域')
        ax.add_collection(lc_obs_zone)

        # 细线 (路径骨架)
        lc_obs_core = LineCollection(obstacle_paths, colors='darkred',
                                     linewidths=1.5, alpha=0.6, zorder=3, label='障碍路径骨架')
        ax.add_collection(lc_obs_core)

    # 3. 绘制搜索走廊 (Corridor)
    g_min_x, g_min_y, g_w, g_h = corridor_coords
    p_x, p_y = router.grid_coords_to_physical(g_min_x, g_min_y)
    # 计算物理尺寸
    # 注意：grid坐标通常是左下角或中心，这里假设转换为物理坐标对应格子中心
    # 微调 rect 起点使其覆盖格子
    p_w = g_w * router.cell_width
    p_h = g_h * router.cell_height
    # 稍微外扩一点以便看清边界
    rect_x = p_x - 0.5 * router.cell_width
    rect_y = p_y - 0.5 * router.cell_height

    corridor_patch = Rectangle((rect_x, rect_y), p_w, p_h,
                               edgecolor='blue', facecolor='skyblue',
                               alpha=0.2, linestyle=':', linewidth=2, zorder=4,
                               label='搜索限制走廊')
    ax.add_patch(corridor_patch)

    # 4. 绘制当前处理的路径 (Current Segment)
    if original_path:
        ox, oy = zip(*original_path)
        ax.plot(ox, oy, color='orange', linestyle='--', linewidth=2.0, zorder=5, label='重构前路径 (Original)')

    if new_path:
        nx, ny = zip(*new_path)
        ax.plot(nx, ny, color='cyan', linestyle='-', linewidth=3.0, zorder=6, label='重构后路径 (Refined)')
    elif new_path is None:
        # 如果失败
        ax.text(0.5, 0.5, "Refinement Failed", transform=ax.transAxes, color='red', fontsize=20, ha='center')

    # 5. 绘制端点 (Keypoints)
    ax.scatter([kp1[0], kp2[0]], [kp1[1], kp2[1]],
               color='lime', marker='o', s=120, zorder=7, edgecolor='k', linewidth=1.5,
               label='当前段端点')

    # 设置视图范围
    ax.set_xlim(0, router.physical_width)
    ax.set_ylim(0, router.physical_height)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"     ...可视化已保存: {save_path}")

# ==============================================================================
# 1. 配置中心：将所有参数集中管理
# ==============================================================================
def get_default_config() -> Dict[str, Any]:
    """返回一个包含所有可调参数的默认配置字典。"""
    return {
        # --- 模式选择 ---
        "skeleton_mode": "graph",  # 'graph' (几何骨架) 或 'hrh' (迭代式)
        "initial_routing_method": "milp",  # 初始寻路方法 (仅 'graph' 模式下有效): 'milp' (精确) 或 'dijkstra' (快速)
        "optimize_mode": "milp",  # 全局优化方法: 'milp' (精确) 或 'mst' (启发式)

        "enable_diagonal_search": True,  # A*搜索中是否允许对角移动

        # --- 网格与空间离散化参数 ---
        "grid_scale": 4,   # 栅格缩放比例，值越大网格越粗糙，计算越快

        # --- 端点与组件的缓存距离 ---
        "buffer": 5,    # 电缆端点距离组件安全边界的额外距离

        # --- HRH 模式专用超参数 ---
        "hrh_shared_bonus": 0.5,
        "hrh_max_iterations": 5,

        # --- 流程开关控制 ---
        "enable_skeleton_optimization": True,  # 是否执行阶段2的拓扑骨架简化与重构
        "enable_geometric_refinement": True,  # 是否执行阶段4的几何路径精细化
        "enable_mst_post_refinement": True,   # 是否执行阶段5的MST最终拓扑精简

        # --- 拓扑优化超参数 ---
        "endpoint_merge_distance": 40.0,
        "internal_merge_distance": 50.0,
        "skeleton_rewiring_penalty": 0.2,

        # --- MILP优化超参数 ---
        "milp_time_limit_seconds": 300,  # MILP求解器的最大求解时间（秒）

        # --- 几何精细化阶段超参数 ---
        "refinement_turn_penalty": 50,  # 转向惩罚
        "refinement_shared_bonus": 0.5,  # 协同布线奖励
        "visualize_refinement_steps": True,
        "corridor_bonus": 0.5,   # 走廊折扣
        "corridor_width": 20,    # 引导区拓展宽度
        "exclusion_padding": 1,  # 禁止区拓展宽度

        # --- B-Spline 平滑配置 ---
        "enable_bspline": True,
        "bspline_smoothness": 500.0,
        "bspline_degree": 5,         # 3阶样条最稳定
        "bspline_sample_density": 20, #  采样密度，越高曲线越细腻
        "min_turning_radius": 0.0,  # 设置具体的毫米数值，例如 10mm
        # --- 可视化与输出 ---
        "save_visualizations": True,
        "output_prefix": "mid_case_testV3" # 输出文件的前缀
    }


# ==============================================================================
# 2. 数据准备模块：封装所有与数据加载和预处理相关的逻辑
# ==============================================================================
def setup_problem_instance(components_data: List[Dict],
                           cable_connections: List[Tuple[int, int]],
                           config: Dict[str, Any]) -> Dict[str, Any]:
    """根据原始组件数据、连接关系和指定的配置，准备路由所需的所有对象。"""
    print("--- 正在准备问题实例 ---")
    # 计算物理布局的边界和偏移量，以确保所有坐标为正
    all_x = [c['x'] for c in components_data]
    all_y = [c['y'] for c in components_data]
    min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
    offset_x, offset_y = -min_x + 100, -min_y + 100 # 增加100的边距
    physical_width, physical_height = int(max_x - min_x) + 200, int(max_y - min_y) + 200

    # 根据配置计算栅格尺寸
    GRID_SCALE = config.get("grid_scale", 1)
    GRID_SIZE = (physical_width // GRID_SCALE, physical_height // GRID_SCALE)

    Buffer = config.get("buffer", 5)
    # 定义辅助函数，用于计算电缆端点的精确物理坐标
    def get_route_endpoint(c, buffer=Buffer):
        x, y, l, w = c['x'], c['y'], c['l'], c['w']
        safety_dist = c.get('safety_distance', 0)
        # 总偏移量 = 组件安全距离 + 全局缓冲距离
        total_offset = safety_dist + buffer
        px = x - w / 2 - total_offset if c['connector'] == '左' else \
             x + w / 2 + total_offset if c['connector'] == '右' else x
        py = y + l / 2 + total_offset if c['connector'] == '上' else \
             y - l / 2 - total_offset if c['connector'] == '下' else y
        # 应用全局坐标偏移
        return (px + offset_x, py + offset_y)

    # 创建障碍物区域（考虑安全距离）
    raw_obstacles = expand_obstacles_with_safety_distance(components_data)
    obstacles_physical = [(x + offset_x, y + offset_y) for x, y in raw_obstacles]

    # 根据组件索引和连接关系，生成电缆需求的物理坐标列表
    cables = [(get_route_endpoint(components_data[s]), get_route_endpoint(components_data[e]))
              for s, e in cable_connections]

    allow_diagonal = config.get("enable_diagonal_search", False)
    print(f"  -> 对角移动搜索已 {'启用' if allow_diagonal else '禁用'}.")
    # 初始化核心布线器对象
    router = GridCableRouter((physical_width, physical_height), GRID_SIZE, obstacles_physical,
                             enable_diagonal=allow_diagonal)

    # 返回一个包含所有必要信息的字典
    return {
        "router": router,
        "cables": cables,
        "components_for_drawing": components_data,
        "offsets": (offset_x, offset_y)
    }

# ==============================================================================
# 3. 流程引擎：封装完整的、分阶段的优化流程
# ==============================================================================
class RoutingPipeline:
    def __init__(self, config: Dict[str, Any], problem_instance: Dict[str, Any]):
        self.config = config
        self.problem = problem_instance
        self.router = problem_instance["router"]
        self.cables = problem_instance["cables"]
        self.results = {}
        self.timings = {"start": time.perf_counter()}

    def _report_topology_stats(self, phase_title: str, paths: Any):
        """
        一个辅助函数，用于提取并报告给定路径集的拓扑统计信息。
        """
        print(f"\n{'='*10} {phase_title} 拓扑统计 {'='*10}")
        if isinstance(paths, list):
            routes_dict = {i: p for i, p in enumerate(paths) if p}
        elif isinstance(paths, dict):
            routes_dict = {k: v for k, v in paths.items() if v}
        else:
            print(f"  错误: 无法识别的路径格式 {type(paths)}。")
            print("=" * (len(phase_title) + 22))
            return
        if not routes_dict:
            print("  路径集为空，无法提取拓扑。")
            print("=" * (len(phase_title) + 22))
            return
        extractor = TopologyExtractor(self.router, routes_dict)
        keypoints, path_segments = extractor.build_simplified_graph()
        print(f"  >>> {phase_title} 摘要: {len(keypoints)} 个关键节点 (Nodes), {len(path_segments)} 个路径段 (Edges) <<<")
        print("=" * (len(phase_title) + 22))

    def run(self):
        """按顺序执行完整的优化流程。"""
        self._phase_1_initial_routing()
        self._phase_2_topology_extraction_and_refinement()
        self._phase_3_global_optimization()
        self.results["final_paths"] = self.results["optimized_paths"]
        if self.config["enable_geometric_refinement"]:
            self._phase_4_geometric_refinement()
            self.results["final_paths"] = self.results["refined_paths"]
        else:
            print("\n=== [4] 几何路径精细化重构 (已跳过) ===")
            self.results["refined_paths"] = self.results["optimized_paths"]
            self.timings["phase_4_done"] = self.timings["phase_3_done"]
        if self.config["enable_mst_post_refinement"]:
            self._phase_5_mst_post_refinement()
            self.results["final_paths"] = self.results["mst_refined_paths"]
        else:
            print("\n=== [5] 后处理: MST优化 (已跳过) ===")
            self.results["mst_refined_paths"] = self.results["refined_paths"]
            self.timings["phase_5_done"] = self.timings["phase_4_done"]
        self.timings["total"] = time.perf_counter() - self.timings["start"]
        self._calculate_final_stats()

    def _phase_1_initial_routing(self):
        """
        阶段 1: 初始路由。
        根据配置选择 'graph' (几何骨架) 或 'hrh' (迭代) 模式。
        在 'graph' 模式下，进一步根据配置选择 'milp' 或 'dijkstra' 作为初始寻路算法。
        """
        print(f"\n=== [1] 初始骨架生成与寻路 (模式: {self.config['skeleton_mode']}) ===")
        start_phase_time = time.perf_counter()

        if self.config["skeleton_mode"] == "graph":
            # --- 几何骨架模式 ---
            # 步骤 1A & 1B: 从布局的几何信息中生成一个完整的、包含所有潜在路径的骨架网络
            print("  [1A/1B] 正在从几何布局生成完整骨架...")
            keypoints, path_segments, raw_nodes, raw_edges = GraphTheory.generate_complete_skeleton(
                self.router, self.problem["components_for_drawing"], self.problem["offsets"]
            )
            # [关键] 保存完整骨架以备后续阶段使用，避免重复计算
            self.results["complete_keypoints"] = keypoints
            self.results["complete_segments"] = path_segments

            # 步骤 1C: 在生成的完整骨架上，为所有电缆需求规划初始路径
            routing_method = self.config.get("initial_routing_method", "milp")
            print(f"  [1C] 在完整骨架上执行初始寻路 (算法: {routing_method.upper()})")

            if routing_method == 'milp':
                milp_time_limit = self.config.get("milp_time_limit_seconds", 300)
                initial_routes = GraphTheory.milp_route_on_skeleton(
                    self.router, keypoints, path_segments, self.cables, milp_time_limit
                )
            elif routing_method == 'dijkstra':
                initial_routes = GraphTheory.route_cables_on_skeleton(
                    self.router, keypoints, path_segments, self.cables
                )
            else:
                raise ValueError(f"未知的初始寻路方法: {routing_method}")

            # 可视化部分
            if self.config["save_visualizations"]:
                # 可视化从栅格图提取的基础路网 (不含接插件)
                GraphTheory.visualize_raw_skeleton(
                    self.router, raw_nodes, raw_edges,
                    save_path=f"{self.config['output_prefix']}_1a_base_skeleton.png",
                    title="1a. 图像生成的基础路网 (无接插件)"
                )
                # 可视化连接了接插件后的完整骨架
                extractor = TopologyExtractor(self.router, {})
                extractor.visualize_skeleton(keypoints, path_segments,
                                             save_path=f"{self.config['output_prefix']}_1b_complete_skeleton.png",
                                             title="1b. 包含所有潜在路径的完整骨架 (连接接插件后)")
                # 可视化在完整骨架上的初始寻路结果
                visualize_paths_on_skeleton(
                    self.router,
                    keypoints,
                    path_segments,
                    initial_routes,
                    title=f"1c. 在完整骨架上的 {routing_method.upper()} 初始寻路结果",
                    save_path=f"{self.config['output_prefix']}_1c_initial_{routing_method}_routes.png"
                )

        elif self.config["skeleton_mode"] == "hrh":
            # --- 迭代式 HRH 模式 ---
            hrh = HRHRouter(self.router, self.cables)
            hrh.run_hrh(shared_bonus=self.config['hrh_shared_bonus'], max_iterations=self.config['hrh_max_iterations'])
            initial_routes = hrh.routes
            if self.config["save_visualizations"]:
                hrh.visualize(save_path=f"{self.config['output_prefix']}_1_hrh_initial_routes.png",
                              title="1. 初始 HRH 寻路结果")
        else:
            raise ValueError(f"未知的骨架模式: {self.config['skeleton_mode']}")

        self.results["initial_routes"] = initial_routes
        self.timings["phase_1_done"] = time.perf_counter()
        print(f"阶段耗时: {self.timings['phase_1_done'] - start_phase_time:.2f}s")
        self._report_topology_stats("阶段 1: 初始路由", self.results["initial_routes"])


    def _phase_2_topology_extraction_and_refinement(self):
        print("\n=== [2] 拓扑提取与骨架优化 ===")
        extractor = TopologyExtractor(self.router, self.results["initial_routes"])
        keypoints, path_segments = extractor.build_simplified_graph()
        if self.config["save_visualizations"]:
            extractor.visualize_skeleton(keypoints, path_segments, title="2a. 提取出的原始骨架",
                                         save_path=f"{self.config['output_prefix']}_{self.config['skeleton_mode']}_2a_skeleton_raw.png")
        if self.config["enable_skeleton_optimization"]:
            print("  -> 正在执行骨架优化与重构...")
            keypoints, path_segments = extractor.optimize_and_rewire_skeleton(
                keypoints, path_segments,
                merge_distances=(self.config["endpoint_merge_distance"], self.config["internal_merge_distance"]),
                turn_penalty=self.config["skeleton_rewiring_penalty"]
            )
            if self.config["save_visualizations"]:
                extractor.visualize_skeleton(keypoints, path_segments, title="2b. 优化并重构后的骨架",
                                             save_path=f"{self.config['output_prefix']}_{self.config['skeleton_mode']}_2b_skeleton_rewired.png")
        else:
            print("  -> 骨架优化与重构 (已跳过)")

        self.results["optimized_keypoints"] = keypoints
        self.results["optimized_segments"] = path_segments
        self.timings["phase_2_done"] = time.perf_counter()
        print(f"阶段耗时: {self.timings['phase_2_done'] - self.timings['phase_1_done']:.2f}s")
        # [修改] 在阶段末尾添加报告 (直接使用已知结果)
        print(f"\n{'='*10} 阶段 2: 拓扑优化 拓扑统计 {'='*10}")
        print(f"  >>> 阶段 2: 拓扑优化 摘要: {len(keypoints)} 个关键节点 (Nodes), {len(path_segments)} 个路径段 (Edges) <<<")
        print("=" * (len("阶段 2: 拓扑优化") + 22))

    def _phase_3_global_optimization(self):
        print(f"\n=== [3] 全局拓扑优化 (模式: {self.config['optimize_mode']}) ===")
        if self.config["optimize_mode"] == "milp":
            milp_time_limit = self.config.get("milp_time_limit_seconds", 120)
            optimized_paths = milp_scip_based_routing(
                self.results["optimized_keypoints"], self.results["optimized_segments"], self.cables, self.router,
                time_limit_seconds=milp_time_limit
            )
        elif self.config["optimize_mode"] == "milp_cycle":
            milp_time_limit = self.config.get("milp_time_limit_seconds", 120)
            optimized_paths = milp_shortest_paths_for_all_cables(
                self.results["optimized_keypoints"], self.results["optimized_segments"], self.cables, self.router,
                time_limit_seconds=milp_time_limit
            )
        elif self.config["optimize_mode"] == "mst":
            optimized_paths = mst_based_routing(
                self.results["optimized_keypoints"], self.results["optimized_segments"], self.cables, self.router
            )
        elif self.config["optimize_mode"] == "spt":
            optimized_paths = spt_based_routing(
                self.results["optimized_keypoints"], self.results["optimized_segments"], self.cables, self.router
            )
        else:
            raise ValueError(f"未知的优化模式: {self.config['optimize_mode']}")

        self.results["optimized_paths"] = optimized_paths
        self.timings["phase_3_done"] = time.perf_counter()
        print(f"阶段耗时: {self.timings['phase_3_done'] - self.timings['phase_2_done']:.2f}s")
        self._report_topology_stats("阶段 3: 全局优化", self.results["optimized_paths"])

    def _phase_4_geometric_refinement(self):
        print("\n=== [4] 几何路径精细化重构 (拓扑保持) ===")
        offsets = self.problem.get("offsets", (0, 0))
        components = self.problem.get("components_for_drawing", [])
        buffer_val = self.config.get("buffer", 5)  # 确保和 setup 时一致

        refined_paths = refine_topology_strict_topology(
            router=self.router,
            optimized_paths=self.results["optimized_paths"],
            components_data=components,
            offsets=offsets,
            global_buffer=buffer_val,
            turn_penalty=self.config.get("refinement_turn_penalty", 50),
            corridor_bonus=self.config.get("corridor_bonus", 0.5),
            corridor_width=self.config.get("corridor_width", 10),
            exclusion_padding=self.config.get("exclusion_padding", 0),
            visualize_steps=self.config.get("visualize_refinement_steps", True),
            output_prefix=self.config.get("output_prefix", "vis"),
            # [NEW] 传递 B-Spline 参数
            enable_bspline=self.config.get("enable_bspline", True),
            bspline_smoothness=self.config.get("bspline_smoothness", 5000.0),
            bspline_degree=self.config.get("bspline_degree", 3),
            bspline_sample_density=self.config.get("bspline_sample_density", 5),
            min_turning_radius=self.config.get("min_turning_radius", 0.0)
        )
        self.results["refined_paths"] = refined_paths
        self.timings["phase_4_done"] = time.perf_counter()
        print(f"阶段总耗时: {self.timings['phase_4_done'] - self.timings['phase_3_done']:.2f}s")
        self._report_topology_stats("阶段 4: 几何精细化", self.results["refined_paths"])

    def _phase_5_mst_post_refinement(self):
        mst_refined_paths = post_optimization_mst_refinement(
            self.router,
            self.results["refined_paths"],
            original_demands=self.cables
        )
        self.results["mst_refined_paths"] = mst_refined_paths
        self.timings["phase_5_done"] = time.perf_counter()
        print(f"阶段耗时: {self.timings['phase_5_done'] - self.timings['phase_4_done']:.2f}s")
        self._report_topology_stats("阶段 5: MST 后处理", self.results["mst_refined_paths"])

    def _calculate_final_stats(self):
        stats = {}
        def get_total_length(paths):
            if isinstance(paths, dict): # Handle initial routes dict
                paths = paths.values()
            return sum(
                math.hypot(p2[0] - p1[0], p2[1] - p1[1]) for path in paths if path for p1, p2 in zip(path, path[1:]))
        stats["initial_total_length"] = get_total_length(self.results["initial_routes"])
        stats["optimized_total_length"] = get_total_length(self.results["optimized_paths"])
        stats["refined_total_length"] = get_total_length(self.results["refined_paths"])
        stats["mst_refined_total_length"] = get_total_length(self.results["mst_refined_paths"]) # 新增
        stats["final_unique_length"] = calculate_unique_edge_length(self.results["final_paths"]) # 使用最终结果
        stats["final_total_length"] = get_total_length(self.results["final_paths"]) # 使用最终结果
        if stats["initial_total_length"] > 0:
            reduction = stats["initial_total_length"] - stats["final_total_length"]
            stats["reduction_amount"] = reduction
            stats["reduction_percent"] = (reduction / stats["initial_total_length"]) * 100
        else:
            stats["reduction_amount"] = 0
            stats["reduction_percent"] = 0
        self.results["stats"] = stats

    def report_and_visualize(self):
        """生成最终的统计报告并调用可视化函数。"""
        stats = self.results["stats"]
        print("\n" + "="*28 + " [ 最终结果报告 ] " + "="*28)
        print(f"配置摘要: skeleton_mode='{self.config['skeleton_mode']}', "
              f"initial_routing='{self.config.get('initial_routing_method', 'N/A')}', "
              f"optimize_mode='{self.config['optimize_mode']}'")
        print("-" * 70)
        print("【电缆总长度统计】")
        print(f"  - 阶段 1 (初始路由)   : {stats['initial_total_length']:.2f}")
        print(f"  - 阶段 3 (全局优化)   : {stats['optimized_total_length']:.2f}")
        if self.config['enable_geometric_refinement']:
            print(f"  - 阶段 4 (几何精细化) : {stats['refined_total_length']:.2f}")
        if self.config['enable_mst_post_refinement']:
             print(f"  - 阶段 5 (MST后处理)  : {stats['mst_refined_total_length']:.2f}")
        print("-" * 70)
        print("【最终线束评估】")
        print(f"  - 最终线束物理长度 (共享路径计一次): {stats['final_unique_length']:.2f}")
        print(f"  - 最终电缆总长度 (所有电缆长度之和): {stats['final_total_length']:.2f}")
        print(f"  - 相比初始状态，总长度减少         : {stats['reduction_amount']:.2f} ({stats['reduction_percent']:.2f}%)")
        print("-" * 70)
        print("【性能】")
        print(f"  - 总耗时: {self.timings['total']:.2f} 秒")
        print("=" * 70)

        if self.config["save_visualizations"]:
            print("\n正在生成可视化图像...")
            visualize_final_result(
                self.router, self.results["initial_routes"], self.results["optimized_paths"],
                self.problem["components_for_drawing"], self.problem["offsets"][0], self.problem["offsets"][1],
                save_path=f"{self.config['output_prefix']}_3_optimized.png",
                title=f"3. {self.config['optimize_mode'].upper()} 直接优化结果"
            )
            if self.config["enable_geometric_refinement"]:
                visualize_final_result(
                    self.router, self.results["initial_routes"], self.results["refined_paths"],
                    self.problem["components_for_drawing"], self.problem["offsets"][0], self.problem["offsets"][1],
                    save_path=f"{self.config['output_prefix']}_4_geometric_refined.png",
                    title="4. 几何精细化结果"
                )
            if self.config["enable_mst_post_refinement"]:
                 visualize_final_result(
                    self.router, self.results["initial_routes"], self.results["mst_refined_paths"],
                    self.problem["components_for_drawing"], self.problem["offsets"][0], self.problem["offsets"][1],
                    save_path=f"{self.config['output_prefix']}_5_final_mst_refined.png",
                    title="5. 最终结果 (经过MST后处理)"
                )

# ==============================================================================
# 4. 主执行入口
# ==============================================================================
def main():
    """
    主执行函数。
    该函数是整个电缆布线优化流程的入口点。它按顺序执行以下操作：
    1. 定义组件布局和电缆连接等原始输入数据。
    2. 加载和(可选地)修改默认配置参数。
    3. 调用 setup_problem_instance 来初始化布线环境，包括栅格、障碍物等。
    4. 创建 RoutingPipeline (流程引擎) 的实例。
    5. 运行流程引擎，它将按顺序执行所有优化阶段。
    6. 调用流程引擎的报告和可视化方法，输出最终结果。
    """

    # --- 步骤 1: 定义原始数据 ---
    # 组件数据: {'l': 长度, 'w': 宽度, 'x': 中心x, 'y': 中心y, 'connector': 出线方向, 'safety_distance': 安全距离}
    # 在组件列表后添加注释索引，方便与电缆连接数据对应
    components_data = [
        {'l': 255.609, 'w': 263.025, 'x': -1017.005, 'y': -326.005, 'connector': '上', 'safety_distance': 0}, # 0
        {'l': 91.647, 'w': 399.393, 'x': -1027.715, 'y': 273.625, 'connector': '下', 'safety_distance': 0},  # 1
        {'l': 162.747, 'w': 162.747, 'x': -1008.215, 'y': 468.595, 'connector': '左', 'safety_distance': 0}, # 2
        {'l': 145.8, 'w': 62.1, 'x': -583.92, 'y': 498.34, 'connector': '上', 'safety_distance': 0},      # 3
        {'l': 153.9, 'w': 68.4, 'x': -488.24, 'y': 487, 'connector': '上', 'safety_distance': 0},        # 4
        {'l': 532.8, 'w': 425.826, 'x': -351.62, 'y': 26.5, 'connector': '左', 'safety_distance': 0},       # 5
        {'l': 145.8, 'w': 62.1, 'x': -577.34, 'y': -360.41, 'connector': '左', 'safety_distance': 0},     # 6
        {'l': 87.3, 'w': 217.98, 'x': -400.1, 'y': -358, 'connector': '右', 'safety_distance': 0},       # 7
        {'l': 62.1, 'w': 145.8, 'x': -406.51, 'y': -480.61, 'connector': '下', 'safety_distance': 0},     # 8
        {'l': 136.458, 'w': 136.458, 'x': -24.65, 'y': 179.15, 'connector': '上', 'safety_distance': 0},   # 9
        {'l': 136.458, 'w': 136.458, 'x': -40.89, 'y': -231.37, 'connector': '右', 'safety_distance': 0},  # 10
        {'l': 120.6, 'w': 27, 'x': -227.32, 'y': 394, 'connector': '右', 'safety_distance': 0},         # 11
        {'l': 187.65, 'w': 189, 'x': 179.77, 'y': -73.72, 'connector': '上', 'safety_distance': 0},      # 12
        {'l': 125.253, 'w': 34.515, 'x': 161.955, 'y': 233.125, 'connector': '右', 'safety_distance': 0}, # 13
        {'l': 125.73, 'w': 32.742, 'x': 108.38, 'y': 433.15, 'connector': '右', 'safety_distance': 0},  # 14
        {'l': 32.742, 'w': 125.73, 'x': 448.23, 'y': 392, 'connector': '上', 'safety_distance': 0},    # 15
        {'l': 136.458, 'w': 136.458, 'x': 363.29, 'y': -227.57, 'connector': '下', 'safety_distance': 0}, # 16
        {'l': 532.8, 'w': 425.826, 'x': 688.38, 'y': -32.5, 'connector': '左', 'safety_distance': 0},      # 17
        {'l': 189, 'w': 127.8, 'x': 697, 'y': 393, 'connector': '上', 'safety_distance': 0},        # 18
        {'l': 263.7, 'w': 174.6, 'x': 917.5, 'y': 450.5, 'connector': '右', 'safety_distance': 0},     # 19
        {'l': 153.9, 'w': 68.4, 'x': 825, 'y': -418, 'connector': '左', 'safety_distance': 0},        # 20
        {'l': 219.6, 'w': 130.5, 'x': 950.75, 'y': -452, 'connector': '下', 'safety_distance': 0},     # 21
        {'l': 106.2, 'w': 72.18, 'x': 963.58, 'y': -147, 'connector': '右', 'safety_distance': 0},      # 22
        {'l': 106.2, 'w': 72.18, 'x': 963.58, 'y': 141, 'connector': '右', 'safety_distance': 0},       # 23
        {'l': 225.18, 'w': 243, 'x': 1265, 'y': 389.5, 'connector': '右', 'safety_distance': 0},      # 24
        {'l': 192.6, 'w': 245.25, 'x': 1256.76, 'y': -400.75, 'connector': '右', 'safety_distance': 0}, # 25
        {'l': 138.465, 'w': 151.074, 'x': 1251.23, 'y': 76.925, 'connector': '左', 'safety_distance': 0},  # 26
        {'l': 138.537, 'w': 151.074, 'x': 1258.77, 'y': -82.965, 'connector': '左', 'safety_distance': 0}, # 27
        {'l': 117, 'w': 140.85, 'x': 1589.25, 'y': -508, 'connector': '下', 'safety_distance': 0},     # 28
        {'l': 145.8, 'w': 62.1, 'x': 1585.98, 'y': 8.67, 'connector': '右', 'safety_distance': 0},      # 29
        {'l': 1044, 'w': 19.26, 'x': -727.62, 'y': -3, 'connector': '左', 'safety_distance': 0},        # 30 (not used)
        {'l': 1044, 'w': 19.26, 'x': 1064.38, 'y': -3, 'connector': '左', 'safety_distance': 0}         # 31 (not used)
    ]

    # 电缆连接需求: (起点组件索引, 终点组件索引)
    cable_connections = [
        (0, 3), (0, 4), (0, 9), (0, 15), (1, 6), (1, 6), (1, 3), (1, 4), (1, 14), (2, 7), (2, 20), (2, 23),
        (2, 21), (2, 13), (2, 9), (5, 22), (5, 14), (5, 26), (5, 23), (5, 26), (5, 27), (5, 28), (5, 28),
        (5, 29), (5, 21), (5, 15), (5, 7), (5, 20), (5, 21), (7, 21), (7, 27), (7, 28), (7, 28), (7, 28),
        (7, 8), (7, 9), (7, 13), (7, 15), (12, 9), (12, 22), (12, 22), (12, 23), (12, 23), (16, 22),
        (16, 26), (16, 26), (16, 9), (10, 20), (10, 26), (10, 27), (10, 27), (18, 29), (18, 11), (18, 8),
        (18, 20), (18, 27), (18, 17), (19, 9), (19, 21), (19, 5), (19, 24), (19, 2), (19, 17), (24, 25),
        (24, 7), (24, 12), (25, 17), (25, 19), (25, 16)
    ]

    # --- 步骤 2: 获取并(可选地)自定义配置 ---
    config = get_default_config()
    '''
    默认配置
    skeleton_mode = Graph
    initial_routing_method = MILP
    optimize_mode = MILP
    grid_scale = 4
    允许对角移动
    开关全为 TRUE
    MILP求解时间（milp_time_limit_seconds）为 300s

    '''

    # --- 示例：如何修改配置 ---
    # config["skeleton_mode"] = "hrh"              # 切换到HRH模式,
    # config["initial_routing_method"] = "dijkstra" # 在graph模式下，使用更快的Dijkstra进行初始寻路
    # config["optimize_mode"] = "spt"              # 使用启发式的MST/spt进行全局无环优化，milp_cycle进行不考虑成环
    # config["grid_scale"] = 8                     # 使用更粗的网格以加速A*计算
    # config["enable_diagonal_search"] = False        # 关闭对角搜索
    # config["enable_skeleton_optimization"] = False      # 关闭骨架优化
    config["enable_geometric_refinement"] = False       # 关闭几何精细化
    config["enable_mst_post_refinement"] = False        # 关闭精细化无环处理
    # config["milp_time_limit_seconds"] =  600          # 修改MILP最大求解时长

    # --- 步骤 3: 准备问题实例 ---
    problem = setup_problem_instance(components_data, cable_connections, config=config)

    # --- 步骤 4: 创建并运行优化流程 ---
    pipeline = RoutingPipeline(config, problem)
    pipeline.run()

    # --- 步骤 5: 报告并可视化结果 ---
    pipeline.report_and_visualize()


if __name__ == "__main__":
    main()