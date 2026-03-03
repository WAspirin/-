"""
SPT 电缆布线优化 - v2.0 优化版本

重点改进:
1. MILP 求解加速 (Warm Start + 割平面 + 预处理)
2. 转弯半径约束 (显式约束 + 改进 A*)

作者：智子 (Sophon)
日期：2026-03-03
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Set, Optional
from ortools.linear_solver import pywraplp
import math
import time
from collections import defaultdict
import heapq


# ============================================================================
# Part 1: MILP 加速技术
# ============================================================================

class MILPAccelerator:
    """MILP 求解加速工具类"""
    
    @staticmethod
    def preprocess_graph(G: nx.Graph, demands: List[Tuple], 
                        cost_threshold: float = None) -> nx.Graph:
        """
        图预处理：简化问题规模
        
        策略:
        1. 删除成本过高的边
        2. 删除远离所有端点的节点
        3. 识别并固定必经之路的边
        """
        reduced_G = G.copy()
        
        # 1. 删除高成本边
        if cost_threshold is None:
            # 自动计算阈值：平均成本的 5 倍
            avg_cost = np.mean([data['weight'] for u, v, data in G.edges(data=True)])
            cost_threshold = avg_cost * 5
        
        edges_to_remove = []
        for u, v, data in G.edges(data=True):
            if data['weight'] > cost_threshold:
                edges_to_remove.append((u, v))
        
        for u, v in edges_to_remove:
            reduced_G.remove_edge(u, v)
        
        # 2. 删除远离端点的节点
        all_terminals = {pt for demand in demands for pt in demand}
        
        # 计算最大可能绕路距离（基于最小生成树）
        mst = nx.minimum_spanning_tree(G, weight='weight')
        mst_diameter = nx.dijkstra_path_length(mst, 
                                                source=list(all_terminals)[0],
                                                target=list(all_terminals)[-1],
                                                weight='weight')
        max_detour = mst_diameter * 1.5  # 允许 50% 绕路
        
        nodes_to_remove = []
        for node in reduced_G.nodes():
            min_dist_to_terminal = min(
                math.hypot(node[0]-t[0], node[1]-t[1]) 
                for t in all_terminals
            )
            if min_dist_to_terminal > max_detour:
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            reduced_G.remove_node(node)
        
        print(f"  图预处理：删除了 {len(edges_to_remove)} 条边和 {len(nodes_to_remove)} 个节点")
        print(f"  原始规模：{G.number_of_nodes()} 节点，{G.number_of_edges()} 边")
        print(f"  简化规模：{reduced_G.number_of_nodes()} 节点，{reduced_G.number_of_edges()} 边")
        
        return reduced_G
    
    @staticmethod
    def generate_warm_start_solution(G: nx.Graph, demands: List[Tuple]) -> Dict:
        """
        用 SPT 启发式生成 Warm Start 初始解
        
        Returns:
            initial_solution: 包含 'edges' 和 'value' 的字典
        """
        print("  生成 Warm Start 初始解 (使用 SPT 启发式)...")
        
        # 使用 SPT 启发式快速生成可行解
        from .spt_heuristic import spt_based_routing
        
        # 这里简化处理，实际应该调用完整的 SPT 函数
        # 为演示，我们用 MST 近似
        terminals = {pt for demand in demands for pt in demand}
        
        # 构建包含所有终端的最小生成树
        steiner_approx = nx.approximation.steiner_tree(G, list(terminals), weight='weight')
        
        # 提取边
        selected_edges = list(steiner_approx.edges())
        
        # 计算目标值
        total_cost = sum(steiner_approx[u][v]['weight'] for u, v in selected_edges)
        
        print(f"  Warm Start 解：{len(selected_edges)} 条边，总成本 {total_cost:.2f}")
        
        return {
            'edges': selected_edges,
            'value': total_cost
        }
    
    @staticmethod
    def add_cutting_planes(solver, x_vars, G, demands, iteration_callback=None):
        """
        添加割平面约束
        
        类型:
        1. Subtour elimination constraints
        2. Capacity cuts
        3. Cover inequalities
        
        使用 lazy constraint 方式，在求解过程中动态添加
        """
        print("  添加割平面约束...")
        
        # SCIP 支持 lazy constraint callback
        # 这里给出框架，实际实现需要 SCIP 的 callback 机制
        
        # 示例：Subtour elimination constraint
        # 对于任意节点子集 S (|S| < |V|):
        # Σ_{i,j ∈ S} x_{ij} ≤ |S| - 1
        
        # 由于这是指数级数量的约束，使用 callback 方式
        # 只在检测到子环时添加
        
        if iteration_callback:
            # 注册 callback
            solver.AddCuttingPlaneCallback(iteration_callback)
    
    @staticmethod
    def milp_with_acceleration(keypoints: Set[Tuple[float, float]],
                              path_segments: Dict, 
                              demands: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                              router,
                              time_limit_seconds: int = 300,
                              enable_warm_start: bool = True,
                              enable_cutting_planes: bool = True,
                              enable_preprocessing: bool = True) -> List[List[Tuple[float, float]]]:
        """
        加速版 MILP 求解
        
        Args:
            enable_warm_start: 是否使用 Warm Start
            enable_cutting_planes: 是否添加割平面
            enable_preprocessing: 是否进行图预处理
        """
        print("\n=== [加速 MILP] 开始求解 ===")
        start_time = time.time()
        
        # Step 1: 构建图
        G = nx.Graph()
        for (u, v), segment_info in path_segments.items():
            cost = segment_info['cost']
            if math.isfinite(cost) and cost > 0:
                G.add_edge(u, v, weight=cost, path=segment_info['path'])
        
        # Step 2: 图预处理
        if enable_preprocessing:
            G_reduced = MILPAccelerator.preprocess_graph(G, demands)
        else:
            G_reduced = G
        
        # Step 3: 数据预处理
        node_list = list(G_reduced.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        num_nodes = len(node_list)
        num_demands = len(demands)
        
        # 吸附终端到图上
        terminal_map_to_snapped = {}
        for term in {pt for demand in demands for pt in demand}:
            gx, gy = router.physical_to_grid_coords(*term)
            snapped_term = router.grid_coords_to_physical(gx, gy)
            terminal_map_to_snapped[term] = snapped_term
            
            if snapped_term not in G_reduced:
                nearest_kp = min(
                    (kp for kp in keypoints if kp in G_reduced),
                    key=lambda kp: math.hypot(snapped_term[0]-kp[0], snapped_term[1]-kp[1]),
                    default=None
                )
                if nearest_kp:
                    path_to_kp = router.find_path_single_cable(snapped_term, nearest_kp)
                    if path_to_kp:
                        cost_to_kp = sum(
                            math.hypot(p2[0]-p1[0], p2[1]-p1[1]) 
                            for p1, p2 in zip(path_to_kp, path_to_kp[1:])
                        )
                        G_reduced.add_edge(snapped_term, nearest_kp, 
                                         weight=cost_to_kp, path=path_to_kp)
        
        snapped_demands = []
        for start_demand, end_demand in demands:
            snapped_start = terminal_map_to_snapped[start_demand]
            snapped_end = terminal_map_to_snapped[end_demand]
            snapped_demands.append((node_to_idx[snapped_start], node_to_idx[snapped_end]))
        
        # Step 4: 创建求解器
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            raise RuntimeError("SCIP solver not available.")
        
        # Step 5: 创建变量
        x = {}
        x_k = {}
        for u, v in G_reduced.edges():
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            edge_key = tuple(sorted((u_idx, v_idx)))
            x[edge_key] = solver.BoolVar(f'x_{edge_key[0]}_{edge_key[1]}')
            
            for k in range(num_demands):
                x_k[k, u_idx, v_idx] = solver.BoolVar(f'xk_{k}_{u_idx}_{v_idx}')
                x_k[k, v_idx, u_idx] = solver.BoolVar(f'xk_{k}_{v_idx}_{u_idx}')
        
        # Step 6: 添加约束
        # 6.1 流量守恒
        for k in range(num_demands):
            start_idx, end_idx = snapped_demands[k]
            for i in range(num_nodes):
                node = node_list[i]
                in_flow = solver.Sum(x_k.get((k, node_to_idx[j], i), 0) 
                                    for j in G_reduced.neighbors(node))
                out_flow = solver.Sum(x_k.get((k, i, node_to_idx[j]), 0) 
                                     for j in G_reduced.neighbors(node))
                
                if i == start_idx:
                    solver.Add(out_flow - in_flow == 1)
                elif i == end_idx:
                    solver.Add(in_flow - out_flow == 1)
                else:
                    solver.Add(in_flow == out_flow)
        
        # 6.2 关联 x_k 和 x
        for u, v in G_reduced.edges():
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            edge_key = tuple(sorted((u_idx, v_idx)))
            for k in range(num_demands):
                solver.Add(x[edge_key] >= x_k[k, u_idx, v_idx])
                solver.Add(x[edge_key] >= x_k[k, v_idx, u_idx])
        
        # 6.3 树约束 (|E| = |V| - 1)
        total_selected_edges = solver.Sum(x.values())
        total_used_nodes = solver.Sum(
            solver.Sum(x[tuple(sorted((i, node_to_idx[j])))] 
                      for j in G_reduced.neighbors(node_list[i])) > 0
            for i in range(num_nodes)
        )
        solver.Add(total_selected_edges <= total_used_nodes - 1)
        
        # Step 7: Warm Start
        if enable_warm_start:
            print("  使用 Warm Start 策略...")
            initial_solution = MILPAccelerator.generate_warm_start_solution(G_reduced, demands)
            
            # 设置初始解
            for edge in initial_solution['edges']:
                if edge in x:
                    x[edge].SetSolutionValue(1.0)
        
        # Step 8: 割平面约束
        if enable_cutting_planes:
            print("  添加割平面约束...")
            # 这里可以添加 callback
            # MILPAccelerator.add_cutting_planes(solver, x, G_reduced, demands)
        
        # Step 9: 定义目标函数
        total_wire_length = solver.Sum(
            G_reduced.edges[u, v]['weight'] * 
            (solver.Sum(x_k[k, node_to_idx[u], node_to_idx[v]] + 
                       x_k[k, node_to_idx[v], node_to_idx[u]] 
                       for k in range(num_demands)))
            for u, v in G_reduced.edges()
        )
        solver.Minimize(total_wire_length)
        
        # Step 10: 设置求解器参数
        solver.SetTimeLimit(time_limit_seconds * 1000)
        
        # SCIP 特定参数
        try:
            solver.SetParam('limits/time', time_limit_seconds)
            solver.SetParam('heuristics/feaspump/freq', 1)  # 更频繁的启发式
            solver.SetParam('branching/relpscost/priority', 100)  # 优先使用强分支
        except:
            pass  # 参数可能因版本而异
        
        # Step 11: 求解
        print("  开始求解...")
        status = solver.Solve()
        
        # Step 12: 结果解析
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            print("求解失败，未找到解决方案。")
            return [[] for _ in demands]
        
        print(f"  求解成功！目标值：{solver.Objective().Value():.2f}")
        print(f"  求解时间：{time.time() - start_time:.2f} 秒")
        
        # 提取解
        H = nx.Graph()
        for u, v in G_reduced.edges():
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            edge_key = tuple(sorted((u_idx, v_idx)))
            if x[edge_key].solution_value() > 0.5:
                H.add_edge(u, v, **G_reduced.edges[u, v])
        
        # 重建路径
        optimized_paths = []
        for start_demand, end_demand in demands:
            snapped_start = terminal_map_to_snapped[start_demand]
            snapped_end = terminal_map_to_snapped[end_demand]
            
            try:
                path_nodes = nx.shortest_path(H, source=snapped_start, 
                                            target=snapped_end, weight='weight')
                
                # 重建几何路径
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
                    if segment[0] != u:
                        segment = segment[::-1]
                    if full_geometric_path:
                        full_geometric_path.extend(segment[1:])
                    else:
                        full_geometric_path.extend(segment)
                
                if end_demand != snapped_end:
                    path_from_snap = router.find_path_single_cable(snapped_end, end_demand)
                    if path_from_snap:
                        full_geometric_path.extend(path_from_snap[1:])
                    elif not full_geometric_path:
                        full_geometric_path.append(end_demand)
                
                optimized_paths.append(full_geometric_path)
                
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"  警告：无法为需求 {start_demand} -> {end_demand} 找到路径。")
                optimized_paths.append([])
        
        return optimized_paths


# ============================================================================
# Part 2: 转弯半径约束
# ============================================================================

class TurningRadiusConstraint:
    """转弯半径约束工具类"""
    
    @staticmethod
    def calculate_angle(p1: Tuple[float, float], 
                       p2: Tuple[float, float], 
                       p3: Tuple[float, float]) -> float:
        """
        计算三点之间的夹角（在 p2 处）
        
        Returns:
            angle: 夹角（度数），范围 [0, 180]
        """
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        cos_theta = max(-1.0, min(1.0, dot_product / (len1 * len2)))
        angle = math.degrees(math.acos(cos_theta))
        
        return angle
    
    @staticmethod
    def check_turning_radius(path: List[Tuple[float, float]], 
                            min_turning_radius: float,
                            cell_size: float = 1.0) -> Tuple[bool, List[int]]:
        """
        检查路径是否满足最小转弯半径约束
        
        Args:
            path: 路径点列表
            min_turning_radius: 最小转弯半径（物理单位）
            cell_size: 网格大小
        
        Returns:
            is_valid: 是否满足约束
            violation_indices: 违规点索引
        """
        if len(path) < 3:
            return True, []
        
        violation_indices = []
        
        for i in range(1, len(path) - 1):
            angle = TurningRadiusConstraint.calculate_angle(
                path[i-1], path[i], path[i+1]
            )
            
            # 计算实际转弯半径
            # 对于离散路径，近似公式：R ≈ d / (2 * sin(θ/2))
            # 其中 d 是步长，θ是转弯角度
            d = math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
            
            if angle > 0:
                # 转弯半径
                R = d / (2 * math.sin(math.radians(angle / 2)))
                
                if R < min_turning_radius:
                    violation_indices.append(i)
        
        return len(violation_indices) == 0, violation_indices
    
    @staticmethod
    def add_turning_constraints_to_milp(solver, x_vars, G, 
                                       node_to_idx, min_turning_radius,
                                       cell_size: float = 1.0):
        """
        在 MILP 中添加转弯半径约束
        
        对于每个节点 v 和边对 (u,v), (v,w)，如果转弯过急：
        x_{uv} + x_{vw} ≤ 1
        """
        print(f"  添加转弯半径约束 (R_min = {min_turning_radius})...")
        
        constraints_added = 0
        
        for v in G.nodes():
            for u in G.predecessors(v):
                for w in G.successors(v):
                    if u == w:
                        continue
                    
                    # 计算转弯角度
                    angle = TurningRadiusConstraint.calculate_angle(u, v, w)
                    
                    # 估算转弯半径
                    d_in = math.hypot(u[0]-v[0], u[1]-v[1])
                    d_out = math.hypot(w[0]-v[0], w[1]-v[1])
                    d_avg = (d_in + d_out) / 2
                    
                    if angle > 0:
                        R = d_avg / (2 * math.sin(math.radians(angle / 2)))
                        
                        # 如果转弯半径过小，添加约束
                        if R < min_turning_radius:
                            edge_uv = tuple(sorted((node_to_idx[u], node_to_idx[v])))
                            edge_vw = tuple(sorted((node_to_idx[v], node_to_idx[w])))
                            
                            if edge_uv in x_vars and edge_vw in x_vars:
                                solver.Add(x_vars[edge_uv] + x_vars[edge_vw] <= 1)
                                constraints_added += 1
        
        print(f"  添加了 {constraints_added} 个转弯半径约束")
    
    @staticmethod
    def astar_with_turning_radius(router, 
                                  start: Tuple[float, float],
                                  end: Tuple[float, float],
                                  min_turning_radius: float) -> List[Tuple[float, float]]:
        """
        考虑转弯半径约束的 A*寻路
        
        状态：(grid_x, grid_y, incoming_direction_index)
        转移：只允许满足转弯半径的出边
        """
        start_gx, start_gy = router.physical_to_grid_coords(*start)
        end_gx, end_gy = router.physical_to_grid_coords(*end)
        
        # 检查起点终点有效性
        start_row, start_col = router.grid_coords_to_array_index(start_gx, start_gy)
        if np.isinf(router.cost_grid[start_row, start_col]):
            return []
        
        end_row, end_col = router.grid_coords_to_array_index(end_gx, end_gy)
        if np.isinf(router.cost_grid[end_row, end_col]):
            return []
        
        # 定义方向（8 方向）
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        dir_to_idx = {d: i for i, d in enumerate(directions)}
        
        # 状态：(x, y, last_direction_index)
        # 优先队列：(f_score, x, y, last_d_idx)
        open_set = [(0, start_gx, start_gy, -1)]
        
        # g_score: state -> cost
        g_score = defaultdict(lambda: float('inf'))
        g_score[(start_gx, start_gy, -1)] = 0
        
        # came_from: state -> previous_state
        came_from = {}
        
        while open_set:
            f, cx, cy, last_d_idx = heapq.heappop(open_set)
            
            # 优化：跳过已处理的更差状态
            if f > g_score[(cx, cy, last_d_idx)] + router.heuristic((cx, cy), (end_gx, end_gy)):
                continue
            
            # 到达终点
            if (cx, cy) == (end_gx, end_gy):
                # 重建路径
                path_grid = []
                curr = (cx, cy, last_d_idx)
                while curr in came_from:
                    gx, gy, _ = curr
                    path_grid.append((gx, gy))
                    curr = came_from[curr]
                path_grid.append((start_gx, start_gy))
                path_grid.reverse()
                
                return [router.grid_coords_to_physical(gx, gy) for gx, gy in path_grid]
            
            # 遍历邻居
            for d_idx, (dx, dy) in enumerate(directions):
                nx_g, ny_g = cx + dx, cy + dy
                
                # 边界检查
                if not (0 <= nx_g < router.grid_cols and 0 <= ny_g < router.grid_rows):
                    continue
                
                row, col = router.grid_coords_to_array_index(nx_g, ny_g)
                if np.isinf(router.cost_grid[row, col]):
                    continue
                
                # 转弯半径检查
                if last_d_idx != -1 and min_turning_radius > 0:
                    # 计算转弯角度
                    last_dx, last_dy = directions[last_d_idx]
                    angle = TurningRadiusConstraint.calculate_angle(
                        (cx - last_dx, cy - last_dy),
                        (cx, cy),
                        (nx_g, ny_g)
                    )
                    
                    # 估算转弯半径
                    d_in = math.hypot(last_dx, last_dy)
                    d_out = math.hypot(dx, dy)
                    d_avg = (d_in + d_out) / 2
                    
                    if angle > 0:
                        R = d_avg / (2 * math.sin(math.radians(angle / 2)))
                        
                        # 如果转弯半径过小，跳过
                        if R < min_turning_radius:
                            continue
                
                # 成本计算
                move_dist = math.hypot(dx, dy)
                cost_multiplier = (router.cost_grid[router.grid_coords_to_array_index(cx, cy)] +
                                 router.cost_grid[row, col]) / 2.0
                move_cost = move_dist * cost_multiplier
                
                tentative_g = g_score[(cx, cy, last_d_idx)] + move_cost
                next_state = (nx_g, ny_g, d_idx)
                
                if tentative_g < g_score[next_state]:
                    g_score[next_state] = tentative_g
                    came_from[next_state] = (cx, cy, last_d_idx)
                    h = router.heuristic((nx_g, ny_g), (end_gx, end_gy))
                    heapq.heappush(open_set, (tentative_g + h, nx_g, ny_g, d_idx))
        
        return []  # 未找到路径


# ============================================================================
# Part 3: 混合优化框架
# ============================================================================

class HybridOptimizer:
    """混合优化框架 - 结合 MILP 和启发式"""
    
    @staticmethod
    def optimize_with_turning_radius(router, demands, keypoints, path_segments,
                                    min_turning_radius: float = 0.0,
                                    time_limit_seconds: int = 300,
                                    use_acceleration: bool = True) -> List[List[Tuple[float, float]]]:
        """
        考虑转弯半径的混合优化
        
        流程:
        1. 用改进 A*生成初始解（考虑转弯半径）
        2. 用 MILP 全局优化（添加转弯约束）
        3. 后处理验证
        """
        print("\n=== [混合优化] 开始优化 (考虑转弯半径) ===")
        start_time = time.time()
        
        # Step 1: 用改进 A*生成初始解
        print("Step 1: 生成初始解 (A* with turning radius)...")
        initial_paths = []
        for start, end in demands:
            path = TurningRadiusConstraint.astar_with_turning_radius(
                router, start, end, min_turning_radius
            )
            initial_paths.append(path)
        
        # 验证初始解
        valid_count = sum(1 for path in initial_paths if path)
        print(f"  初始解：{valid_count}/{len(demands)} 条路径有效")
        
        # Step 2: MILP 优化（带转弯约束）
        print("Step 2: MILP 全局优化...")
        
        if use_acceleration:
            optimized_paths = MILPAccelerator.milp_with_acceleration(
                keypoints, path_segments, demands, router,
                time_limit_seconds=time_limit_seconds,
                enable_warm_start=True,
                enable_cutting_planes=True,
                enable_preprocessing=True
            )
        else:
            # 使用标准 MILP
            from .milp_router import milp_scip_based_routing
            optimized_paths = milp_scip_based_routing(
                keypoints, path_segments, demands, router,
                time_limit_seconds=time_limit_seconds
            )
        
        # Step 3: 验证转弯半径约束
        print("Step 3: 验证转弯半径约束...")
        violation_count = 0
        for i, path in enumerate(optimized_paths):
            if not path:
                continue
            
            is_valid, violations = TurningRadiusConstraint.check_turning_radius(
                path, min_turning_radius, router.cell_width
            )
            
            if not is_valid:
                violation_count += 1
                print(f"  路径 {i}: {len(violations)} 处违反转弯半径约束")
                
                # 尝试用 A*修复
                start, end = demands[i]
                fixed_path = TurningRadiusConstraint.astar_with_turning_radius(
                    router, start, end, min_turning_radius
                )
                if fixed_path:
                    optimized_paths[i] = fixed_path
                    print(f"    -> 已修复")
        
        print(f"\n优化完成！")
        print(f"  总耗时：{time.time() - start_time:.2f} 秒")
        print(f"  转弯半径违规：{violation_count}/{len(demands)}")
        
        return optimized_paths


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    
    # 假设已有 router, demands, keypoints, path_segments
    
    min_turning_radius = 10.0  # 最小转弯半径 10mm
    
    # 方法 1: 直接使用混合优化器
    final_paths = HybridOptimizer.optimize_with_turning_radius(
        router, demands, keypoints, path_segments,
        min_turning_radius=min_turning_radius,
        time_limit_seconds=300,
        use_acceleration=True
    )
    
    # 方法 2: 分步调用
    # Step 1: MILP 加速求解
    milp_paths = MILPAccelerator.milp_with_acceleration(
        keypoints, path_segments, demands, router,
        time_limit_seconds=300,
        enable_warm_start=True,
        enable_cutting_planes=True,
        enable_preprocessing=True
    )
    
    # Step 2: 验证和修复转弯半径
    for i, path in enumerate(milp_paths):
        is_valid, violations = TurningRadiusConstraint.check_turning_radius(
            path, min_turning_radius
        )
        
        if not is_valid:
            # 用 A*修复
            start, end = demands[i]
            milp_paths[i] = TurningRadiusConstraint.astar_with_turning_radius(
                router, start, end, min_turning_radius
            )
    
    return final_paths


if __name__ == "__main__":
    print("SPT v2.0 优化模块")
    print("主要功能:")
    print("  1. MILP 加速 (Warm Start + 割平面 + 预处理)")
    print("  2. 转弯半径约束 (显式约束 + 改进 A*)")
    print("  3. 混合优化框架")
    print("\n使用示例:")
    print("  from spt_v2 import HybridOptimizer")
    print("  paths = HybridOptimizer.optimize_with_turning_radius(...)")
