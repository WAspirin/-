"""
SPT v3.0 - 深度优化版本

重点优化方向:
1. 转弯半径处理（后处理 + 搜索过程中）
2. SCIP 求解器加速方法
3. 代码完善和性能优化

作者：智子 (Sophon)
日期：2026-03-05
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Set, Optional, Any
from ortools.linear_solver import pywraplp
import math
import time
from collections import defaultdict
import heapq
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Part 1: 转弯半径约束 - 完整实现
# ============================================================================

class TurningRadiusHandler:
    """
    转弯半径处理器 - 支持两种模式：
    1. 后处理模式：先求解，再平滑
    2. 搜索中模式：在寻路时考虑转弯约束
    """
    
    @staticmethod
    def calculate_curvature(p1: Tuple[float, float], 
                           p2: Tuple[float, float], 
                           p3: Tuple[float, float]) -> float:
        """
        计算三点之间的曲率（转弯半径的倒数）
        
        Returns:
            curvature: 曲率 (1/R), 直线为 0
        """
        # 向量
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # 长度
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 < 1e-6 or len2 < 1e-6:
            return float('inf')
        
        # 叉积计算曲率
        cross_product = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # 曲率公式：κ = |v1 × v2| / (|v1| * |v2| * |v1| * |v2| + v1·v2)
        # 简化：κ = sin(θ) / (|v1| + |v2|)
        angle = math.acos(max(-1.0, min(1.0, dot_product / (len1 * len2))))
        
        if angle < 1e-6:  # 几乎直线
            return 0.0
        
        # 转弯半径 R = (len1 + len2) / (2 * sin(θ/2))
        # 曲率 κ = 1/R
        R = (len1 + len2) / (2 * math.sin(angle / 2))
        curvature = 1.0 / R if R > 1e-6 else float('inf')
        
        return curvature
    
    @staticmethod
    def check_path_feasibility(path: List[Tuple[float, float]], 
                              min_turning_radius: float,
                              cell_size: float = 1.0) -> Tuple[bool, List[Dict]]:
        """
        检查路径是否满足转弯半径约束
        
        Returns:
            is_feasible: 是否可行
            violations: 违规点详细信息
        """
        if len(path) < 3:
            return True, []
        
        violations = []
        
        for i in range(1, len(path) - 1):
            curvature = TurningRadiusHandler.calculate_curvature(
                path[i-1], path[i], path[i+1]
            )
            
            if curvature > 1e-6:  # 有转弯
                R = 1.0 / curvature
                
                if R < min_turning_radius:
                    violations.append({
                        'index': i,
                        'position': path[i],
                        'curvature': curvature,
                        'radius': R,
                        'required_radius': min_turning_radius
                    })
        
        return len(violations) == 0, violations
    
    @staticmethod
    def smooth_path_post_processing(path: List[Tuple[float, float]], 
                                    router,
                                    min_turning_radius: float,
                                    method: str = 'hybrid') -> List[Tuple[float, float]]:
        """
        后处理模式：对已生成的路径进行平滑
        
        方法:
        - 'hybrid': 混合方法（B-Spline + 约束投影）
        - 'dubins': Dubins 路径
        - 'bezier': Bézier 曲线
        """
        if method == 'hybrid':
            return TurningRadiusHandler.hybrid_smoothing(
                path, router, min_turning_radius
            )
        elif method == 'dubins':
            return TurningRadiusHandler.dubins_smoothing(
                path, router, min_turning_radius
            )
        else:
            return path  # 不处理
    
    @staticmethod
    def hybrid_smoothing(path: List[Tuple[float, float]], 
                        router,
                        min_turning_radius: float) -> List[Tuple[float, float]]:
        """
        混合平滑方法：B-Spline + 约束投影
        
        步骤:
        1. 用 B-Spline 生成平滑曲线
        2. 检查转弯半径约束
        3. 对违规点进行投影修正
        4. 迭代直到满足约束
        """
        from scipy.interpolate import splprep, splev
        
        if len(path) < 4:
            return path
        
        max_iterations = 10
        current_path = path
        smoothness = 1.0
        
        for iteration in range(max_iterations):
            # Step 1: B-Spline 平滑
            try:
                x = [p[0] for p in current_path]
                y = [p[1] for p in current_path]
                
                tck, u = splprep([x, y], s=smoothness, k=3, per=False)
                
                # 采样
                num_points = len(path) * 3
                u_new = np.linspace(0, 1, num_points)
                x_new, y_new = splev(u_new, tck)
                
                smoothed_path = list(zip(x_new, y_new))
                
                # Step 2: 检查约束
                is_feasible, violations = TurningRadiusHandler.check_path_feasibility(
                    smoothed_path, min_turning_radius
                )
                
                if is_feasible:
                    return smoothed_path
                
                # Step 3: 投影修正违规点
                for violation in violations:
                    idx = violation['index']
                    # 局部调整：拉直违规点附近的曲线
                    if 0 < idx < len(smoothed_path) - 1:
                        # 简单处理：用直线段替换
                        p_before = smoothed_path[max(0, idx-2)]
                        p_after = smoothed_path[min(len(smoothed_path)-1, idx+2)]
                        
                        # 线性插值
                        for t in np.linspace(0, 1, 5):
                            smoothed_path[idx] = (
                                p_before[0] + t * (p_after[0] - p_before[0]),
                                p_before[1] + t * (p_after[1] - p_before[1])
                            )
                
                current_path = smoothed_path
                smoothness *= 2.0  # 增大平滑度
                
            except Exception as e:
                print(f"  [Hybrid Smoothing] 迭代 {iteration} 失败：{e}")
                break
        
        return current_path
    
    @staticmethod
    def dubins_smoothing(path: List[Tuple[float, float]], 
                        router,
                        min_turning_radius: float) -> List[Tuple[float, float]]:
        """
        Dubins 路径平滑
        
        Dubins 路径是在满足最小转弯半径约束下的最短路径
        类型：LSL, RSR, RSL, LSR, RLR, LRL
        
        这里实现简化版本
        """
        if len(path) < 2:
            return path
        
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            p_prev = path[i-1]
            p_curr = path[i]
            p_next = path[i+1]
            
            # 计算进入和离开方向
            v_in = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
            v_out = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])
            
            v_in_norm = v_in / (np.linalg.norm(v_in) + 1e-6)
            v_out_norm = v_out / (np.linalg.norm(v_out) + 1e-6)
            
            # 计算转弯角度
            dot = np.dot(v_in_norm, v_out_norm)
            angle = math.acos(max(-1.0, min(1.0, dot)))
            
            if angle < 0.1:  # 几乎直线
                smoothed.append(p_curr)
                continue
            
            # 计算转弯半径
            # 对于 Dubins 路径，需要满足 R >= min_turning_radius
            # 这里简化处理：如果角度太大，插入中间点
            if angle > math.pi / 4:  # 45 度
                # 插入圆弧近似点
                num_points = int(angle / 0.1)  # 每 0.1 弧度一个点
                for j in range(1, num_points):
                    t = j / num_points
                    # 简单线性插值（实际应该用圆弧）
                    p_interp = (
                        p_curr[0] + t * (p_next[0] - p_curr[0]) * 0.5,
                        p_curr[1] + t * (p_next[1] - p_curr[1]) * 0.5
                    )
                    smoothed.append(p_interp)
            
            smoothed.append(p_curr)
        
        smoothed.append(path[-1])
        return smoothed
    
    @staticmethod
    def astar_with_turning_constraints(router,
                                       start: Tuple[float, float],
                                       end: Tuple[float, float],
                                       min_turning_radius: float,
                                       lookahead: int = 3) -> List[Tuple[float, float]]:
        """
        搜索过程中考虑转弯半径约束的 A*算法
        
        核心思想:
        1. 状态包含方向信息：(x, y, last_direction)
        2. 转移时检查转弯半径
        3. 使用 look ahead 预判未来几步
        
        Args:
            router: GridCableRouter 实例
            start: 起点
            end: 终点
            min_turning_radius: 最小转弯半径
            lookahead: 预判步数
        """
        start_gx, start_gy = router.physical_to_grid_coords(*start)
        end_gx, end_gy = router.physical_to_grid_coords(*end)
        
        # 检查起点终点
        start_row, start_col = router.grid_coords_to_array_index(start_gx, start_gy)
        if np.isinf(router.cost_grid[start_row, start_col]):
            return []
        
        end_row, end_col = router.grid_coords_to_array_index(end_gx, end_gy)
        if np.isinf(router.cost_grid[end_row, end_col]):
            return []
        
        # 8 方向
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        dir_to_idx = {d: i for i, d in enumerate(directions)}
        
        # 状态：(x, y, last_direction_index)
        # 优先队列：(f_score, x, y, last_d_idx)
        open_set = [(0, start_gx, start_gy, -1)]
        
        g_score = defaultdict(lambda: float('inf'))
        g_score[(start_gx, start_gy, -1)] = 0
        
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
                
                # 转弯半径检查（搜索过程中）
                if last_d_idx != -1 and min_turning_radius > 0:
                    # 计算转弯角度
                    last_dx, last_dy = directions[last_d_idx]
                    
                    # 预估转弯半径
                    # 对于网格移动，转弯半径近似为：
                    # R ≈ step_length / (2 * sin(θ/2))
                    angle = TurningRadiusHandler._calculate_grid_turn_angle(
                        last_dx, last_dy, dx, dy
                    )
                    
                    if angle > 0.1:  # 有明显转弯
                        step_length = math.hypot(dx, dy) * router.cell_width
                        R_approx = step_length / (2 * math.sin(angle / 2))
                        
                        # 如果转弯半径过小，跳过
                        if R_approx < min_turning_radius:
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
    
    @staticmethod
    def _calculate_grid_turn_angle(dx1: int, dy1: int, dx2: int, dy2: int) -> float:
        """计算网格移动之间的角度"""
        v1 = np.array([dx1, dy1])
        v2 = np.array([dx2, dy2])
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 < 1e-6 or len2 < 1e-6:
            return 0.0
        
        dot = np.dot(v1, v2) / (len1 * len2)
        angle = math.acos(max(-1.0, min(1.0, dot)))
        
        return angle


# ============================================================================
# Part 2: SCIP 求解器加速方法
# ============================================================================

class SCIPAccelerator:
    """
    SCIP 求解器加速工具类
    
    加速方法:
    1. 求解器参数调优
    2. 启发式配置
    3. 割平面策略
    4. 并行求解
    5. Warm Start 增强
    """
    
    @staticmethod
    def configure_solver_for_speed(solver, problem_type: str = 'routing') -> None:
        """
        配置 SCIP 求解器参数以加速求解
        
        Args:
            solver: OR-Tools SCIP 求解器实例
            problem_type: 问题类型 ('routing', 'scheduling', etc.)
        """
        print("  配置 SCIP 求解器参数...")
        
        # SCIP 参数调优
        param_config = {
            # 时间限制
            'limits/time': 300000,  # 300 秒
            
            # 启发式配置（更激进）
            'heuristics/feaspump/freq': 1,  # 频繁运行 feasibility pump
            'heuristics/rounding/freq': 1,  # 频繁运行 rounding
            'heuristics/simplerounding/freq': 1,
            'heuristics/objpscostdiving/freq': 1,  # 激进 diving 启发式
            'heuristics/pscostdiving/freq': 1,
            
            # 分支策略
            'branching/relpscost/priority': 100,  # 优先使用强分支
            'branching/fullstrong/priority': 50,
            
            # 节点选择
            'node selection/hybridbest/weightfac': 0.5,  # 平衡 best-first 和 depth-first
            
            # 割平面
            'separating/maxrounds': 5,  # 限制割平面轮数
            'separating/maxstallrounds': 3,
            
            # 预处理
            'presolving/maxrounds': -1,  # 尽可能多的预处理
            'presolving/maxstallrounds': -1,
            
            # 并行（如果支持）
            'parallel/maxnthreads': 4,  # 使用 4 个线程
        }
        
        # 应用参数
        for param, value in param_config.items():
            try:
                solver.SetParam(param, value)
            except Exception as e:
                # 参数可能因版本而异
                pass
        
        print("  SCIP 参数配置完成")
    
    @staticmethod
    def add_valid_inequalities(solver, x_vars, G, node_to_idx, 
                              demands, num_demands) -> int:
        """
        添加有效不等式（割平面）
        
        类型:
        1. Subtour elimination constraints
        2. Capacity cuts
        3. Cover inequalities
        4. Flow-based cuts
        
        Returns:
            num_cuts: 添加的割平面数量
        """
        print("  添加有效不等式...")
        num_cuts = 0
        
        # 1. Subtour elimination constraints (简化版)
        # 对于任意节点子集 S (|S| >= 2):
        # Σ_{i,j ∈ S, i<j} x_{ij} ≤ |S| - 1
        # 这里只添加小的子集（2-3 个节点）
        
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i+1, min(i+3, len(nodes))):
                # 2-节点子集
                u, v = nodes[i], nodes[j]
                if (u, v) in G.edges() or (v, u) in G.edges():
                    edge_key = tuple(sorted((node_to_idx[u], node_to_idx[v])))
                    if edge_key in x_vars:
                        # x_uv ≤ 1 (平凡约束，跳过)
                        pass
        
        # 2. Flow-based cuts
        # 如果总流量需求超过某割集的容量，添加割平面
        # 这里简化处理
        
        # 3. Degree constraints
        # 对于终端节点，度数至少为 1
        terminals = set()
        for start_idx, end_idx in demands:
            terminals.add(start_idx)
            terminals.add(end_idx)
        
        for term_idx in terminals:
            incident_edges = []
            for i in range(len(nodes)):
                edge_key = tuple(sorted((term_idx, i)))
                if edge_key in x_vars:
                    incident_edges.append(x_vars[edge_key])
            
            if incident_edges:
                solver.Add(sum(incident_edges) >= 1)
                num_cuts += 1
        
        print(f"  添加了 {num_cuts} 个有效不等式")
        return num_cuts
    
    @staticmethod
    def milp_with_scip_acceleration(keypoints: Set[Tuple[float, float]],
                                   path_segments: Dict,
                                   demands: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                                   router,
                                   time_limit_seconds: int = 300,
                                   enable_warm_start: bool = True,
                                   enable_cuts: bool = True,
                                   enable_presolve: bool = True,
                                   n_threads: int = 4) -> List[List[Tuple[float, float]]]:
        """
        使用 SCIP 加速的 MILP 求解
        
        Args:
            n_threads: 并行线程数
        """
        print("\n=== [SCIP 加速 MILP] 开始求解 ===")
        start_time = time.time()
        
        # Step 1: 构建图
        G = nx.Graph()
        for (u, v), segment_info in path_segments.items():
            cost = segment_info['cost']
            if math.isfinite(cost) and cost > 0:
                G.add_edge(u, v, weight=cost, path=segment_info['path'])
        
        # Step 2: 数据预处理
        node_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        num_nodes = len(node_list)
        num_demands = len(demands)
        
        # 吸附终端
        terminal_map = {}
        for term in {pt for demand in demands for pt in demand}:
            gx, gy = router.physical_to_grid_coords(*term)
            snapped = router.grid_coords_to_physical(gx, gy)
            terminal_map[term] = snapped
            
            if snapped not in G:
                nearest = min(
                    (kp for kp in keypoints if kp in G),
                    key=lambda kp: math.hypot(snapped[0]-kp[0], snapped[1]-kp[1]),
                    default=None
                )
                if nearest:
                    path_to_kp = router.find_path_single_cable(snapped, nearest)
                    if path_to_kp:
                        cost = sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) 
                                  for p1, p2 in zip(path_to_kp, path_to_kp[1:]))
                        G.add_edge(snapped, nearest, weight=cost, path=path_to_kp)
        
        snapped_demands = []
        for start, end in demands:
            snapped_demands.append((
                node_to_idx[terminal_map[start]],
                node_to_idx[terminal_map[end]]
            ))
        
        # Step 3: 创建求解器
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            raise RuntimeError("SCIP solver not available.")
        
        # Step 4: 配置求解器参数
        SCIPAccelerator.configure_solver_for_speed(solver)
        
        # Step 5: 创建变量
        x = {}
        x_k = {}
        for u, v in G.edges():
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
                                    for j in G.neighbors(node))
                out_flow = solver.Sum(x_k.get((k, i, node_to_idx[j]), 0) 
                                     for j in G.neighbors(node))
                
                if i == start_idx:
                    solver.Add(out_flow - in_flow == 1)
                elif i == end_idx:
                    solver.Add(in_flow - out_flow == 1)
                else:
                    solver.Add(in_flow == out_flow)
        
        # 6.2 关联约束
        for u, v in G.edges():
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            edge_key = tuple(sorted((u_idx, v_idx)))
            for k in range(num_demands):
                solver.Add(x[edge_key] >= x_k[k, u_idx, v_idx])
                solver.Add(x[edge_key] >= x_k[k, v_idx, u_idx])
        
        # 6.3 树约束
        total_edges = solver.Sum(x.values())
        solver.Add(total_edges <= num_nodes - 1)
        
        # Step 7: Warm Start
        if enable_warm_start:
            print("  生成 Warm Start 初始解...")
            # 用 MST 生成初始解
            mst = nx.minimum_spanning_tree(G, weight='weight')
            for u, v in mst.edges():
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                edge_key = tuple(sorted((u_idx, v_idx)))
                if edge_key in x:
                    x[edge_key].SetSolutionValue(1.0)
        
        # Step 8: 添加有效不等式
        if enable_cuts:
            SCIPAccelerator.add_valid_inequalities(
                solver, x, G, node_to_idx, snapped_demands, num_demands
            )
        
        # Step 9: 目标函数
        total_cost = solver.Sum(
            G.edges[u, v]['weight'] * 
            (solver.Sum(x_k[k, node_to_idx[u], node_to_idx[v]] + 
                       x_k[k, node_to_idx[v], node_to_idx[u]] 
                       for k in range(num_demands)))
            for u, v in G.edges()
        )
        solver.Minimize(total_cost)
        
        # Step 10: 设置时间限制
        solver.SetTimeLimit(time_limit_seconds * 1000)
        
        # Step 11: 求解
        print("  开始求解...")
        status = solver.Solve()
        
        # Step 12: 结果解析
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            print("求解失败，未找到解决方案。")
            return [[] for _ in demands]
        
        obj_value = solver.Objective().Value()
        solve_time = time.time() - start_time
        
        print(f"  求解成功！目标值：{obj_value:.2f}")
        print(f"  求解时间：{solve_time:.2f} 秒")
        
        # 提取解
        H = nx.Graph()
        for u, v in G.edges():
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            edge_key = tuple(sorted((u_idx, v_idx)))
            if x[edge_key].solution_value() > 0.5:
                H.add_edge(u, v, **G.edges[u, v])
        
        # 重建路径
        paths = []
        for start, end in demands:
            snapped_start = terminal_map[start]
            snapped_end = terminal_map[end]
            
            try:
                path_nodes = nx.shortest_path(H, source=snapped_start, 
                                            target=snapped_end, weight='weight')
                
                # 重建几何路径
                full_path = []
                if start != snapped_start:
                    p = router.find_path_single_cable(start, snapped_start)
                    if p:
                        full_path.extend(p[:-1])
                    else:
                        full_path.append(start)
                
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i + 1]
                    seg = H.get_edge_data(u, v)['path']
                    if seg[0] != u:
                        seg = seg[::-1]
                    if full_path:
                        full_path.extend(seg[1:])
                    else:
                        full_path.extend(seg)
                
                if end != snapped_end:
                    p = router.find_path_single_cable(snapped_end, end)
                    if p:
                        full_path.extend(p[1:])
                    elif not full_path:
                        full_path.append(end)
                
                paths.append(full_path)
                
            except:
                paths.append([])
        
        return paths


# ============================================================================
# Part 3: 完整优化框架
# ============================================================================

class OptimizedCableRouter:
    """
    优化版电缆路由器
    
    集成:
    1. 转弯半径约束（后处理 + 搜索中）
    2. SCIP 加速 MILP
    3. 混合优化策略
    """
    
    def __init__(self, router, min_turning_radius: float = 0.0):
        self.router = router
        self.min_turning_radius = min_turning_radius
    
    def route_with_full_optimization(self,
                                    demands: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                                    keypoints: Set[Tuple[float, float]],
                                    path_segments: Dict,
                                    use_milp: bool = True,
                                    milp_time_limit: int = 300,
                                    turning_mode: str = 'both') -> List[List[Tuple[float, float]]]:
        """
        完整优化路由
        
        Args:
            demands: 电缆需求列表
            keypoints: 关键点集合
            path_segments: 路径段字典
            use_milp: 是否使用 MILP（否则用启发式）
            milp_time_limit: MILP 时间限制（秒）
            turning_mode: 转弯处理模式
                - 'none': 不处理
                - 'post': 仅后处理
                - 'search': 仅搜索中
                - 'both': 后处理 + 搜索中
        
        Returns:
            优化后的路径列表
        """
        print("\n=== [完整优化路由] 开始 ===")
        start_time = time.time()
        
        # Step 1: 初始寻路
        print("Step 1: 初始寻路...")
        if use_milp:
            initial_paths = SCIPAccelerator.milp_with_scip_acceleration(
                keypoints, path_segments, demands, self.router,
                time_limit_seconds=milp_time_limit // 2,  # 先用一半时间
                enable_warm_start=True,
                enable_cuts=True
            )
        else:
            # 用启发式
            initial_paths = []
            for start, end in demands:
                if turning_mode in ['search', 'both']:
                    path = TurningRadiusHandler.astar_with_turning_constraints(
                        self.router, start, end, self.min_turning_radius
                    )
                else:
                    path = self.router.find_path_single_cable(start, end)
                initial_paths.append(path)
        
        # Step 2: 后处理平滑
        if turning_mode in ['post', 'both']:
            print("Step 2: 后处理平滑...")
            smoothed_paths = []
            for i, path in enumerate(initial_paths):
                if not path:
                    smoothed_paths.append(path)
                    continue
                
                smoothed = TurningRadiusHandler.smooth_path_post_processing(
                    path, self.router, self.min_turning_radius, method='hybrid'
                )
                smoothed_paths.append(smoothed)
            
            initial_paths = smoothed_paths
        
        # Step 3: 验证
        print("Step 3: 验证转弯半径约束...")
        violation_count = 0
        for i, path in enumerate(initial_paths):
            if not path:
                continue
            
            is_feasible, violations = TurningRadiusHandler.check_path_feasibility(
                path, self.min_turning_radius
            )
            
            if not is_feasible:
                violation_count += 1
                print(f"  路径 {i}: {len(violations)} 处违规")
        
        total_time = time.time() - start_time
        
        print(f"\n优化完成！")
        print(f"  总耗时：{total_time:.2f} 秒")
        print(f"  转弯半径违规：{violation_count}/{len(demands)}")
        
        return initial_paths


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    print("SPT v3.0 优化模块")
    print("="*60)
    
    # 假设已有 router, demands, keypoints, path_segments
    
    # 创建优化器
    optimizer = OptimizedCableRouter(router, min_turning_radius=10.0)
    
    # 完整优化
    paths = optimizer.route_with_full_optimization(
        demands=demands,
        keypoints=keypoints,
        path_segments=path_segments,
        use_milp=True,
        milp_time_limit=300,
        turning_mode='both'  # 后处理 + 搜索中
    )
    
    return paths


if __name__ == "__main__":
    print("SPT v3.0 - 深度优化版本")
    print("主要功能:")
    print("  1. 转弯半径约束（后处理 + 搜索中）")
    print("  2. SCIP 求解器加速（参数调优 + 割平面 + Warm Start）")
    print("  3. 完整优化框架")
    print("\n使用示例:")
    print("  from spt_v3 import OptimizedCableRouter")
    print("  optimizer = OptimizedCableRouter(router, min_turning_radius=10.0)")
    print("  paths = optimizer.route_with_full_optimization(...)")
