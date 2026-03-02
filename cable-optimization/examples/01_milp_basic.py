"""
MILP 基础示例 - 简单网络布线优化

问题描述:
给定一个图 G=(V,E)，每条边有成本 c_e 和容量 cap_e
需要从源点 s 到汇点 t 运送流量 F
目标：最小化总布线成本

数学模型:
min Σ_e c_e * x_e
s.t.
    Σ_{e∈out(i)} x_e - Σ_{e∈in(i)} x_e = b_i  ∀i∈V  (流量守恒)
    0 ≤ x_e ≤ cap_e * y_e                        ∀e∈E  (容量约束)
    y_e ∈ {0,1}                                   ∀e∈E  (是否使用边 e)

其中 b_i = F (如果 i=s), -F (如果 i=t), 0 (其他节点)
"""

import pulp as pl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def create_example_graph():
    """创建一个简单的示例图"""
    G = nx.DiGraph()
    
    # 添加节点
    nodes = ['s', 'A', 'B', 'C', 'D', 't']
    G.add_nodes_from(nodes)
    
    # 添加边 (起点，终点，成本，容量)
    edges = [
        ('s', 'A', 10, 100),
        ('s', 'B', 15, 100),
        ('A', 'B', 5, 50),
        ('A', 'C', 20, 100),
        ('B', 'C', 10, 100),
        ('B', 'D', 15, 100),
        ('C', 'D', 5, 50),
        ('C', 't', 10, 100),
        ('D', 't', 10, 100),
    ]
    
    for u, v, cost, cap in edges:
        G.add_edge(u, v, cost=cost, capacity=cap)
    
    return G


def solve_milp_routing(G, source, sink, flow_demand):
    """
    使用 MILP 求解最优布线路径
    
    参数:
        G: networkx 图
        source: 源点
        sink: 汇点
        flow_demand: 需要运送的流量
    
    返回:
        optimal_cost: 最优成本
        flow_dict: 每条边的流量
        selected_edges: 选中的边
    """
    
    # 创建 MILP 问题
    prob = pl.LpProblem("Cable_Routing_Optimization", pl.LpMinimize)
    
    # 决策变量
    # x_e: 边 e 上的流量 (连续变量)
    # y_e: 是否使用边 e (0-1 变量)
    x = {}
    y = {}
    
    for u, v, data in G.edges(data=True):
        edge_name = f"x_{u}_{v}"
        x[(u, v)] = pl.LpVariable(edge_name, lowBound=0, upBound=data['capacity'])
        
        edge_name_y = f"y_{u}_{v}"
        y[(u, v)] = pl.LpVariable(edge_name_y, cat='Binary')
    
    # 目标函数：最小化总成本
    prob += pl.lpSum([G[u][v]['cost'] * y[(u, v)] for u, v in G.edges()]), "Total_Cost"
    
    # 约束 1: 流量守恒
    for node in G.nodes():
        inflow = pl.lpSum([x[(u, node)] for u in G.predecessors(node)])
        outflow = pl.lpSum([x[(node, v)] for v in G.successors(node)])
        
        if node == source:
            prob += (outflow - inflow == flow_demand), f"Flow_Conservation_{node}"
        elif node == sink:
            prob += (inflow - outflow == flow_demand), f"Flow_Conservation_{node}"
        else:
            prob += (inflow == outflow), f"Flow_Conservation_{node}"
    
    # 约束 2: 容量约束和边选择
    for u, v, data in G.edges(data=True):
        prob += (x[(u, v)] <= data['capacity'] * y[(u, v)]), f"Capacity_{u}_{v}"
    
    # 求解
    prob.solve(pl.PULP_CBC_CMD(msg=0))
    
    # 提取结果
    optimal_cost = pl.value(prob.objective)
    flow_dict = {(u, v): pl.value(x[(u, v)]) for u, v in G.edges()}
    selected_edges = [(u, v) for u, v in G.edges() if pl.value(y[(u, v)]) > 0.5]
    
    return optimal_cost, flow_dict, selected_edges, prob.status


def visualize_solution(G, flow_dict, selected_edges, source, sink):
    """可视化布线路径"""
    pos = nx.spring_layout(G, seed=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：原始图
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=12)
    
    # 绘制所有边
    edge_weights = [G[u][v]['cost'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', 
                          width=1, style='dashed', arrows=True)
    
    # 标注边的成本和容量
    edge_labels = {f"{u}->{v}": f"${G[u][v]['cost']}\n(Cap:{G[u][v]['capacity']})" 
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1, 
                                font_size=8, label_pos=0.3)
    
    ax1.set_title("原始网络图", fontsize=14)
    ax1.axis('off')
    
    # 右图：最优解
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='lightgreen', node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=12)
    
    # 绘制选中的边
    if selected_edges:
        nx.draw_networkx_edges(G, pos, edgelist=selected_edges, ax=ax2,
                              edge_color='red', width=2, arrows=True)
        
        # 标注流量
        flow_labels = {f"{u}->{v}": f"Flow:{flow_dict[(u,v)]:.1f}" 
                      for u, v in selected_edges if flow_dict[(u,v)] > 0}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=flow_labels, ax=ax2,
                                    font_size=9, fontcolor='red')
    
    # 标记源点和汇点
    nx.draw_networkx_nodes(G, pos, nodelist=[source], ax=ax2,
                          node_color='red', node_size=600, label='Source')
    nx.draw_networkx_nodes(G, pos, nodelist=[sink], ax=ax2,
                          node_color='blue', node_size=600, label='Sink')
    
    ax2.set_title(f"最优布线路径 (总成本：${sum(G[u][v]['cost'] for u,v in selected_edges):.1f})", 
                 fontsize=14)
    ax2.axis('off')
    ax2.legend()
    
    plt.tight_layout()
    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("MILP 基础示例 - 简单网络布线优化")
    print("=" * 60)
    
    # 创建示例图
    G = create_example_graph()
    print(f"\n✓ 创建示例网络：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")
    
    # 设置源点、汇点和流量需求
    source = 's'
    sink = 't'
    flow_demand = 50
    print(f"✓ 布线任务：从 {source} 到 {sink}, 流量需求 = {flow_demand}")
    
    # 求解 MILP
    print("\n⏳ 正在求解 MILP 模型...")
    optimal_cost, flow_dict, selected_edges, status = solve_milp_routing(
        G, source, sink, flow_demand
    )
    
    # 检查求解状态
    if status == pl.LpStatusOptimal:
        print(f"✓ 找到最优解！")
    elif status == pl.LpStatusInfeasible:
        print("✗ 问题无可行解")
        return
    else:
        print(f"? 求解状态：{pl.LpStatus[status]}")
    
    # 输出结果
    print(f"\n📊 优化结果:")
    print(f"  最优总成本：${optimal_cost:.2f}")
    print(f"  使用的边数：{len(selected_edges)}")
    print(f"\n  选中的边:")
    for u, v in selected_edges:
        flow = flow_dict[(u, v)]
        cost = G[u][v]['cost']
        cap = G[u][v]['capacity']
        print(f"    {u} → {v}: 流量={flow:.1f}, 成本=${cost}, 容量={cap}")
    
    # 可视化
    fig = visualize_solution(G, flow_dict, selected_edges, source, sink)
    
    # 保存图像
    output_path = "/root/.openclaw/workspace/cable-optimization/examples/milp_basic_result.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 结果图已保存到：{output_path}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
