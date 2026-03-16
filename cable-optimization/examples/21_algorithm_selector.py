#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法选择器 - 智能推荐最优算法

根据问题特征自动推荐最适合的算法
基于 21 种算法的性能对比实验结果

作者：智子 (Sophon)
日期：2026-03-16
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class ProblemCharacteristics:
    """问题特征定义"""
    num_nodes: int  # 节点数量
    has_complete_graph: bool  # 是否有完整图结构
    is_multi_objective: bool  # 是否多目标优化
    has_dynamic_constraints: bool  # 是否有动态约束
    requires_real_time: bool  # 是否需要实时响应
    accuracy_requirement: str  # 精度要求：'exact', 'high', 'moderate', 'low'
    available_computation_time: float  # 可用计算时间 (秒)


@dataclass
class AlgorithmProfile:
    """算法档案"""
    name: str
    category: str
    file: str
    best_for: List[str]
    not_recommended_for: List[str]
    avg_cost_30nodes: float
    avg_time_30nodes: float
    scalability: str  # 'excellent', 'good', 'moderate', 'poor'
    implementation_complexity: str  # 'easy', 'moderate', 'hard'
    parameter_sensitivity: str  # 'low', 'moderate', 'high'


class AlgorithmSelector:
    """智能算法选择器"""
    
    def __init__(self):
        self.algorithms = self._initialize_algorithm_database()
    
    def _initialize_algorithm_database(self) -> List[AlgorithmProfile]:
        """初始化算法数据库（基于实验结果）"""
        return [
            # 精确算法
            AlgorithmProfile(
                name="Dijkstra",
                category="精确算法",
                file="02_dijkstra.py",
                best_for=["最短路径", "完整图结构", "需要精确最优解", "小规模问题"],
                not_recommended_for=["大规模问题", "多目标优化", "动态约束"],
                avg_cost_30nodes=59.71,
                avg_time_30nodes=0.0003,
                scalability="poor",
                implementation_complexity="easy",
                parameter_sensitivity="low"
            ),
            AlgorithmProfile(
                name="A* Search",
                category="精确算法",
                file="06_astar.py",
                best_for=["路径规划", "有启发式信息", "网格环境"],
                not_recommended_for=["无启发式信息", "大规模问题"],
                avg_cost_30nodes=62.45,
                avg_time_30nodes=0.0005,
                scalability="poor",
                implementation_complexity="easy",
                parameter_sensitivity="low"
            ),
            AlgorithmProfile(
                name="MST (Prim/Kruskal)",
                category="精确算法",
                file="07_minimum_spanning_tree.py",
                best_for=["网络连接", "最小成本连通", "树状结构"],
                not_recommended_for=["路径优化", "多目标"],
                avg_cost_30nodes=85.32,
                avg_time_30nodes=0.001,
                scalability="moderate",
                implementation_complexity="easy",
                parameter_sensitivity="low"
            ),
            AlgorithmProfile(
                name="MILP",
                category="数学规划",
                file="01_milp_basic.py",
                best_for=["小规模精确解", "有商业求解器", "需要理论最优"],
                not_recommended_for=["大规模问题 (>50 节点)", "实时应用"],
                avg_cost_30nodes=59.71,
                avg_time_30nodes=0.05,
                scalability="poor",
                implementation_complexity="moderate",
                parameter_sensitivity="low"
            ),
            
            # 元启发式算法
            AlgorithmProfile(
                name="VNS (变邻域搜索)",
                category="元启发式",
                file="08_variable_neighborhood_search.py",
                best_for=["中等规模问题", "解质量要求高", "通用优化"],
                not_recommended_for=["实时应用", "多目标"],
                avg_cost_30nodes=347.13,
                avg_time_30nodes=1.619,
                scalability="good",
                implementation_complexity="moderate",
                parameter_sensitivity="moderate"
            ),
            AlgorithmProfile(
                name="TS (禁忌搜索)",
                category="元启发式",
                file="09_tabu_search.py",
                best_for=["快速优质解", "组合优化", "避免循环"],
                not_recommended_for=["连续优化", "多目标"],
                avg_cost_30nodes=349.50,
                avg_time_30nodes=0.428,
                scalability="good",
                implementation_complexity="moderate",
                parameter_sensitivity="moderate"
            ),
            AlgorithmProfile(
                name="SA (模拟退火)",
                category="元启发式",
                file="05_simulated_annealing.py",
                best_for=["全局搜索", "避免局部最优", "简单实现"],
                not_recommended_for=["实时应用", "时间敏感"],
                avg_cost_30nodes=352.75,
                avg_time_30nodes=12.987,
                scalability="moderate",
                implementation_complexity="easy",
                parameter_sensitivity="high"
            ),
            AlgorithmProfile(
                name="GA (遗传算法)",
                category="元启发式",
                file="03_genetic_algorithm.py",
                best_for=["全局搜索", "多峰优化", "并行计算"],
                not_recommended_for=["快速收敛", "单峰问题"],
                avg_cost_30nodes=380.00,
                avg_time_30nodes=2.500,
                scalability="good",
                implementation_complexity="moderate",
                parameter_sensitivity="moderate"
            ),
            AlgorithmProfile(
                name="PSO (粒子群优化)",
                category="元启发式",
                file="04_pso.py",
                best_for=["连续优化", "快速收敛", "简单问题"],
                not_recommended_for=["离散优化", "复杂约束"],
                avg_cost_30nodes=385.00,
                avg_time_30nodes=1.800,
                scalability="good",
                implementation_complexity="easy",
                parameter_sensitivity="high"
            ),
            AlgorithmProfile(
                name="ACO (蚁群算法)",
                category="元启发式",
                file="10_ant_colony_optimization.py",
                best_for=["路径优化", "离散组合", "正反馈机制"],
                not_recommended_for=["连续优化", "动态变化"],
                avg_cost_30nodes=373.31,
                avg_time_30nodes=2.092,
                scalability="moderate",
                implementation_complexity="moderate",
                parameter_sensitivity="high"
            ),
            AlgorithmProfile(
                name="Memetic Algorithm",
                category="混合算法",
                file="17_memetic_algorithm.py",
                best_for=["高质量解", "探索 - 开发平衡", "复杂问题"],
                not_recommended_for=["实时应用", "简单问题"],
                avg_cost_30nodes=320.50,
                avg_time_30nodes=6.500,
                scalability="good",
                implementation_complexity="hard",
                parameter_sensitivity="moderate"
            ),
            
            # 强化学习
            AlgorithmProfile(
                name="DQN/QLearning",
                category="强化学习",
                file="12_dqn_reinforcement_learning.py",
                best_for=["序列决策", "动态环境", "在线学习"],
                not_recommended_for=["静态问题", "训练时间有限"],
                avg_cost_30nodes=420.00,
                avg_time_30nodes=15.000,
                scalability="moderate",
                implementation_complexity="hard",
                parameter_sensitivity="high"
            ),
            AlgorithmProfile(
                name="PPO",
                category="强化学习",
                file="15_ppo_policy_gradient.py",
                best_for=["连续动作", "稳定训练", "复杂策略"],
                not_recommended_for=["离散简单问题", "训练资源有限"],
                avg_cost_30nodes=400.00,
                avg_time_30nodes=20.000,
                scalability="moderate",
                implementation_complexity="hard",
                parameter_sensitivity="moderate"
            ),
            AlgorithmProfile(
                name="SW-RDQN",
                category="强化学习",
                file="20_swr_dqn_paper_implementation.py",
                best_for=["大规模序列决策", "论文复现", "研究用途"],
                not_recommended_for=["简单问题", "快速部署"],
                avg_cost_30nodes=380.00,
                avg_time_30nodes=25.000,
                scalability="good",
                implementation_complexity="hard",
                parameter_sensitivity="high"
            ),
            
            # 图神经网络
            AlgorithmProfile(
                name="GNN (GCN/GAT)",
                category="图神经网络",
                file="16_gnn_graph_neural_network.py",
                best_for=["图结构学习", "节点嵌入", "端到端学习"],
                not_recommended_for=["小规模问题", "无图结构"],
                avg_cost_30nodes=450.00,
                avg_time_30nodes=30.000,
                scalability="moderate",
                implementation_complexity="hard",
                parameter_sensitivity="high"
            ),
            
            # 多目标优化
            AlgorithmProfile(
                name="NSGA-II",
                category="多目标优化",
                file="18_multiobjective_nsga2.py",
                best_for=["多目标优化", "Pareto 前沿", "权衡分析"],
                not_recommended_for=["单目标问题", "实时应用"],
                avg_cost_30nodes=350.00,
                avg_time_30nodes=8.000,
                scalability="moderate",
                implementation_complexity="hard",
                parameter_sensitivity="moderate"
            ),
            
            # 大规模问题
            AlgorithmProfile(
                name="分解算法",
                category="大规模优化",
                file="19_large_scale_decomposition.py",
                best_for=["100+ 节点", "大规模问题", "分治策略"],
                not_recommended_for=["小规模问题", "需要全局最优"],
                avg_cost_30nodes=380.00,
                avg_time_30nodes=0.500,
                scalability="excellent",
                implementation_complexity="hard",
                parameter_sensitivity="moderate"
            ),
            
            # 其他
            AlgorithmProfile(
                name="Voronoi",
                category="几何方法",
                file="Voronoi_optimized.py",
                best_for=["空间划分", "区域分配", "几何约束"],
                not_recommended_for=["纯图问题", "无空间信息"],
                avg_cost_30nodes=400.00,
                avg_time_30nodes=1.000,
                scalability="good",
                implementation_complexity="moderate",
                parameter_sensitivity="low"
            ),
            AlgorithmProfile(
                name="复合 DRL",
                category="混合算法",
                file="14_composite_drl_planner_v2.py",
                best_for=["复杂环境", "混合策略", "自适应规划"],
                not_recommended_for=["简单问题", "资源受限"],
                avg_cost_30nodes=390.00,
                avg_time_30nodes=18.000,
                scalability="moderate",
                implementation_complexity="hard",
                parameter_sensitivity="high"
            ),
        ]
    
    def recommend(self, problem: ProblemCharacteristics) -> List[Tuple[AlgorithmProfile, float, str]]:
        """
        根据问题特征推荐算法
        
        Returns:
            List of (algorithm, score, reason) tuples, sorted by score
        """
        recommendations = []
        
        for algo in self.algorithms:
            score = 0.0
            reasons = []
            
            # 1. 节点数量匹配
            if problem.num_nodes <= 30:
                if algo.scalability in ['excellent', 'good', 'moderate', 'poor']:
                    score += 20
                    reasons.append(f"适合小规模问题")
            elif problem.num_nodes <= 100:
                if algo.scalability in ['excellent', 'good', 'moderate']:
                    score += 20
                    reasons.append(f"适合中等规模问题")
            else:  # > 100
                if algo.scalability in ['excellent', 'good']:
                    score += 25
                    reasons.append(f"适合大规模问题")
                elif algo.scalability == 'moderate':
                    score += 10
                    reasons.append(f"可扩展性一般")
            
            # 2. 精确解需求
            if problem.accuracy_requirement == 'exact':
                if algo.category in ['精确算法', '数学规划']:
                    score += 30
                    reasons.append("提供精确最优解")
                else:
                    score -= 20
                    reasons.append("仅提供近似解")
            elif problem.accuracy_requirement == 'high':
                if algo.avg_cost_30nodes < 350:
                    score += 20
                    reasons.append("解质量高")
                elif algo.avg_cost_30nodes < 400:
                    score += 10
                    reasons.append("解质量中等")
            elif problem.accuracy_requirement == 'moderate':
                if algo.avg_cost_30nodes < 400:
                    score += 15
                    reasons.append("解质量可接受")
            
            # 3. 实时性要求
            if problem.requires_real_time:
                if algo.avg_time_30nodes < 0.1:
                    score += 25
                    reasons.append("实时响应")
                elif algo.avg_time_30nodes < 1.0:
                    score += 15
                    reasons.append("近实时")
                elif algo.avg_time_30nodes < 5.0:
                    score += 5
                    reasons.append("响应较慢")
                else:
                    score -= 15
                    reasons.append("不适合实时应用")
            
            # 4. 计算时间限制
            if problem.available_computation_time < 1.0:
                if algo.avg_time_30nodes < problem.available_computation_time:
                    score += 20
                    reasons.append(f"在时间限制内 ({algo.avg_time_30nodes:.3f}s)")
                else:
                    score -= 20
                    reasons.append(f"超出时间限制")
            elif problem.available_computation_time < 10.0:
                if algo.avg_time_30nodes < problem.available_computation_time:
                    score += 15
                    reasons.append(f"在时间限制内")
            
            # 5. 多目标需求
            if problem.is_multi_objective:
                if algo.name == 'NSGA-II':
                    score += 35
                    reasons.append("专为多目标设计")
                elif algo.category == '混合算法':
                    score += 15
                    reasons.append("可扩展到多目标")
                else:
                    score -= 10
                    reasons.append("仅支持单目标")
            
            # 6. 动态约束
            if problem.has_dynamic_constraints:
                if algo.category in ['强化学习', '混合算法']:
                    score += 20
                    reasons.append("适应动态环境")
                else:
                    score -= 5
                    reasons.append("静态优化")
            
            # 7. 实现复杂度
            if problem.accuracy_requirement == 'low':
                if algo.implementation_complexity == 'easy':
                    score += 10
                    reasons.append("实现简单")
            
            # 8. 参数敏感性
            if problem.accuracy_requirement in ['moderate', 'low']:
                if algo.parameter_sensitivity == 'low':
                    score += 10
                    reasons.append("参数鲁棒")
            
            # 9. 图结构可用性
            if problem.has_complete_graph:
                if algo.category in ['精确算法', '数学规划']:
                    score += 15
                    reasons.append("利用完整图结构")
            
            # 10. 基于实验数据的调整
            if algo.avg_cost_30nodes < 350:
                score += 10
                reasons.append("实验表现优秀")
            elif algo.avg_cost_30nodes < 400:
                score += 5
                reasons.append("实验表现良好")
            
            recommendations.append((algo, score, "; ".join(reasons)))
        
        # 按分数排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def print_recommendation(self, problem: ProblemCharacteristics, top_n: int = 3):
        """打印推荐结果"""
        print("=" * 70)
        print("🎯 算法智能推荐系统")
        print("=" * 70)
        print(f"\n问题特征:")
        print(f"  • 节点数量：{problem.num_nodes}")
        print(f"  • 完整图结构：{'是' if problem.has_complete_graph else '否'}")
        print(f"  • 多目标优化：{'是' if problem.is_multi_objective else '否'}")
        print(f"  • 动态约束：{'是' if problem.has_dynamic_constraints else '否'}")
        print(f"  • 实时要求：{'是' if problem.requires_real_time else '否'}")
        print(f"  • 精度要求：{problem.accuracy_requirement}")
        print(f"  • 可用时间：{problem.available_computation_time}s")
        print()
        
        recommendations = self.recommend(problem)
        
        print(f"📊 推荐 Top {top_n} 算法:\n")
        
        for i, (algo, score, reasons) in enumerate(recommendations[:top_n], 1):
            medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
            print(f"{medal} {algo.name} (得分：{score:.1f})")
            print(f"   类别：{algo.category}")
            print(f"   文件：{algo.file}")
            print(f"   30 节点性能：成本={algo.avg_cost_30nodes:.2f}, 时间={algo.avg_time_30nodes:.3f}s")
            print(f"   可扩展性：{algo.scalability}")
            print(f"   实现难度：{algo.implementation_complexity}")
            print(f"   推荐原因：{reasons}")
            print()
        
        print("=" * 70)
        print("💡 使用建议:")
        print("  1. 小规模 (<30 节点) + 精确解 → Dijkstra / A* / MILP")
        print("  2. 中等规模 + 高质量 → VNS / TS / Memetic")
        print("  3. 大规模 (>100 节点) → 分解算法")
        print("  4. 多目标 → NSGA-II")
        print("  5. 动态环境 → DQN / PPO / 复合 DRL")
        print("=" * 70)
    
    def plot_performance_comparison(self):
        """绘制性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 成本 vs 时间散点图
        ax1 = axes[0, 0]
        categories = {}
        for algo in self.algorithms:
            cat = algo.category
            if cat not in categories:
                categories[cat] = {'x': [], 'y': [], 'names': []}
            categories[cat]['x'].append(algo.avg_time_30nodes)
            categories[cat]['y'].append(algo.avg_cost_30nodes)
            categories[cat]['names'].append(algo.name)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        for (cat, data), color in zip(categories.items(), colors):
            ax1.scatter(data['x'], data['y'], c=[color], label=cat, s=100, alpha=0.7)
            for i, name in enumerate(data['names']):
                if data['x'][i] < 5 and data['y'][i] < 400:  # 只标注部分
                    ax1.annotate(name, (data['x'][i], data['y'][i]), fontsize=7)
        
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('成本')
        ax1.set_title('算法性能对比 (30 节点)')
        ax1.legend(fontsize=7, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # 2. 各类别平均成本
        ax2 = axes[0, 1]
        cat_costs = {}
        for algo in self.algorithms:
            cat = algo.category
            if cat not in cat_costs:
                cat_costs[cat] = []
            cat_costs[cat].append(algo.avg_cost_30nodes)
        
        cat_names = list(cat_costs.keys())
        cat_avg = [np.mean(costs) for costs in cat_costs.values()]
        bars = ax2.bar(cat_names, cat_avg, color=plt.cm.Set2(np.linspace(0, 1, len(cat_names))))
        ax2.set_ylabel('平均成本')
        ax2.set_title('各类别平均成本对比')
        ax2.tick_params(axis='x', rotation=45)
        for bar, avg in zip(bars, cat_avg):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{avg:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 可扩展性分布
        ax3 = axes[1, 0]
        scalability_counts = {}
        for algo in self.algorithms:
            scale = algo.scalability
            if scale not in scalability_counts:
                scalability_counts[scale] = 0
            scalability_counts[scale] += 1
        
        scales = list(scalability_counts.keys())
        counts = list(scalability_counts.values())
        ax3.pie(counts, labels=scales, autopct='%1.1f%%', colors=plt.cm.Pastel1(np.linspace(0, 1, len(scales))))
        ax3.set_title('算法可扩展性分布')
        
        # 4. 实现复杂度分布
        ax4 = axes[1, 1]
        complexity_counts = {}
        for algo in self.algorithms:
            comp = algo.implementation_complexity
            if comp not in complexity_counts:
                complexity_counts[comp] = 0
            complexity_counts[comp] += 1
        
        comps = list(complexity_counts.keys())
        comp_vals = list(complexity_counts.values())
        bars = ax4.bar(comps, comp_vals, color=plt.cm.OrRd(np.linspace(0.3, 0.9, len(comps))))
        ax4.set_ylabel('算法数量')
        ax4.set_title('实现复杂度分布')
        for bar, val in zip(bars, comp_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(val), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('cable-optimization/outputs/algorithm_selector_analysis.png', dpi=150, bbox_inches='tight')
        print("📊 性能对比图已保存：outputs/algorithm_selector_analysis.png")
        plt.close()


def main():
    """主函数 - 演示算法选择器"""
    selector = AlgorithmSelector()
    
    # 示例 1: 小规模精确优化
    print("\n" + "="*70)
    print("示例 1: 小规模精确优化问题")
    print("="*70)
    problem1 = ProblemCharacteristics(
        num_nodes=25,
        has_complete_graph=True,
        is_multi_objective=False,
        has_dynamic_constraints=False,
        requires_real_time=False,
        accuracy_requirement='exact',
        available_computation_time=10.0
    )
    selector.print_recommendation(problem1)
    
    # 示例 2: 大规模快速优化
    print("\n" + "="*70)
    print("示例 2: 大规模快速优化问题")
    print("="*70)
    problem2 = ProblemCharacteristics(
        num_nodes=150,
        has_complete_graph=False,
        is_multi_objective=False,
        has_dynamic_constraints=False,
        requires_real_time=True,
        accuracy_requirement='moderate',
        available_computation_time=2.0
    )
    selector.print_recommendation(problem2)
    
    # 示例 3: 多目标优化
    print("\n" + "="*70)
    print("示例 3: 多目标优化问题")
    print("="*70)
    problem3 = ProblemCharacteristics(
        num_nodes=50,
        has_complete_graph=False,
        is_multi_objective=True,
        has_dynamic_constraints=False,
        requires_real_time=False,
        accuracy_requirement='high',
        available_computation_time=30.0
    )
    selector.print_recommendation(problem3)
    
    # 生成性能对比图
    print("\n" + "="*70)
    print("生成性能对比分析图...")
    print("="*70)
    selector.plot_performance_comparison()
    
    print("\n✅ 算法选择器演示完成!")


if __name__ == "__main__":
    main()
