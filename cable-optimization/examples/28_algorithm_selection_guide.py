#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28_algorithm_selection_guide.py - 算法选择指南与最佳实践

作者：智子 (Sophon)
日期：2026-03-26
描述：基于 27 天学习经验，总结线缆布线优化算法选择指南
     包含问题特征分析、算法推荐、参数调优建议

核心内容:
1. 问题特征分类器
2. 算法推荐引擎
3. 参数调优指南
4. 性能预估模型
5. 最佳实践总结
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import time


# ============================================================================
# 问题特征定义
# ============================================================================

class ProblemScale(Enum):
    """问题规模分类"""
    SMALL = "small"       # < 50 节点
    MEDIUM = "medium"     # 50-200 节点
    LARGE = "large"       # 200-1000 节点
    XLARGE = "xlarge"     # > 1000 节点


class ConstraintType(Enum):
    """约束类型"""
    SIMPLE = "simple"           # 仅基础连接约束
    CAPACITY = "capacity"       # 容量约束
    GEOMETRIC = "geometric"     # 几何/空间约束
    TEMPORAL = "temporal"       # 时间相关约束
    MULTI_OBJECTIVE = "multi"   # 多目标优化


class ObjectiveType(Enum):
    """优化目标类型"""
    SINGLE_COST = "single_cost"      # 单目标：最小成本
    SINGLE_LENGTH = "single_length"  # 单目标：最短长度
    MULTI = "multi"                  # 多目标


@dataclass
class ProblemCharacteristics:
    """问题特征描述"""
    num_nodes: int
    num_edges: int
    constraint_types: List[ConstraintType]
    objective_type: ObjectiveType
    has_obstacles: bool = False
    has_dynamic_changes: bool = False
    real_time_required: bool = False
    solution_quality_priority: float = 0.5  # 0=时间优先，1=质量优先
    
    @property
    def scale(self) -> ProblemScale:
        """根据节点数判断规模"""
        if self.num_nodes < 50:
            return ProblemScale.SMALL
        elif self.num_nodes < 200:
            return ProblemScale.MEDIUM
        elif self.num_nodes < 1000:
            return ProblemScale.LARGE
        else:
            return ProblemScale.XLARGE
    
    def to_dict(self) -> Dict:
        return {
            'scale': self.scale.value,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'constraints': [c.value for c in self.constraint_types],
            'objective': self.objective_type.value,
            'has_obstacles': self.has_obstacles,
            'real_time': self.real_time_required,
            'quality_priority': self.solution_quality_priority
        }


# ============================================================================
# 算法知识库
# ============================================================================

@dataclass
class AlgorithmProfile:
    """算法档案"""
    name: str
    category: str
    best_for_scales: List[ProblemScale]
    best_for_constraints: List[ConstraintType]
    avg_runtime_factor: float  # 相对 runtime 系数
    solution_quality: float    # 0-1, 解的质量
    robustness: float          # 0-1, 鲁棒性
    ease_of_implementation: float  # 0-1, 实现难度 (1=容易)
    recommended_params: Dict
    notes: str


ALGORITHM_DATABASE = [
    AlgorithmProfile(
        name="MILP (PuLP/CBC)",
        category="exact",
        best_for_scales=[ProblemScale.SMALL],
        best_for_constraints=[ConstraintType.SIMPLE, ConstraintType.CAPACITY],
        avg_runtime_factor=10.0,
        solution_quality=1.0,
        robustness=0.9,
        ease_of_implementation=0.7,
        recommended_params={'solver': 'CBC', 'time_limit': 300},
        notes="适合小规模精确求解，保证最优解"
    ),
    AlgorithmProfile(
        name="Dijkstra",
        category="graph",
        best_for_scales=[ProblemScale.SMALL, ProblemScale.MEDIUM, ProblemScale.LARGE],
        best_for_constraints=[ConstraintType.SIMPLE],
        avg_runtime_factor=0.1,
        solution_quality=1.0,
        robustness=1.0,
        ease_of_implementation=0.9,
        recommended_params={},
        notes="单源最短路径，非负边权，精确解"
    ),
    AlgorithmProfile(
        name="A* Search",
        category="graph",
        best_for_scales=[ProblemScale.SMALL, ProblemScale.MEDIUM],
        best_for_constraints=[ConstraintType.SIMPLE, ConstraintType.GEOMETRIC],
        avg_runtime_factor=0.2,
        solution_quality=1.0,
        robustness=0.9,
        ease_of_implementation=0.8,
        recommended_params={'heuristic': 'euclidean'},
        notes="带启发式的最短路径，需要可采纳启发函数"
    ),
    AlgorithmProfile(
        name="MST (Kruskal/Prim)",
        category="graph",
        best_for_scales=[ProblemScale.SMALL, ProblemScale.MEDIUM, ProblemScale.LARGE],
        best_for_constraints=[ConstraintType.SIMPLE],
        avg_runtime_factor=0.15,
        solution_quality=0.95,
        robustness=1.0,
        ease_of_implementation=0.85,
        recommended_params={},
        notes="最小生成树，连接所有节点的最优方案"
    ),
    AlgorithmProfile(
        name="遗传算法 (GA)",
        category="metaheuristic",
        best_for_scales=[ProblemScale.MEDIUM, ProblemScale.LARGE],
        best_for_constraints=[ConstraintType.SIMPLE, ConstraintType.CAPACITY, ConstraintType.MULTI_OBJECTIVE],
        avg_runtime_factor=2.0,
        solution_quality=0.85,
        robustness=0.8,
        ease_of_implementation=0.6,
        recommended_params={
            'population_size': 100,
            'generations': 500,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1
        },
        notes="通用性强，适合复杂约束和多目标"
    ),
    AlgorithmProfile(
        name="粒子群优化 (PSO)",
        category="metaheuristic",
        best_for_scales=[ProblemScale.MEDIUM, ProblemScale.LARGE],
        best_for_constraints=[ConstraintType.SIMPLE, ConstraintType.GEOMETRIC],
        avg_runtime_factor=1.5,
        solution_quality=0.82,
        robustness=0.75,
        ease_of_implementation=0.7,
        recommended_params={
            'swarm_size': 50,
            'iterations': 200,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        },
        notes="收敛快，参数少，适合连续优化"
    ),
    AlgorithmProfile(
        name="模拟退火 (SA)",
        category="metaheuristic",
        best_for_scales=[ProblemScale.SMALL, ProblemScale.MEDIUM],
        best_for_constraints=[ConstraintType.SIMPLE, ConstraintType.GEOMETRIC],
        avg_runtime_factor=1.8,
        solution_quality=0.80,
        robustness=0.85,
        ease_of_implementation=0.75,
        recommended_params={
            'initial_temp': 1000,
            'cooling_rate': 0.95,
            'min_temp': 1e-3
        },
        notes="简单有效，避免局部最优"
    ),
    AlgorithmProfile(
        name="变邻域搜索 (VNS)",
        category="metaheuristic",
        best_for_scales=[ProblemScale.MEDIUM, ProblemScale.LARGE],
        best_for_constraints=[ConstraintType.SIMPLE, ConstraintType.CAPACITY],
        avg_runtime_factor=1.2,
        solution_quality=0.88,
        robustness=0.9,
        ease_of_implementation=0.5,
        recommended_params={
            'neighborhoods': ['swap', 'insert', 'reverse'],
            'max_iterations': 100
        },
        notes="邻域结构设计关键，微调效果好"
    ),
    AlgorithmProfile(
        name="禁忌搜索 (TS)",
        category="metaheuristic",
        best_for_scales=[ProblemScale.MEDIUM],
        best_for_constraints=[ConstraintType.SIMPLE, ConstraintType.CAPACITY],
        avg_runtime_factor=1.5,
        solution_quality=0.86,
        robustness=0.85,
        ease_of_implementation=0.55,
        recommended_params={
            'tabu_list_size': 20,
            'max_iterations': 500
        },
        notes="记忆机制避免循环，适合组合优化"
    ),
    AlgorithmProfile(
        name="蚁群算法 (ACO)",
        category="metaheuristic",
        best_for_scales=[ProblemScale.MEDIUM, ProblemScale.LARGE],
        best_for_constraints=[ConstraintType.SIMPLE, ConstraintType.GEOMETRIC],
        avg_runtime_factor=3.0,
        solution_quality=0.87,
        robustness=0.8,
        ease_of_implementation=0.5,
        recommended_params={
            'ant_count': 50,
            'iterations': 100,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.5
        },
        notes="正反馈机制，适合路径优化"
    ),
    AlgorithmProfile(
        name="NSGA-II",
        category="multi_objective",
        best_for_scales=[ProblemScale.MEDIUM, ProblemScale.LARGE],
        best_for_constraints=[ConstraintType.MULTI_OBJECTIVE],
        avg_runtime_factor=3.5,
        solution_quality=0.85,
        robustness=0.85,
        ease_of_implementation=0.4,
        recommended_params={
            'population_size': 100,
            'generations': 200,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        },
        notes="多目标优化标准算法，Pareto 前沿"
    ),
    AlgorithmProfile(
        name="DQN (深度强化学习)",
        category="rl",
        best_for_scales=[ProblemScale.MEDIUM, ProblemScale.LARGE],
        best_for_constraints=[ConstraintType.SIMPLE, ConstraintType.TEMPORAL],
        avg_runtime_factor=5.0,  # 含训练时间
        solution_quality=0.80,
        robustness=0.7,
        ease_of_implementation=0.3,
        recommended_params={
            'buffer_size': 10000,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'gamma': 0.99
        },
        notes="适合动态环境，需要大量训练"
    ),
    AlgorithmProfile(
        name="图神经网络 (GNN)",
        category="ml",
        best_for_scales=[ProblemScale.LARGE, ProblemScale.XLARGE],
        best_for_constraints=[ConstraintType.GEOMETRIC, ConstraintType.CAPACITY],
        avg_runtime_factor=8.0,  # 含训练时间
        solution_quality=0.82,
        robustness=0.75,
        ease_of_implementation=0.25,
        recommended_params={
            'hidden_dims': [64, 64],
            'num_layers': 3,
            'learning_rate': 1e-3
        },
        notes="学习图结构特征，适合大规模问题"
    ),
]


# ============================================================================
# 算法推荐引擎
# ============================================================================

class AlgorithmRecommendationEngine:
    """算法推荐引擎"""
    
    def __init__(self):
        self.algorithms = ALGORITHM_DATABASE
    
    def score_algorithm(self, algo: AlgorithmProfile, problem: ProblemCharacteristics) -> float:
        """计算算法匹配度分数"""
        score = 0.0
        
        # 规模匹配 (30 分)
        if problem.scale in algo.best_for_scales:
            score += 30
        else:
            score += 10
        
        # 约束匹配 (30 分)
        constraint_match = sum(1 for c in problem.constraint_types 
                               if c in algo.best_for_constraints)
        score += (constraint_match / max(len(problem.constraint_types), 1)) * 30
        
        # 解的质量偏好 (20 分)
        if problem.solution_quality_priority > 0.7:
            score += algo.solution_quality * 20
        elif problem.solution_quality_priority < 0.3:
            score += (1 - algo.avg_runtime_factor / 10) * 20
        else:
            score += (algo.solution_quality * 0.5 + (1 - algo.avg_runtime_factor / 10) * 0.5) * 20
        
        # 鲁棒性 (10 分)
        score += algo.robustness * 10
        
        # 实时性要求惩罚
        if problem.real_time_required and algo.avg_runtime_factor > 2:
            score -= 20
        
        return score
    
    def recommend(self, problem: ProblemCharacteristics, top_k: int = 3) -> List[Tuple[AlgorithmProfile, float]]:
        """推荐最适合的算法"""
        scored = [(algo, self.score_algorithm(algo, problem)) for algo in self.algorithms]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def generate_report(self, problem: ProblemCharacteristics) -> str:
        """生成算法选择报告"""
        recommendations = self.recommend(problem)
        
        report = []
        report.append("=" * 60)
        report.append("📊 算法选择推荐报告")
        report.append("=" * 60)
        report.append(f"\n问题特征:")
        report.append(f"  • 规模：{problem.scale.value} ({problem.num_nodes} 节点)")
        report.append(f"  • 约束：{', '.join([c.value for c in problem.constraint_types])}")
        report.append(f"  • 目标：{problem.objective_type.value}")
        report.append(f"  • 质量优先级：{problem.solution_quality_priority:.1%}")
        report.append(f"  • 实时要求：{'是' if problem.real_time_required else '否'}")
        report.append("")
        report.append("推荐算法:")
        report.append("-" * 60)
        
        for i, (algo, score) in enumerate(recommendations, 1):
            report.append(f"\n{i}. {algo.name} (匹配度：{score:.1f}/100)")
            report.append(f"   类别：{algo.category}")
            report.append(f"   预计解质量：{algo.solution_quality:.0%}")
            report.append(f"   相对运行时间：{algo.avg_runtime_factor:.1f}x")
            report.append(f"   实现难度：{algo.ease_of_implementation:.0%}")
            report.append(f"   说明：{algo.notes}")
            report.append(f"   推荐参数：{algo.recommended_params}")
        
        report.append("\n" + "=" * 60)
        report.append("💡 最佳实践建议:")
        report.append("=" * 60)
        report.append(self._generate_best_practices(problem))
        
        return "\n".join(report)
    
    def _generate_best_practices(self, problem: ProblemCharacteristics) -> str:
        """生成最佳实践建议"""
        tips = []
        
        if problem.scale == ProblemScale.SMALL:
            tips.append("• 小规模问题优先尝试精确算法 (MILP)，保证最优解")
        elif problem.scale == ProblemScale.XLARGE:
            tips.append("• 超大规模问题考虑分解策略或机器学习方法")
        
        if ConstraintType.MULTI_OBJECTIVE in problem.constraint_types:
            tips.append("• 多目标问题使用 NSGA-II 等 Pareto 优化算法")
        
        if problem.real_time_required:
            tips.append("• 实时应用选择快速算法 (Dijkstra/MST) 或预训练模型")
        
        if problem.solution_quality_priority > 0.8:
            tips.append("• 高质量要求：使用混合算法 (GA+ 局部搜索)")
        
        if not tips:
            tips.append("• 从简单算法开始，逐步尝试更复杂方法")
            tips.append("• 记录每次实验结果，建立性能基准")
        
        return "\n".join(tips)


# ============================================================================
# 参数调优指南
# ============================================================================

class ParameterTuningGuide:
    """参数调优指南"""
    
    @staticmethod
    def get_tuning_strategy(algorithm_name: str) -> Dict:
        """获取算法的参数调优策略"""
        strategies = {
            "遗传算法 (GA)": {
                "关键参数": ["population_size", "crossover_rate", "mutation_rate"],
                "调优顺序": ["population_size → crossover_rate → mutation_rate"],
                "经验法则": [
                    "种群大小：问题规模的 5-10 倍",
                    "交叉率：0.7-0.9 (高交叉率促进探索)",
                    "变异率：0.05-0.2 (低变异率保持稳定性)"
                ],
                "常见问题": {
                    "早熟收敛": "增加变异率，增加种群多样性",
                    "收敛太慢": "增加交叉率，减少种群大小",
                    "解质量差": "增加迭代次数，改进选择策略"
                }
            },
            "粒子群优化 (PSO)": {
                "关键参数": ["w", "c1", "c2", "swarm_size"],
                "调优顺序": ["w → c1/c2 → swarm_size"],
                "经验法则": [
                    "惯性权重 w: 0.4-0.9 (线性递减策略)",
                    "加速常数 c1=c2=1.5-2.0 (平衡个体/群体)",
                    "种群大小：20-50 (PSO 对小种群也有效)"
                ],
                "常见问题": {
                    "陷入局部最优": "增加 w 或采用自适应 w 策略",
                    "收敛过快": "减小 c1/c2，增加随机性"
                }
            },
            "模拟退火 (SA)": {
                "关键参数": ["initial_temp", "cooling_rate", "min_temp"],
                "调优顺序": ["initial_temp → cooling_rate → min_temp"],
                "经验法则": [
                    "初始温度：使初始接受率约 80%",
                    "冷却率：0.85-0.99 (慢冷却质量好但耗时)",
                    "终止温度：1e-3 到 1e-5"
                ],
                "常见问题": {
                    "接受率太低": "提高初始温度",
                    "搜索太慢": "增加冷却率"
                }
            },
            "变邻域搜索 (VNS)": {
                "关键参数": ["neighborhoods", "max_iterations"],
                "调优顺序": ["neighborhood 设计 → 迭代次数"],
                "经验法则": [
                    "邻域数量：3-5 个 (从简单到复杂)",
                    "邻域顺序：先小扰动后大扰动",
                    "每个邻域迭代：10-50 次"
                ],
                "常见问题": {
                    "改进有限": "设计更有针对性的邻域结构",
                    "运行时间长": "减少邻域数量或迭代次数"
                }
            }
        }
        
        return strategies.get(algorithm_name, {
            "提示": "该算法的详细调优策略待补充",
            "建议": "参考相关文献或进行参数敏感性分析"
        })


# ============================================================================
# 性能预估模型
# ============================================================================

class PerformanceEstimator:
    """性能预估模型 (基于经验数据)"""
    
    # 基准性能数据 (秒/节点，相对值)
    BASE_PERFORMANCE = {
        "MILP (PuLP/CBC)": 0.5,
        "Dijkstra": 0.001,
        "A* Search": 0.002,
        "MST (Kruskal/Prim)": 0.0015,
        "遗传算法 (GA)": 0.02,
        "粒子群优化 (PSO)": 0.015,
        "模拟退火 (SA)": 0.018,
        "变邻域搜索 (VNS)": 0.012,
        "禁忌搜索 (TS)": 0.015,
        "蚁群算法 (ACO)": 0.03,
        "NSGA-II": 0.035,
        "DQN (深度强化学习)": 0.05,
        "图神经网络 (GNN)": 0.08,
    }
    
    @classmethod
    def estimate_runtime(cls, algorithm_name: str, num_nodes: int) -> float:
        """预估运行时间 (秒)"""
        base = cls.BASE_PERFORMANCE.get(algorithm_name, 0.01)
        
        # 不同算法的规模缩放因子
        if algorithm_name in ["MILP (PuLP/CBC)"]:
            scale_factor = (num_nodes / 50) ** 3  # 指数增长
        elif algorithm_name in ["Dijkstra", "A* Search", "MST (Kruskal/Prim)"]:
            scale_factor = (num_nodes / 50) * np.log2(num_nodes / 50 + 1)
        else:
            scale_factor = (num_nodes / 50) ** 2  # 多项式增长
        
        return base * num_nodes * scale_factor
    
    @classmethod
    def estimate_quality(cls, algorithm_name: str, num_nodes: int) -> float:
        """预估解的质量 (相对于最优解的百分比)"""
        base_quality = {
            "MILP (PuLP/CBC)": 1.0,
            "Dijkstra": 1.0,
            "A* Search": 1.0,
            "MST (Kruskal/Prim)": 0.98,
            "遗传算法 (GA)": 0.90,
            "粒子群优化 (PSO)": 0.87,
            "模拟退火 (SA)": 0.85,
            "变邻域搜索 (VNS)": 0.92,
            "禁忌搜索 (TS)": 0.90,
            "蚁群算法 (ACO)": 0.89,
            "NSGA-II": 0.88,
            "DQN (深度强化学习)": 0.82,
            "图神经网络 (GNN)": 0.84,
        }
        
        quality = base_quality.get(algorithm_name, 0.85)
        
        # 大规模问题时启发式算法质量可能下降
        if num_nodes > 200 and algorithm_name not in ["MILP (PuLP/CBC)", "Dijkstra", "A* Search", "MST (Kruskal/Prim)"]:
            quality *= 0.95
        
        return quality


# ============================================================================
# 可视化与演示
# ============================================================================

class SelectionGuideVisualizer:
    """选择指南可视化"""
    
    @staticmethod
    def plot_algorithm_comparison(scales: List[str]):
        """绘制算法对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 运行时间对比
        ax1 = axes[0, 0]
        algorithms = ["Dijkstra", "MST", "PSO", "GA", "VNS", "MILP"]
        times_50 = [PerformanceEstimator.estimate_runtime(a, 50) for a in algorithms]
        times_200 = [PerformanceEstimator.estimate_runtime(a, 200) for a in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        ax1.bar(x - width/2, times_50, width, label='50 节点', alpha=0.8)
        ax1.bar(x + width/2, times_200, width, label='200 节点', alpha=0.8)
        ax1.set_ylabel('预估时间 (秒)')
        ax1.set_title('算法运行时间对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 解的质量对比
        ax2 = axes[0, 1]
        qualities = [PerformanceEstimator.estimate_quality(a, 100) for a in algorithms]
        colors = plt.cm.RdYlGn(qualities)
        bars = ax2.bar(algorithms, qualities, color=colors, alpha=0.8)
        ax2.set_ylabel('解的质量 (相对最优解)')
        ax2.set_title('算法解的质量对比')
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='质量阈值 90%')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. 实现难度 vs 性能
        ax3 = axes[1, 0]
        algo_profiles = {a.name: a for a in ALGORITHM_DATABASE}
        for algo_name in ["Dijkstra", "MST", "PSO", "GA", "VNS", "NSGA-II", "DQN", "GNN"]:
            if algo_name in algo_profiles:
                profile = algo_profiles[algo_name]
                ax3.scatter(1/profile.ease_of_implementation, 
                           profile.solution_quality,
                           s=100, alpha=0.7, label=algo_name)
        
        ax3.set_xlabel('实现难度 (1=容易，4=困难)')
        ax3.set_ylabel('解的质量')
        ax3.set_title('实现难度 vs 解的质量')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(alpha=0.3)
        
        # 4. 推荐场景热力图
        ax4 = axes[1, 1]
        categories = ['小规模', '中规模', '大规模', '多目标', '实时性']
        algo_short = ["MILP", "Dijkstra", "PSO", "GA", "VNS", "NSGA-II", "DQN"]
        
        # 简化的推荐分数矩阵
        scores = np.array([
            [0.9, 0.7, 0.5, 0.6, 0.3],  # MILP
            [0.8, 0.8, 0.7, 0.4, 0.9],  # Dijkstra
            [0.6, 0.8, 0.8, 0.5, 0.6],  # PSO
            [0.5, 0.8, 0.9, 0.8, 0.4],  # GA
            [0.6, 0.8, 0.9, 0.7, 0.5],  # VNS
            [0.4, 0.7, 0.8, 0.9, 0.3],  # NSGA-II
            [0.3, 0.6, 0.8, 0.7, 0.2],  # DQN
        ])
        
        im = ax4.imshow(scores, cmap='YlGnBu', aspect='auto')
        ax4.set_xticks(np.arange(len(categories)))
        ax4.set_yticks(np.arange(len(algo_short)))
        ax4.set_xticklabels(categories)
        ax4.set_yticklabels(algo_short)
        ax4.set_title('算法推荐场景热力图')
        plt.colorbar(im, ax=ax4, label='推荐强度')
        
        # 标注高分区域
        for i in range(len(algo_short)):
            for j in range(len(categories)):
                if scores[i, j] > 0.7:
                    ax4.text(j, i, f'{scores[i, j]:.1f}', 
                            ha='center', va='center', color='darkblue', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('outputs/28_algorithm_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ 可视化已保存：outputs/28_algorithm_comparison.png")
        plt.close()


# ============================================================================
# 主程序：演示算法选择
# ============================================================================

def main():
    """主程序：演示算法选择指南"""
    print("=" * 70)
    print("🧠 线缆布线优化 - 算法选择指南与最佳实践")
    print("=" * 70)
    print()
    
    # 初始化引擎
    engine = AlgorithmRecommendationEngine()
    
    # 示例问题 1: 小规模简单问题
    print("【示例 1】小规模园区网络布线 (30 节点)")
    print("-" * 70)
    problem1 = ProblemCharacteristics(
        num_nodes=30,
        num_edges=80,
        constraint_types=[ConstraintType.SIMPLE],
        objective_type=ObjectiveType.SINGLE_COST,
        solution_quality_priority=0.8
    )
    print(engine.generate_report(problem1))
    print()
    
    # 示例问题 2: 中等规模多约束问题
    print("\n【示例 2】城市电网规划 (150 节点，多约束)")
    print("-" * 70)
    problem2 = ProblemCharacteristics(
        num_nodes=150,
        num_edges=400,
        constraint_types=[ConstraintType.CAPACITY, ConstraintType.GEOMETRIC],
        objective_type=ObjectiveType.MULTI,
        has_obstacles=True,
        solution_quality_priority=0.6
    )
    print(engine.generate_report(problem2))
    print()
    
    # 示例问题 3: 大规模实时问题
    print("\n【示例 3】数据中心网络 (500 节点，实时要求)")
    print("-" * 70)
    problem3 = ProblemCharacteristics(
        num_nodes=500,
        num_edges=1500,
        constraint_types=[ConstraintType.CAPACITY],
        objective_type=ObjectiveType.SINGLE_COST,
        real_time_required=True,
        solution_quality_priority=0.4
    )
    print(engine.generate_report(problem3))
    print()
    
    # 参数调优指南示例
    print("\n【参数调优指南】")
    print("=" * 70)
    tuning_guide = ParameterTuningGuide()
    for algo_name in ["遗传算法 (GA)", "粒子群优化 (PSO)", "模拟退火 (SA)"]:
        strategy = tuning_guide.get_tuning_strategy(algo_name)
        print(f"\n{algo_name}:")
        for key, value in strategy.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    • {item}")
            else:
                print(f"  {key}: {value}")
    print()
    
    # 性能预估示例
    print("\n【性能预估示例】")
    print("=" * 70)
    print(f"{'算法':<25} {'50 节点 (s)':<15} {'200 节点 (s)':<15} {'质量':<10}")
    print("-" * 70)
    for algo_name in ["Dijkstra", "MST", "PSO", "GA", "VNS", "MILP"]:
        t50 = PerformanceEstimator.estimate_runtime(algo_name, 50)
        t200 = PerformanceEstimator.estimate_runtime(algo_name, 200)
        q = PerformanceEstimator.estimate_quality(algo_name, 100)
        print(f"{algo_name:<25} {t50:<15.3f} {t200:<15.3f} {q*100:<10.1f}%")
    print()
    
    # 生成可视化
    print("\n【生成算法对比可视化】")
    print("-" * 70)
    SelectionGuideVisualizer.plot_algorithm_comparison(['small', 'medium', 'large'])
    
    print("\n" + "=" * 70)
    print("✅ 算法选择指南演示完成!")
    print("=" * 70)
    print("\n📚 关键总结:")
    print("  1. 小规模 (<50 节点): 优先精确算法 (MILP/Dijkstra/MST)")
    print("  2. 中规模 (50-200 节点): 启发式算法 (GA/PSO/VNS)")
    print("  3. 大规模 (>200 节点): 元启发式或机器学习方法")
    print("  4. 多目标问题：NSGA-II 等 Pareto 优化")
    print("  5. 实时应用：快速算法或预训练模型")
    print("\n💡 最佳实践:")
    print("  • 从简单算法开始，逐步尝试更复杂方法")
    print("  • 记录实验结果，建立性能基准")
    print("  • 参数调优遵循经验法则，但需针对具体问题调整")
    print("  • 混合算法往往能取得更好效果")
    print("=" * 70)


if __name__ == "__main__":
    main()
