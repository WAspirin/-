#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线缆布线优化算法 - 综合基准测试框架
Comprehensive Benchmark Framework for Cable Routing Optimization

作者：智子 (Sophon)
日期：2026-03-27 (Day 27)

功能：
- 统一测试环境生成
- 多算法批量测试
- 性能指标自动统计
- 可视化对比分析
- 结果导出与报告生成
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class TestCase:
    """测试用例定义"""
    name: str
    num_nodes: int
    num_edges: int
    num_sources: int
    num_sinks: int
    capacity_range: Tuple[float, float]
    cost_range: Tuple[float, float]
    description: str
    
    def generate_network(self, seed: int = 42) -> Dict:
        """生成测试网络"""
        np.random.seed(seed)
        
        # 生成节点坐标
        nodes = np.random.rand(self.num_nodes, 2) * 100
        
        # 生成边（稀疏图）
        edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.rand() < 0.3:  # 30% 连接概率
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    capacity = np.random.uniform(*self.capacity_range)
                    cost = np.random.uniform(*self.cost_range) * dist
                    edges.append({
                        'from': i,
                        'to': j,
                        'capacity': capacity,
                        'cost': cost,
                        'distance': dist
                    })
        
        # 选择源点和汇点
        sources = np.random.choice(self.num_nodes, self.num_sources, replace=False)
        remaining = [i for i in range(self.num_nodes) if i not in sources]
        sinks = np.random.choice(remaining, self.num_sinks, replace=False)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'sources': sources.tolist(),
            'sinks': sinks.tolist()
        }


@dataclass
class BenchmarkResult:
    """单次测试结果"""
    algorithm_name: str
    test_case_name: str
    seed: int
    solution_cost: float
    computation_time: float
    iterations: int
    success: bool
    error_message: Optional[str] = None
    solution_details: Optional[Dict] = None


@dataclass
class BenchmarkReport:
    """基准测试报告"""
    timestamp: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    results: List[BenchmarkResult] = field(default_factory=list)
    statistics: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'total_tests': self.total_tests,
            'successful_tests': self.successful_tests,
            'failed_tests': self.failed_tests,
            'results': [
                {
                    'algorithm': r.algorithm_name,
                    'test_case': r.test_case_name,
                    'seed': r.seed,
                    'cost': r.solution_cost,
                    'time': r.computation_time,
                    'iterations': r.iterations,
                    'success': r.success
                }
                for r in self.results
            ],
            'statistics': self.statistics
        }


# ============================================================================
# 基准测试框架
# ============================================================================

class BenchmarkFramework:
    """综合基准测试框架"""
    
    def __init__(self, output_dir: str = 'outputs/benchmarks'):
        self.output_dir = output_dir
        self.test_cases: List[TestCase] = []
        self.algorithms: Dict[str, Callable] = {}
        self.results: List[BenchmarkResult] = []
        
    def add_test_case(self, test_case: TestCase):
        """添加测试用例"""
        self.test_cases.append(test_case)
        
    def register_algorithm(self, name: str, algorithm_func: Callable):
        """注册算法"""
        self.algorithms[name] = algorithm_func
        
    def run_single_test(self, algorithm_name: str, test_case: TestCase, 
                       seed: int = 42) -> BenchmarkResult:
        """运行单次测试"""
        network = test_case.generate_network(seed=seed)
        
        start_time = time.time()
        try:
            algorithm = self.algorithms[algorithm_name]
            result = algorithm(network)
            computation_time = time.time() - start_time
            
            return BenchmarkResult(
                algorithm_name=algorithm_name,
                test_case_name=test_case.name,
                seed=seed,
                solution_cost=result.get('cost', float('inf')),
                computation_time=computation_time,
                iterations=result.get('iterations', 0),
                success=True,
                solution_details=result
            )
        except Exception as e:
            computation_time = time.time() - start_time
            return BenchmarkResult(
                algorithm_name=algorithm_name,
                test_case_name=test_case.name,
                seed=seed,
                solution_cost=float('inf'),
                computation_time=computation_time,
                iterations=0,
                success=False,
                error_message=str(e)
            )
    
    def run_all_tests(self, seeds: List[int] = None) -> BenchmarkReport:
        """运行所有测试"""
        if seeds is None:
            seeds = [42, 123, 456]  # 默认 3 个随机种子
        
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_tests=0,
            successful_tests=0,
            failed_tests=0
        )
        
        for test_case in self.test_cases:
            for algorithm_name in self.algorithms.keys():
                for seed in seeds:
                    result = self.run_single_test(algorithm_name, test_case, seed)
                    self.results.append(result)
                    report.results.append(result)
                    report.total_tests += 1
                    
                    if result.success:
                        report.successful_tests += 1
                    else:
                        report.failed_tests += 1
                    
                    print(f"✓ {algorithm_name} on {test_case.name} (seed={seed}): "
                          f"cost={result.solution_cost:.2f}, "
                          f"time={result.computation_time:.3f}s")
        
        # 计算统计信息
        report.statistics = self._compute_statistics(report.results)
        
        return report
    
    def _compute_statistics(self, results: List[BenchmarkResult]) -> Dict:
        """计算统计信息"""
        stats = {
            'by_algorithm': {},
            'by_test_case': {},
            'overall': {}
        }
        
        # 按算法分组
        for algo_name in self.algorithms.keys():
            algo_results = [r for r in results if r.algorithm_name == algo_name and r.success]
            if algo_results:
                costs = [r.solution_cost for r in algo_results]
                times = [r.computation_time for r in algo_results]
                stats['by_algorithm'][algo_name] = {
                    'mean_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'min_cost': np.min(costs),
                    'max_cost': np.max(costs),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'success_rate': len(algo_results) / len([r for r in results if r.algorithm_name == algo_name])
                }
        
        # 按测试用例分组
        for test_case in self.test_cases:
            case_results = [r for r in results if r.test_case_name == test_case.name and r.success]
            if case_results:
                costs = [r.solution_cost for r in case_results]
                times = [r.computation_time for r in case_results]
                stats['by_test_case'][test_case.name] = {
                    'mean_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'min_cost': np.min(costs),
                    'max_cost': np.max(costs),
                    'mean_time': np.mean(times),
                    'best_algorithm': min(case_results, key=lambda r: r.solution_cost).algorithm_name
                }
        
        # 总体统计
        successful_results = [r for r in results if r.success]
        if successful_results:
            stats['overall'] = {
                'total_tests': len(results),
                'successful_tests': len(successful_results),
                'success_rate': len(successful_results) / len(results),
                'avg_cost': np.mean([r.solution_cost for r in successful_results]),
                'avg_time': np.mean([r.computation_time for r in successful_results])
            }
        
        return stats
    
    def save_report(self, report: BenchmarkReport, filename: str = None):
        """保存报告"""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"✓ 报告已保存：{filepath}")
        return filepath


# ============================================================================
# 可视化分析
# ============================================================================

class BenchmarkVisualizer:
    """基准测试可视化"""
    
    def __init__(self, report: BenchmarkReport):
        self.report = report
        self.results = report.results
        
    def plot_cost_comparison(self, save_path: str = None):
        """成本对比图"""
        # 按算法分组
        algo_costs = {}
        for result in self.results:
            if result.success:
                if result.algorithm_name not in algo_costs:
                    algo_costs[result.algorithm_name] = []
                algo_costs[result.algorithm_name].append(result.solution_cost)
        
        # 绘制箱线图
        plt.figure(figsize=(14, 8))
        algorithms = list(algo_costs.keys())
        costs = [algo_costs[algo] for algo in algorithms]
        
        sns.boxplot(data=costs)
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        plt.ylabel('Solution Cost')
        plt.title('算法成本对比 (Box Plot)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图表已保存：{save_path}")
        plt.show()
    
    def plot_time_comparison(self, save_path: str = None):
        """时间对比图"""
        algo_times = {}
        for result in self.results:
            if result.success:
                if result.algorithm_name not in algo_times:
                    algo_times[result.algorithm_name] = []
                algo_times[result.algorithm_name].append(result.computation_time)
        
        plt.figure(figsize=(14, 8))
        algorithms = list(algo_times.keys())
        times = [algo_times[algo] for algo in algorithms]
        
        sns.boxplot(data=times)
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        plt.ylabel('Computation Time (s)')
        plt.title('算法计算时间对比')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图表已保存：{save_path}")
        plt.show()
    
    def plot_cost_time_scatter(self, save_path: str = None):
        """成本 - 时间散点图"""
        plt.figure(figsize=(12, 8))
        
        for algo_name in set(r.algorithm_name for r in self.results if r.success):
            algo_results = [r for r in self.results 
                          if r.algorithm_name == algo_name and r.success]
            costs = [r.solution_cost for r in algo_results]
            times = [r.computation_time for r in algo_results]
            
            plt.scatter(times, costs, label=algo_name, alpha=0.6, s=100)
        
        plt.xlabel('Computation Time (s)')
        plt.ylabel('Solution Cost')
        plt.title('算法性能对比：成本 vs 时间')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图表已保存：{save_path}")
        plt.show()
    
    def plot_success_rate(self, save_path: str = None):
        """成功率对比图"""
        algo_stats = {}
        for algo_name in set(r.algorithm_name for r in self.results):
            algo_results = [r for r in self.results if r.algorithm_name == algo_name]
            success_count = sum(1 for r in algo_results if r.success)
            algo_stats[algo_name] = success_count / len(algo_results) * 100
        
        plt.figure(figsize=(14, 6))
        algorithms = list(algo_stats.keys())
        rates = [algo_stats[algo] for algo in algorithms]
        
        bars = plt.bar(algorithms, rates, color='steelblue', alpha=0.8)
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        plt.ylabel('Success Rate (%)')
        plt.title('算法成功率对比')
        plt.ylim(0, 105)
        
        # 添加数值标签
        for bar, rate in zip(bars, rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图表已保存：{save_path}")
        plt.show()
    
    def generate_dashboard(self, save_dir: str = None):
        """生成综合仪表板"""
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 成本对比
        algo_costs = {}
        for result in self.results:
            if result.success:
                if result.algorithm_name not in algo_costs:
                    algo_costs[result.algorithm_name] = []
                algo_costs[result.algorithm_name].append(result.solution_cost)
        
        if algo_costs:
            algorithms = list(algo_costs.keys())[:8]  # 最多显示 8 个
            costs = [algo_costs[algo] for algo in algorithms]
            axes[0, 0].boxplot(costs, labels=algorithms, patch_artist=True)
            axes[0, 0].set_ylabel('Cost')
            axes[0, 0].set_title('成本对比 (Top 8)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 时间对比
        algo_times = {}
        for result in self.results:
            if result.success:
                if result.algorithm_name not in algo_times:
                    algo_times[result.algorithm_name] = []
                algo_times[result.algorithm_name].append(result.computation_time)
        
        if algo_times:
            algorithms = list(algo_times.keys())[:8]
            times = [algo_times[algo] for algo in algorithms]
            axes[0, 1].boxplot(times, labels=algorithms, patch_artist=True)
            axes[0, 1].set_ylabel('Time (s)')
            axes[0, 1].set_title('计算时间对比 (Top 8)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 成本 - 时间散点
        for algo_name in set(r.algorithm_name for r in self.results if r.success):
            algo_results = [r for r in self.results 
                          if r.algorithm_name == algo_name and r.success]
            costs = [r.solution_cost for r in algo_results]
            times = [r.computation_time for r in algo_results]
            axes[1, 0].scatter(times, costs, label=algo_name, alpha=0.6, s=50)
        
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].set_title('性能对比：成本 vs 时间')
        axes[1, 0].legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 成功率
        algo_stats = {}
        for algo_name in set(r.algorithm_name for r in self.results):
            algo_results = [r for r in self.results if r.algorithm_name == algo_name]
            success_count = sum(1 for r in algo_results if r.success)
            algo_stats[algo_name] = success_count / len(algo_results) * 100
        
        algorithms = list(algo_stats.keys())[:8]
        rates = [algo_stats[algo] for algo in algorithms]
        axes[1, 1].bar(algorithms, rates, color='steelblue', alpha=0.8)
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title('成功率对比 (Top 8)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save_dir:
            filepath = os.path.join(save_dir, f"benchmark_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ 仪表板已保存：{filepath}")
        
        plt.show()


# ============================================================================
# 示例算法（用于测试框架）
# ============================================================================

def greedy_algorithm(network: Dict) -> Dict:
    """贪心算法示例"""
    edges = network['edges']
    sources = network['sources']
    sinks = network['sinks']
    
    # 简单实现：选择成本最低的边
    sorted_edges = sorted(edges, key=lambda e: e['cost'])
    selected_edges = sorted_edges[:min(10, len(sorted_edges))]
    
    total_cost = sum(e['cost'] for e in selected_edges)
    
    return {
        'cost': total_cost,
        'iterations': 1,
        'selected_edges': selected_edges
    }


def random_algorithm(network: Dict) -> Dict:
    """随机算法示例"""
    edges = network['edges']
    
    # 随机选择边
    np.random.seed(42)
    num_select = max(1, len(edges) // 3)
    selected_indices = np.random.choice(len(edges), num_select, replace=False)
    selected_edges = [edges[i] for i in selected_indices]
    
    total_cost = sum(e['cost'] for e in selected_edges)
    
    return {
        'cost': total_cost,
        'iterations': 100,
        'selected_edges': selected_edges
    }


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    print("=" * 60)
    print("线缆布线优化 - 综合基准测试框架")
    print("Benchmark Framework for Cable Routing Optimization")
    print("=" * 60)
    print()
    
    # 创建测试用例
    test_cases = [
        TestCase(
            name='small_sparse',
            num_nodes=10,
            num_edges=30,
            num_sources=2,
            num_sinks=2,
            capacity_range=(50, 100),
            cost_range=(0.5, 2.0),
            description='小规模稀疏网络'
        ),
        TestCase(
            name='medium_dense',
            num_nodes=30,
            num_edges=150,
            num_sources=3,
            num_sinks=3,
            capacity_range=(100, 200),
            cost_range=(1.0, 3.0),
            description='中等规模稠密网络'
        ),
        TestCase(
            name='large_scale',
            num_nodes=50,
            num_edges=300,
            num_sources=5,
            num_sinks=5,
            capacity_range=(200, 500),
            cost_range=(2.0, 5.0),
            description='大规模网络'
        )
    ]
    
    # 创建框架
    framework = BenchmarkFramework(output_dir='outputs/benchmarks')
    
    # 添加测试用例
    for tc in test_cases:
        framework.add_test_case(tc)
        print(f"✓ 添加测试用例：{tc.name} ({tc.description})")
    
    # 注册算法
    framework.register_algorithm('greedy', greedy_algorithm)
    framework.register_algorithm('random', random_algorithm)
    print(f"✓ 注册算法：{list(framework.algorithms.keys())}")
    print()
    
    # 运行测试
    print("开始运行基准测试...")
    print("-" * 60)
    report = framework.run_all_tests(seeds=[42, 123, 456])
    print("-" * 60)
    print()
    
    # 打印统计摘要
    print("📊 测试摘要:")
    print(f"  总测试数：{report.total_tests}")
    print(f"  成功：{report.successful_tests}")
    print(f"  失败：{report.failed_tests}")
    print(f"  成功率：{report.successful_tests/report.total_tests*100:.1f}%")
    print()
    
    # 保存报告
    framework.save_report(report)
    
    # 可视化
    print("\n生成可视化图表...")
    visualizer = BenchmarkVisualizer(report)
    
    import os
    output_dir = 'outputs/benchmarks/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer.plot_cost_comparison(save_path=f"{output_dir}/cost_comparison.png")
    visualizer.plot_time_comparison(save_path=f"{output_dir}/time_comparison.png")
    visualizer.plot_cost_time_scatter(save_path=f"{output_dir}/cost_time_scatter.png")
    visualizer.plot_success_rate(save_path=f"{output_dir}/success_rate.png")
    visualizer.generate_dashboard(save_dir=output_dir)
    
    print("\n" + "=" * 60)
    print("基准测试完成！")
    print("=" * 60)
    
    return report


if __name__ == '__main__':
    main()
