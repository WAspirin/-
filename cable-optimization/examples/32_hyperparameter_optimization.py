#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数自动调优框架 - 使用 Optuna 优化线缆布线算法

作者：智子 (Sophon)
日期：2026-03-30
学习阶段：深化应用阶段 - 超参数优化专题

核心功能:
- 使用 Optuna 自动调优 PSO、SA、GA 等算法参数
- 构建通用的超参数优化框架
- 对比手动调参 vs 自动调优效果
- 可视化优化过程和结果
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from typing import Dict, List, Tuple, Callable, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna 未安装，请运行：pip install optuna")

from scipy.spatial.distance import cdist

# ============================================================================
# 基础网络模型
# ============================================================================

class CableNetwork:
    """线缆布线网络基础模型"""
    
    def __init__(self, nodes: np.ndarray, node_type: List[str] = None):
        """
        初始化网络
        
        Args:
            nodes: 节点坐标 (n_nodes, 2)
            node_type: 节点类型列表 (core/aggregation/access)
        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.node_type = node_type or ['access'] * self.n_nodes
        
        # 计算距离矩阵
        self.dist_matrix = cdist(nodes, nodes, metric='euclidean')
        np.fill_diagonal(self.dist_matrix, np.inf)
        
    def get_distance(self, i: int, j: int) -> float:
        """获取节点间距离"""
        return self.dist_matrix[i, j]
    
    def get_total_cable_length(self, edges: List[Tuple[int, int]]) -> float:
        """计算总线缆长度"""
        return sum(self.dist_matrix[i, j] for i, j in edges)


# ============================================================================
# 优化算法实现（简化版用于调优测试）
# ============================================================================

class PSOOptimizer:
    """粒子群优化算法 - 支持参数调优"""
    
    def __init__(self, network: CableNetwork, 
                 n_particles: int = 20,
                 max_iter: int = 100,
                 w: float = 0.7,  # 惯性权重
                 c1: float = 1.5,  # 个体学习因子
                 c2: float = 1.5,  # 群体学习因子
                 v_max: float = None):
        self.network = network
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max or (np.max(network.dist_matrix) * 0.1)
        
        # 初始化粒子
        self.positions = np.random.rand(n_particles, network.n_nodes, 2)
        self.velocities = np.random.rand(n_particles, network.n_nodes, 2) * 2 - 1
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(n_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        
    def evaluate(self, position: np.ndarray) -> float:
        """评估粒子位置（简化：最小化总距离）"""
        # 将粒子位置转换为路径
        path = np.argsort(position[:, 0])
        
        # 计算路径总长度
        total_length = 0
        for i in range(len(path) - 1):
            total_length += self.network.get_distance(path[i], path[i+1])
        total_length += self.network.get_distance(path[-1], path[0])  # 回到起点
        
        return total_length
    
    def optimize(self, verbose: bool = False) -> Tuple[float, List[float]]:
        """运行 PSO 优化"""
        convergence_history = []
        
        for iter in range(self.max_iter):
            for i in range(self.n_particles):
                # 评估当前粒子
                score = self.evaluate(self.positions[i])
                
                # 更新个体最优
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i].copy()
                
                # 更新全局最优
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()
            
            convergence_history.append(self.global_best_score)
            
            # 更新速度和位置
            r1, r2 = np.random.rand(2, self.n_particles, self.n_nodes, 2)
            self.velocities = (self.w * self.velocities + 
                             self.c1 * r1 * (self.best_positions - self.positions) +
                             self.c2 * r2 * (self.global_best_position - self.positions))
            
            # 速度限制
            self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
            self.positions += self.velocities
            
            # 位置边界
            self.positions = np.clip(self.positions, 0, 1)
            
            if verbose and (iter + 1) % 20 == 0:
                print(f"  Iter {iter+1}/{self.max_iter}, Best: {self.global_best_score:.2f}")
        
        return self.global_best_score, convergence_history


class SimulatedAnnealingOptimizer:
    """模拟退火算法 - 支持参数调优"""
    
    def __init__(self, network: CableNetwork,
                 initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995,
                 min_temp: float = 1e-8,
                 n_iterations: int = 100):
        self.network = network
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.n_iterations = n_iterations
        
    def evaluate(self, path: List[int]) -> float:
        """评估路径质量"""
        total_length = 0
        for i in range(len(path) - 1):
            total_length += self.network.get_distance(path[i], path[i+1])
        total_length += self.network.get_distance(path[-1], path[0])
        return total_length
    
    def neighbor(self, path: List[int]) -> List[int]:
        """生成邻域解（交换两个节点）"""
        new_path = path.copy()
        i, j = np.random.choice(len(path), 2, replace=False)
        new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path
    
    def optimize(self, verbose: bool = False) -> Tuple[float, List[float]]:
        """运行模拟退火"""
        # 初始解
        current_path = list(np.random.permutation(self.network.n_nodes))
        current_score = self.evaluate(current_path)
        
        best_path = current_path.copy()
        best_score = current_score
        
        temp = self.initial_temp
        convergence_history = [best_score]
        
        iteration = 0
        while temp > self.min_temp and iteration < self.n_iterations:
            # 生成邻域解
            new_path = self.neighbor(current_path)
            new_score = self.evaluate(new_path)
            
            # Metropolis 准则
            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current_path = new_path
                current_score = new_score
                
                # 更新最优
                if current_score < best_score:
                    best_path = current_path.copy()
                    best_score = current_score
            
            convergence_history.append(best_score)
            temp *= self.cooling_rate
            iteration += 1
            
            if verbose and iteration % 200 == 0:
                print(f"  Iter {iteration}, Temp: {temp:.4f}, Best: {best_score:.2f}")
        
        return best_score, convergence_history


# ============================================================================
# Optuna 超参数优化框架
# ============================================================================

class HyperparameterTuner:
    """通用超参数调优器"""
    
    def __init__(self, network: CableNetwork, algorithm: str = 'pso'):
        self.network = network
        self.algorithm = algorithm
        self.best_params = None
        self.best_score = np.inf
        self.study = None
        
    def create_study(self, study_name: str = None) -> optuna.Study:
        """创建 Optuna Study"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna 未安装")
        
        study_name = study_name or f"{self.algorithm}_tuning"
        self.study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            load_if_exists=True
        )
        return self.study
    
    def objective_pso(self, trial: optuna.Trial) -> float:
        """PSO 超参数优化目标函数"""
        # 定义搜索空间
        n_particles = trial.suggest_int('n_particles', 10, 50, step=5)
        max_iter = trial.suggest_int('max_iter', 50, 200, step=10)
        w = trial.suggest_float('w', 0.4, 0.9)
        c1 = trial.suggest_float('c1', 1.0, 2.5)
        c2 = trial.suggest_float('c2', 1.0, 2.5)
        
        # 创建并运行优化器
        optimizer = PSOOptimizer(
            self.network,
            n_particles=n_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2
        )
        
        best_score, _ = optimizer.optimize()
        return best_score
    
    def objective_sa(self, trial: optuna.Trial) -> float:
        """SA 超参数优化目标函数"""
        initial_temp = trial.suggest_float('initial_temp', 100, 10000, log=True)
        cooling_rate = trial.suggest_float('cooling_rate', 0.9, 0.999)
        n_iterations = trial.suggest_int('n_iterations', 100, 500, step=50)
        
        optimizer = SimulatedAnnealingOptimizer(
            self.network,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            n_iterations=n_iterations
        )
        
        best_score, _ = optimizer.optimize()
        return best_score
    
    def tune(self, n_trials: int = 20, timeout: int = None, verbose: bool = True) -> Dict:
        """
        执行超参数调优
        
        Args:
            n_trials: 试验次数
            timeout: 超时时间（秒）
            verbose: 是否打印进度
        
        Returns:
            最佳参数和结果
        """
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna 未安装，使用默认参数运行")
            return self._run_default()
        
        self.create_study()
        
        # 选择目标函数
        if self.algorithm == 'pso':
            objective = self.objective_pso
        elif self.algorithm == 'sa':
            objective = self.objective_sa
        else:
            raise ValueError(f"不支持的算法：{self.algorithm}")
        
        # 运行优化
        if verbose:
            print(f"\n🔍 开始 {self.algorithm.upper()} 超参数调优...")
            print(f"   试验次数：{n_trials}")
            print("-" * 60)
        
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose
        )
        
        # 提取最佳结果
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ 超参数调优完成!")
            print(f"\n📊 最佳参数:")
            for key, value in self.best_params.items():
                print(f"   {key}: {value}")
            print(f"\n🎯 最佳得分：{self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def _run_default(self) -> Dict:
        """使用默认参数运行（Optuna 不可用时）"""
        if self.algorithm == 'pso':
            self.best_params = {
                'n_particles': 30,
                'max_iter': 100,
                'w': 0.7,
                'c1': 1.5,
                'c2': 1.5
            }
            optimizer = PSOOptimizer(self.network, **self.best_params)
            self.best_score, _ = optimizer.optimize()
        elif self.algorithm == 'sa':
            self.best_params = {
                'initial_temp': 1000,
                'cooling_rate': 0.995,
                'n_iterations': 200
            }
            optimizer = SimulatedAnnealingOptimizer(self.network, **self.best_params)
            self.best_score, _ = optimizer.optimize()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': 0,
            'study': None
        }


# ============================================================================
# 可视化与对比分析
# ============================================================================

class TuningVisualizer:
    """超参数调优可视化"""
    
    def __init__(self, tuner: HyperparameterTuner):
        self.tuner = tuner
        self.study = tuner.study
        
    def plot_results(self, save_path: str = None) -> None:
        """绘制调优结果"""
        if not OPTUNA_AVAILABLE or self.study is None:
            print("⚠️  无法绘制：Optuna 未安装或无研究数据")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 子图 1: 优化历史
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot([t.value for t in self.study.trials], 'o-', alpha=0.7)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization History')
        ax1.grid(True, alpha=0.3)
        
        # 子图 2: 最佳参数（条形图）
        ax2 = fig.add_subplot(gs[0, 1])
        params = self.tuner.best_params
        param_names = list(params.keys())
        param_values = list(params.values())
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(param_names)))
        bars = ax2.barh(param_names, param_values, color=colors)
        ax2.set_xlabel('Value')
        ax2.set_title('Best Hyperparameters')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 在条形上标注数值
        for bar, value in zip(bars, param_values):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', fontsize=9)
        
        # 子图 3: 参数重要性（如果有）
        ax3 = fig.add_subplot(gs[1, 0])
        try:
            import optuna.visualization as optvis
            # 简化版：显示参数分布
            param_importance = {}
            for trial in self.study.trials:
                for param, value in trial.params.items():
                    if param not in param_importance:
                        param_importance[param] = []
                    param_importance[param].append(value)
            
            # 计算每个参数的方差（作为重要性代理）
            importance = {k: np.var(v) for k, v in param_importance.items()}
            sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            param_names = [p[0] for p in sorted_params]
            importance_vals = [p[1] for p in sorted_params]
            
            colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(param_names)))
            ax3.barh(param_names, importance_vals, color=colors)
            ax3.set_xlabel('Variance (Importance Proxy)')
            ax3.set_title('Parameter Importance (by Variance)')
            ax3.grid(True, alpha=0.3, axis='x')
        except Exception as e:
            ax3.text(0.5, 0.5, f'Importance calc failed:\n{str(e)}',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Parameter Importance')
        
        # 子图 4: 收敛曲线
        ax4 = fig.add_subplot(gs[1, 1])
        best_values = []
        best_so_far = np.inf
        for trial in self.study.trials:
            if trial.value is not None:
                best_so_far = min(best_so_far, trial.value)
                best_values.append(best_so_far)
        
        ax4.plot(best_values, 'r-', linewidth=2, label='Best Value')
        ax4.set_xlabel('Trial Number')
        ax4.set_ylabel('Best Objective Value')
        ax4.set_title('Convergence Curve')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 总标题
        fig.suptitle(f'Hyperparameter Tuning Results - {self.tuner.algorithm.upper()}',
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 图表已保存：{save_path}")
        
        plt.show()
    
    def print_summary(self) -> None:
        """打印调优总结"""
        print("\n" + "=" * 70)
        print(f"📋 超参数调优总结 - {self.tuner.algorithm.upper()}")
        print("=" * 70)
        
        print(f"\n🎯 最佳目标值：{self.tuner.best_score:.4f}")
        print(f"📊 总试验次数：{len(self.study.trials)}")
        
        # 统计信息
        values = [t.value for t in self.study.trials if t.value is not None]
        print(f"\n📈 统计信息:")
        print(f"   平均值：{np.mean(values):.4f}")
        print(f"   标准差：{np.std(values):.4f}")
        print(f"   最小值：{np.min(values):.4f}")
        print(f"   最大值：{np.max(values):.4f}")
        
        print(f"\n⚙️  最佳参数配置:")
        for param, value in self.tuner.best_params.items():
            print(f"   • {param:20s} = {value}")
        
        print("\n" + "=" * 70)


# ============================================================================
# 主程序：对比实验
# ============================================================================

def create_test_network(n_nodes: int = 15, seed: int = 42) -> CableNetwork:
    """创建测试网络"""
    np.random.seed(seed)
    
    # 生成随机节点
    nodes = np.random.rand(n_nodes, 2) * 100
    
    # 分配节点类型
    node_type = []
    for i in range(n_nodes):
        if i == 0:
            node_type.append('core')
        elif i < 3:
            node_type.append('aggregation')
        else:
            node_type.append('access')
    
    return CableNetwork(nodes, node_type)


def compare_manual_vs_auto(network: CableNetwork, algorithm: str = 'pso') -> Dict:
    """对比手动调参 vs 自动调优"""
    
    print("\n" + "=" * 70)
    print(f"🔬 对比实验：手动调参 vs 自动调优 ({algorithm.upper()})")
    print("=" * 70)
    
    results = {}
    
    # 1. 手动调参（默认参数）
    print("\n1️⃣  手动调参（默认参数）...")
    if algorithm == 'pso':
        manual_params = {
            'n_particles': 30,
            'max_iter': 100,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        }
        optimizer = PSOOptimizer(network, **manual_params)
    else:  # sa
        manual_params = {
            'initial_temp': 1000,
            'cooling_rate': 0.995,
            'n_iterations': 200
        }
        optimizer = SimulatedAnnealingOptimizer(network, **manual_params)
    
    start_time = time.time()
    manual_score, _ = optimizer.optimize()
    manual_time = time.time() - start_time
    
    results['manual'] = {
        'score': manual_score,
        'time': manual_time,
        'params': manual_params
    }
    
    print(f"   得分：{manual_score:.4f}")
    print(f"   时间：{manual_time:.2f}秒")
    
    # 2. 自动调优
    print(f"\n2️⃣  自动调优（Optuna, 20 trials）...")
    tuner = HyperparameterTuner(network, algorithm=algorithm)
    
    start_time = time.time()
    tuning_result = tuner.tune(n_trials=20, verbose=True)
    tuning_time = time.time() - start_time
    
    results['auto'] = {
        'score': tuning_result['best_score'],
        'time': tuning_time,
        'params': tuning_result['best_params']
    }
    
    # 3. 对比分析
    print("\n" + "=" * 70)
    print("📊 对比结果")
    print("=" * 70)
    
    improvement = (manual_score - tuning_result['best_score']) / manual_score * 100
    
    print(f"\n{'指标':<15} {'手动调参':<15} {'自动调优':<15} {'改进':<10}")
    print("-" * 55)
    print(f"{'目标得分':<15} {manual_score:<15.4f} {tuning_result['best_score']:<15.4f} {improvement:+.2f}%")
    print(f"{'运行时间':<15} {manual_time:<15.2f}s {tuning_time:<15.2f}s {tuning_time/manual_time:.1f}x")
    
    print(f"\n💡 洞察:")
    if improvement > 5:
        print(f"   ✅ 自动调优显著提升性能（{improvement:.1f}%）")
    elif improvement > 0:
        print(f"   ✓ 自动调优略有提升（{improvement:.1f}%）")
    else:
        print(f"   ⚠️  默认参数已经很好，自动调优提升有限")
    
    return results


def main():
    """主程序"""
    print("=" * 70)
    print("🚀 超参数自动调优框架 - Optuna 实战")
    print("=" * 70)
    print("\n作者：智子 (Sophon)")
    print("日期：2026-03-30")
    print("学习阶段：深化应用阶段 - 超参数优化专题")
    
    # 检查 Optuna
    if not OPTUNA_AVAILABLE:
        print("\n⚠️  Optuna 未安装，将使用默认参数演示")
        print("   安装命令：pip install optuna")
    
    # 创建测试网络
    print("\n📡 创建测试网络...")
    network = create_test_network(n_nodes=15, seed=42)
    print(f"   节点数：{network.n_nodes}")
    print(f"   节点类型：{network.node_type}")
    
    # 对比实验：PSO
    print("\n" + "=" * 70)
    print("🧪 实验 1: PSO 超参数调优")
    print("=" * 70)
    pso_results = compare_manual_vs_auto(network, algorithm='pso')
    
    # 对比实验：SA
    print("\n" + "=" * 70)
    print("🧪 实验 2: SA 超参数调优")
    print("=" * 70)
    sa_results = compare_manual_vs_auto(network, algorithm='sa')
    
    # 可视化（如果 Optuna 可用）
    if OPTUNA_AVAILABLE:
        print("\n" + "=" * 70)
        print("📊 可视化调优结果")
        print("=" * 70)
        
        # PSO 可视化
        pso_tuner = HyperparameterTuner(network, algorithm='pso')
        pso_tuner.tune(n_trials=15, verbose=False)
        viz_pso = TuningVisualizer(pso_tuner)
        viz_pso.plot_results(save_path='/root/.openclaw/workspace/cable-optimization/outputs/32_pso_tuning.png')
        viz_pso.print_summary()
        
        # SA 可视化
        sa_tuner = HyperparameterTuner(network, algorithm='sa')
        sa_tuner.tune(n_trials=15, verbose=False)
        viz_sa = TuningVisualizer(sa_tuner)
        viz_sa.plot_results(save_path='/root/.openclaw/workspace/cable-optimization/outputs/32_sa_tuning.png')
        viz_sa.print_summary()
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 超参数调优框架演示完成!")
    print("=" * 70)
    
    print("\n📚 关键要点:")
    print("   1. Optuna 提供自动化超参数搜索")
    print("   2. 支持多种采样算法（TPE, Random, CMA-ES）")
    print("   3. 可并行运行加速调优过程")
    print("   4. 内置可视化功能")
    
    print("\n💡 实际应用建议:")
    print("   • 对于关键算法，投入时间进行自动调优")
    print("   • 使用早停策略减少调优时间")
    print("   • 保存最佳参数供后续使用")
    print("   • 定期重新调优以适应数据变化")
    
    print("\n📁 输出文件:")
    print("   • examples/32_hyperparameter_optimization.py")
    print("   • outputs/32_pso_tuning.png")
    print("   • outputs/32_sa_tuning.png")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
