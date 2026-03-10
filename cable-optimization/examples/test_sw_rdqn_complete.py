"""
SW-RDQN 完整系统测试脚本

整合:
1. Voronoi.py - 关键点生成器
2. SW_RDQN.py - SW-RDQN 控制器

功能:
- 端到端测试
- 训练可视化
- 性能对比

作者：WonderXi + 智子 (Sophon)
日期：2026-03-10
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
from typing import List, Tuple
import sys
import os

# 导入 Voronoi 规划器
from Voronoi import VoronoiKeyPointGenerator, create_paper_map

# 导入 SW-RDQN 控制器
from SW_RDQN import LocalSWRDQNController, DynamicObstacle

# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class GridEnvironment:
    """网格环境"""
    def __init__(self, grid_map: np.ndarray):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
    
    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.height and 0 <= y < self.width and self.grid_map[x, y] == 0


def plot_training_curve(stats_list: List[dict], save_path: str = None):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 成功率曲线
    ax1.plot([s['success_rate'] for s in stats_list], 'b-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Training Success Rate')
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线
    ax2.plot([s['total_loss'] for s in stats_list], 'r-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def test_voronoi():
    """测试 Voronoi 关键点生成"""
    print("="*60)
    print("测试 1: Voronoi 关键点生成")
    print("="*60)
    
    grid_map = create_paper_map()
    generator = VoronoiKeyPointGenerator(grid_map, safety_distance=4.5, remove_collinear=True)
    
    start_pos = (12, 12)
    goal_pos = (75, 130)
    
    try:
        key_points = generator.extract_key_points(start_pos, goal_pos)
        print(f"✓ 生成 {len(key_points)} 个关键点")
        for i, kp in enumerate(key_points):
            print(f"  {i}: {kp}")
        
        generator.visualize(save_path="voronoi_result.png")
        print("✓ 可视化已保存：voronoi_result.png")
        
        return key_points
    except Exception as e:
        print(f"✗ 错误：{e}")
        return None


def test_sw_rdqn(key_points: List[Tuple[int, int]]):
    """测试 SW-RDQN 控制器"""
    print("\n" + "="*60)
    print("测试 2: SW-RDQN 训练")
    print("="*60)
    
    grid_map = create_paper_map()
    env = GridEnvironment(grid_map)
    
    start = (12, 12)
    goal = (75, 130)
    
    # 动态障碍物
    dynamic_obstacles = [
        DynamicObstacle(30, 30, 0.5, 0.3, size=3),
        DynamicObstacle(50, 50, -0.3, 0.5, size=2)
    ]
    
    # 初始化控制器
    controller = LocalSWRDQNController(
        grid_map=grid_map,
        key_path_points=key_points
    )
    
    # 训练
    n_episodes = 100
    stats_history = []
    
    print(f"开始训练：{n_episodes} 回合")
    print(f"关键点数量：{len(key_points)}")
    print(f"当前阶段：{controller.current_stage}/{len(key_points)-1}")
    
    for episode in range(n_episodes):
        # 重置阶段
        controller.current_stage = 0
        
        total_reward, success = controller.train_episode(
            env, start, goal, dynamic_obstacles, max_steps=200
        )
        
        stats_history.append({
            'success_rate': controller.stats['success_rate'],
            'total_loss': controller.stats['total_loss']
        })
        
        if (episode + 1) % 10 == 0:
            print(f"回合 {episode+1}/{n_episodes}: "
                  f"成功率 = {controller.stats['success_rate']:.1%}, "
                  f"epsilon = {controller.epsilon:.3f}")
        
        # 早停检查
        if controller.should_early_stop():
            print(f"早停触发：连续{controller.max_patience}回合无改进")
            break
    
    # 绘制训练曲线
    plot_training_curve(stats_history, save_path="training_curve.png")
    print("✓ 训练曲线已保存：training_curve.png")
    
    return controller, stats_history


def main():
    """主测试流程"""
    print("="*60)
    print("SW-RDQN 完整系统测试")
    print("="*60)
    
    # 测试 1: Voronoi 关键点生成
    key_points = test_voronoi()
    
    if key_points is None:
        print("✗ Voronoi 测试失败，终止")
        return
    
    # 测试 2: SW-RDQN 训练
    controller, stats_history = test_sw_rdqn(key_points)
    
    # 总结
    print("\n" + "="*60)
    print("测试完成总结")
    print("="*60)
    print(f"✓ Voronoi 关键点：{len(key_points)} 个")
    print(f"✓ SW-RDQN 训练：{len(stats_history)} 回合")
    print(f"✓ 最终成功率：{controller.stats['success_rate']:.1%}")
    print(f"✓ 输出文件:")
    print(f"  - voronoi_result.png")
    print(f"  - training_curve.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
