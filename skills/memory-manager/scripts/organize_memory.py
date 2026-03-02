#!/usr/bin/env python3
"""
记忆整理脚本 - 帮助智子整理过期的每日记忆文件
将重要的内容提炼到 MEMORY.md，归档或删除旧文件
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

def get_memory_dir():
    """获取记忆目录路径"""
    workspace = Path("/root/.openclaw/workspace")
    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    return memory_dir

def get_daily_files(days_old=7):
    """获取指定天数之前的每日记忆文件"""
    memory_dir = get_memory_dir()
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    old_files = []
    for f in memory_dir.glob("*.md"):
        try:
            # 解析文件名中的日期
            date_str = f.stem  # YYYY-MM-DD
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff_date:
                old_files.append(f)
        except ValueError:
            # 不是日期格式的文件，跳过
            continue
    
    return sorted(old_files)

def extract_key_points(file_path):
    """从文件中提取关键点（简单实现）"""
    key_points = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 简单提取：包含"重要"、"决定"、"偏好"的行
            for line in content.split('\n'):
                if any(keyword in line for keyword in ['重要', '决定', '偏好', '记住', '喜欢', '不喜欢']):
                    key_points.append(line.strip())
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
    
    return key_points

def organize_memory(days_old=7, dry_run=False):
    """整理记忆文件"""
    print(f"🧠 开始整理 {days_old} 天前的记忆文件...")
    
    old_files = get_daily_files(days_old)
    
    if not old_files:
        print("✅ 没有需要整理的旧文件")
        return
    
    print(f"📋 找到 {len(old_files)} 个旧文件:")
    for f in old_files:
        print(f"  - {f.name}")
    
    if dry_run:
        print("\n[干跑模式] 不会实际修改文件")
        return
    
    # 提取关键点
    all_points = []
    for f in old_files:
        points = extract_key_points(f)
        if points:
            all_points.append(f"\n### 来自 {f.stem}")
            all_points.extend(points)
    
    # 如果有关键点，追加到 MEMORY.md
    if all_points:
        memory_md = Path("/root/.openclaw/workspace/MEMORY.md")
        with open(memory_md, 'a', encoding='utf-8') as f:
            f.write(f"\n\n## 自动整理 - {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write('\n'.join(all_points))
            f.write("\n")
        print(f"✅ 已提取 {len(all_points)} 条关键点写入 MEMORY.md")
    
    # 归档旧文件（移动到其他目录而不是删除）
    archive_dir = get_memory_dir() / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    for f in old_files:
        dest = archive_dir / f.name
        f.rename(dest)
        print(f"📦 已归档：{f.name}")
    
    print("\n✨ 记忆整理完成！")

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    dry_run = "--dry-run" in sys.argv
    organize_memory(days, dry_run)
