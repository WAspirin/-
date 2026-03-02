#!/usr/bin/env python3
"""
技能打包脚本 - 将智子的技能打包成 .skill 文件
使用方法：python package_skills.py [skill-name]
"""

import os
import sys
import zipfile
from pathlib import Path
from datetime import datetime

SKILLS_DIR = Path("/root/.openclaw/workspace/skills")
OUTPUT_DIR = Path("/root/.openclaw/workspace/skills/dist")

def validate_skill(skill_path):
    """验证技能目录结构"""
    errors = []
    
    # 检查 SKILL.md 是否存在
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        errors.append(f"缺少 SKILL.md 文件")
        return errors
    
    # 检查 frontmatter
    with open(skill_md, 'r', encoding='utf-8') as f:
        content = f.read()
        if not content.startswith('---'):
            errors.append("SKILL.md 缺少 YAML frontmatter")
        if 'name:' not in content:
            errors.append("SKILL.md 缺少 name 字段")
        if 'description:' not in content:
            errors.append("SKILL.md 缺少 description 字段")
    
    # 检查是否有 symlinks（不允许）
    for root, dirs, files in os.walk(skill_path):
        for f in files:
            file_path = Path(root) / f
            if file_path.is_symlink():
                errors.append(f"发现不允许的 symlink: {file_path}")
    
    return errors

def package_skill(skill_name):
    """打包单个技能"""
    skill_path = SKILLS_DIR / skill_name
    
    if not skill_path.exists():
        print(f"❌ 技能目录不存在：{skill_path}")
        return False
    
    # 验证
    print(f"🔍 验证技能：{skill_name}")
    errors = validate_skill(skill_path)
    if errors:
        print(f"❌ 验证失败:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print(f"✅ 验证通过")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 打包
    output_file = OUTPUT_DIR / f"{skill_name}.skill"
    print(f"📦 打包到：{output_file}")
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(skill_path):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(SKILLS_DIR.parent)
                zipf.write(file_path, arcname)
    
    print(f"✨ 打包完成：{output_file.name}")
    return True

def package_all():
    """打包所有技能"""
    if not SKILLS_DIR.exists():
        print("❌ 技能目录不存在")
        return
    
    skills = [d for d in SKILLS_DIR.iterdir() if d.is_dir() and d.name != 'dist']
    
    if not skills:
        print("📭 没有找到技能")
        return
    
    print(f"🎯 找到 {len(skills)} 个技能")
    
    success_count = 0
    for skill in skills:
        if package_skill(skill.name):
            success_count += 1
        print()
    
    print(f"📊 打包完成：{success_count}/{len(skills)} 个技能成功")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 打包指定技能
        skill_name = sys.argv[1]
        package_skill(skill_name)
    else:
        # 打包所有技能
        package_all()
