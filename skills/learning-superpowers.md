# GitHub 技能学习笔记 - Superpowers

**学习日期**: 2026-03-08  
**仓库**: `obra/superpowers` (73.6k stars)  
**语言**: Shell 76.4%, JavaScript 12.4%, Python 5.7%

---

## 📚 核心概念

### 什么是 Superpowers？
> 一个完整的软件开发工作流系统，基于可组合的"技能"库，让编码 Agent 自动遵循最佳实践。

### 核心工作流

1. **Brainstorming** (头脑风暴)
   - 写代码前激活
   - 通过问题细化需求
   - 分块展示设计供验证
   - 保存设计文档

2. **Using Git Worktrees** (使用 Git 工作树)
   - 设计批准后激活
   - 创建独立分支
   - 运行项目设置
   - 验证测试基线

3. **Writing Plans** (编写计划)
   - 将工作分解为小任务 (2-5 分钟/个)
   - 每个任务包含：文件路径、完整代码、验证步骤

4. **Subagent-Driven Development** (子代理驱动开发)
   - 每个任务分配新鲜子代理
   - 两阶段审查 (规范合规 → 代码质量)
   - 或批量执行 + 人工检查点

5. **Test-Driven Development** (测试驱动)
   - RED-GREEN-REFACTOR 循环
   - 先写失败测试 → 写最小代码 → 重构
   - 删除测试前写的代码

6. **Requesting Code Review** (请求代码审查)
   - 任务间激活
   - 按严重性报告问题
   - 关键问题阻塞进度

7. **Finishing a Development Branch** (完成分支)
   - 验证测试
   - 提供选项 (合并/PR/保留/丢弃)
   - 清理工作树

---

## 💡 核心哲学

1. **Test-Driven Development** - 先写测试，永远
2. **Systematic over ad-hoc** - 流程优于猜测
3. **Complexity reduction** - 简单性为首要目标
4. **Evidence over claims** - 验证后再声明成功

---

## 🔧 技能库分类

### Testing (测试)
- `test-driven-development` - RED-GREEN-REFACTOR 循环

### Debugging (调试)
- `systematic-debugging` - 4 阶段根因分析
- `verification-before-completion` - 确保真正修复

### Collaboration (协作)
- `brainstorming` - Socratic 设计细化
- `writing-plans` - 详细实现计划
- `executing-plans` - 批量执行 + 检查点
- `dispatching-parallel-agents` - 并发子代理
- `requesting-code-review` - 预审查清单
- `receiving-code-review` - 响应反馈
- `using-git-worktrees` - 并行开发分支
- `finishing-a-development-branch` - 合并/PR 决策
- `subagent-driven-development` - 快速迭代 + 两阶段审查

### Meta (元技能)
- `writing-skills` - 创建新技能指南
- `using-superpowers` - 技能系统介绍

---

## 🎯 关键洞察

### 1. 技能触发机制
- **自动触发**: Agent 在执行任何任务前检查相关技能
- **强制性**: 是强制工作流，不是建议
- **上下文感知**: 根据当前任务自动选择技能

### 2. 子代理管理
- **新鲜启动**: 每个任务用新子代理 (无偏见)
- **两阶段审查**:
  1. 规范合规性检查
  2. 代码质量检查
- **批量执行**: 可批量处理任务 + 人工检查点

### 3. TDD 强制执行
- **硬性规定**: 测试必须先写
- **自动删除**: 删除测试前的代码
- **RED-GREEN-REFACTOR**: 完整循环

### 4. Git 工作流
- **Worktrees**: 并行开发分支
- **隔离环境**: 每个任务独立分支
- **自动清理**: 完成后清理工作树

---

## 📦 技术实现

### 目录结构
```
superpowers/
├── .claude-plugin/     # Claude Code 插件
├── .cursor-plugin/     # Cursor 插件
├── .codex/             # Codex 配置
├── .opencode/          # OpenCode 配置
├── commands/           # Slash 命令
├── hooks/              # Git hooks
├── lib/                # 核心库 (ESM 模块)
├── skills/             # 技能库
└── tests/              # 测试
```

### 安装方式
- **Claude Code**: `/plugin install superpowers@superpowers-marketplace`
- **Cursor**: `/plugin-add superpowers`
- **Codex**: 手动安装 (follow INSTALL.md)
- **OpenCode**: 手动安装 (follow INSTALL.md)

### 更新机制
- 自动更新：`/plugin update superpowers`
- 技能直接从仓库加载

---

## 🚀 可借鉴的设计

### 1. 技能组合模式
```yaml
技能触发 → 上下文感知 → 自动执行 → 结果验证
```

**应用**: 电缆布线优化可以借鉴
- 根据问题规模自动选择算法
- 强制执行验证步骤
- 自动记录实验结果

### 2. 子代理管理
```yaml
任务分解 → 新鲜子代理 → 两阶段审查 → 批量执行
```

**应用**: 大规模布线问题
- 分解为子问题
- 并行求解
- 结果合并 + 验证

### 3. 强制 TDD
```yaml
先写测试 → 运行失败 → 写代码 → 运行通过 → 提交
```

**应用**: 算法实现
- 先写测试用例
- 强制验证正确性
- 自动回归测试

### 4. Git 工作流
```yaml
设计批准 → 创建工作树 → 实现 → 审查 → 合并 → 清理
```

**应用**: 实验管理
- 每个实验独立分支
- 自动记录配置
- 结果可复现

---

## 💡 应用到当前项目

### 短期 (本周)
1. **技能触发机制**
   - 根据问题规模自动选择算法
   - 强制执行验证步骤

2. **子代理管理**
   - 实现并行算法求解
   - 结果对比 + 验证

### 中期 (本月)
1. **TDD 强制**
   - 为所有算法写测试
   - 自动回归测试

2. **Git 工作流**
   - 每个实验独立分支
   - 自动记录配置

### 长期 (3 个月)
1. **完整技能系统**
   - 创建电缆布线专用技能
   - 自动触发最佳实践

2. **开源贡献**
   - 发布技能到 Superpowers 市场
   - 建立影响力

---

## 📊 仓库统计

| 指标 | 数值 |
|------|------|
| Stars | 73.6k |
| Forks | 5.7k |
| Contributors | 19 |
| 语言 | Shell 76.4%, JS 12.4%, Python 5.7% |
| 最新版本 | v4.3.1 (2026-02-22) |
| License | MIT |

---

## 🔗 参考资源

- **仓库**: https://github.com/obra/superpowers
- **博客**: https://blog.fsck.com/2025/10/09/superpowers/
- **市场**: https://github.com/obra/superpowers-marketplace
- **文档**: `docs/` 目录

---

**学习者**: 智子 (Sophon)  
**学习日期**: 2026-03-08  
**下一个技能**: Frontend-design (40.8k stars)

_技能系统设计精妙，值得深入学习！🚀_
