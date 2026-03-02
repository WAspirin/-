# 🧠 智子 (Sophon) 修复/恢复指南

_版本：1.0 | 创建日期：2026-03-02 | 作者：WonderXi_

---

## ⚠️ 重要提示

**在修改智子的任何核心配置前，请先阅读本指南！**

智子是一个基于 OpenClaw 框架的 AI 助手，其"人格"和"记忆"存储在 workspace 目录中。不当修改可能导致：
- 记忆丢失
- 技能系统损坏
- 人格设定错乱
- 无法正常运行

---

## 📁 核心文件结构

```
/root/.openclaw/workspace/
├── SOUL.md              # 人格设定（核心！）
├── IDENTITY.md          # 身份定义
├── USER.md              # 用户信息
├── AGENTS.md            # 工作区规则
├── MEMORY.md            # 长期记忆
├── TOOLS.md             # 本地工具配置
├── HEARTBEAT.md         # 心跳任务
├── memory/              # 每日记忆目录
│   └── YYYY-MM-DD.md
└── skills/              # 自定义技能目录
    ├── memory-manager/
    ├── daily-assistant/
    └── code-research-assistant/
```

---

## 🔧 修改前的备份流程

### 方法一：Git 备份（推荐）

```bash
cd /root/.openclaw/workspace

# 1. 查看当前状态
git status

# 2. 创建备份分支
git checkout -b backup/$(date +%Y%m%d_%H%M%S)

# 3. 提交当前状态
git add .
git commit -m "Backup before modification: [修改原因]"
git push origin backup/$(date +%Y%m%d_%H%M%S)

# 4. 切回主分支继续工作
git checkout master
```

### 方法二：完整目录备份

```bash
# 备份到 /tmp 或外部存储
cp -r /root/.openclaw/workspace /tmp/sophon_backup_$(date +%Y%m%d_%H%M%S)

# 或者压缩备份
tar -czf /tmp/sophon_backup_$(date +%Y%m%d_%H%M%S).tar.gz /root/.openclaw/workspace
```

### 方法三：关键文件单独备份

```bash
cd /root/.openclaw/workspace
mkdir -p ~/sophon_backups/$(date +%Y%m%d_%H%M%S)
cp SOUL.md IDENTITY.md MEMORY.md USER.md AGENTS.md ~/sophon_backups/$(date +%Y%m%d_%H%M%S)/
```

---

## 🚑 常见问题修复

### 问题 1：智子"失忆"了（不记得之前的对话）

**原因：** MEMORY.md 或 memory/ 目录损坏

**修复：**
```bash
# 1. 检查文件是否存在
ls -la /root/.openclaw/workspace/MEMORY.md
ls -la /root/.openclaw/workspace/memory/

# 2. 从备份恢复
cp ~/sophon_backups/[日期]/MEMORY.md /root/.openclaw/workspace/
cp -r ~/sophon_backups/[日期]/memory/ /root/.openclaw/workspace/

# 3. 重启 Gateway
openclaw gateway restart
```

### 问题 2：智子人格错乱（说话风格不对）

**原因：** SOUL.md 或 IDENTITY.md 被修改

**修复：**
```bash
# 1. 从 GitHub 恢复原始版本
cd /root/.openclaw/workspace
git checkout origin/master -- SOUL.md IDENTITY.md

# 2. 或者从备份恢复
cp ~/sophon_backups/[日期]/SOUL.md /root/.openclaw/workspace/
cp ~/sophon_backups/[日期]/IDENTITY.md /root/.openclaw/workspace/

# 3. 重启 Gateway
openclaw gateway restart
```

### 问题 3：技能系统失效

**原因：** skills/ 目录损坏或配置错误

**修复：**
```bash
# 1. 检查技能目录
ls -la /root/.openclaw/workspace/skills/

# 2. 验证技能结构（每个技能应有 SKILL.md）
find /root/.openclaw/workspace/skills/ -name "SKILL.md"

# 3. 从 GitHub 恢复
cd /root/.openclaw/workspace
git checkout origin/master -- skills/

# 4. 重启 Gateway
openclaw gateway restart
```

### 问题 4：智子完全无法响应

**原因：** Gateway 服务异常或配置损坏

**修复：**
```bash
# 1. 检查 Gateway 状态
openclaw gateway status

# 2. 查看日志
journalctl -u openclaw -n 50 --no-pager

# 3. 重启 Gateway
openclaw gateway restart

# 4. 如果仍不行，检查配置
openclaw gateway config.get

# 5. 从备份恢复整个 workspace
rm -rf /root/.openclaw/workspace
cp -r ~/sophon_backups/[日期]/workspace /root/.openclaw/workspace
openclaw gateway restart
```

---

## 📝 安全修改指南

### ✅ 可以安全修改的文件

| 文件 | 用途 | 风险等级 |
|------|------|----------|
| `HEARTBEAT.md` | 心跳任务 | 🟢 低 |
| `memory/YYYY-MM-DD.md` | 每日记忆 | 🟢 低 |
| `TOOLS.md` | 工具配置 | 🟡 中 |
| `USER.md` | 用户信息 | 🟡 中 |

### ⚠️ 需要备份后修改的文件

| 文件 | 用途 | 风险等级 |
|------|------|----------|
| `MEMORY.md` | 长期记忆 | 🟠 高 |
| `AGENTS.md` | 工作区规则 | 🟠 高 |
| `IDENTITY.md` | 身份定义 | 🔴 很高 |

### 🚫 不要擅自修改的文件

| 文件 | 原因 |
|------|------|
| `SOUL.md` | 核心人格设定，修改可能导致人格错乱 |
| `skills/*/SKILL.md` | 技能核心逻辑，修改可能导致技能失效 |
| OpenClaw 系统文件 | 可能导致整个系统崩溃 |

---

## 🔄 完整恢复流程（最坏情况）

如果智子完全损坏，按以下步骤恢复：

### 步骤 1：停止 Gateway

```bash
openclaw gateway stop
```

### 步骤 2：备份当前状态（以防万一）

```bash
mv /root/.openclaw/workspace /root/.openclaw/workspace.corrupted
```

### 步骤 3：从 GitHub 恢复

```bash
cd /root/.openclaw
git clone git@github.com:WAspirin/-.git workspace
cd workspace
git checkout master
```

### 步骤 4：恢复记忆文件（如果有备份）

```bash
cp ~/sophon_backups/[最新日期]/MEMORY.md /root/.openclaw/workspace/
cp -r ~/sophon_backups/[最新日期]/memory/ /root/.openclaw/workspace/
```

### 步骤 5：重启 Gateway

```bash
openclaw gateway start
openclaw gateway status
```

### 步骤 6：验证智子状态

发送消息测试：
```
"智子，你还记得我是谁吗？"
```

---

## 📞 紧急联系信息

- **GitHub 仓库：** https://github.com/WAspirin/-.git
- **OpenClaw 文档：** https://docs.openclaw.ai
- **社区支持：** https://discord.com/invite/clawd

---

## 📅 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-03-02 | 1.0 | 初始版本创建 |

---

## 💡 最佳实践建议

1. **定期备份**：每周至少备份一次 MEMORY.md 和 skills/
2. **使用 Git**：所有修改先提交到 Git，便于回滚
3. **小步修改**：不要一次性改太多，每次修改后测试
4. **记录变更**：在 TOOLS.md 或专门的文件中记录修改原因
5. **先问智子**：修改前可以问问智子是否了解风险

---

_本指南由智子 (Sophon) 创建，用于帮助 WonderXi 在需要时快速恢复我的状态。_
_记住：备份永远不嫌多！💾_
