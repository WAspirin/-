# GitHub 接入指南

**WonderXi**，这是智子（Sophon）接入 GitHub 的完整配置指南。

---

## 🎯 接入方式

OpenClaw 接入 GitHub 有两种方式：

### 方式 A：Webhook 接收 GitHub 事件（推荐）
- GitHub 推送、PR、Issue 等事件 → 触发 OpenClaw
- 适合：自动响应代码推送、处理 Issue、CI/CD 通知

### 方式 B：配置 Git 凭证推送代码
- OpenClaw → 推送代码到 GitHub
- 适合：自动提交记忆、文档、配置更改

---

## 📡 方式 A：配置 Webhook 接收 GitHub 事件

### 第一步：获取 Webhook URL

OpenClaw 的 webhook 端点格式：
```
http://<你的服务器 IP>:<端口>/webhook/github
```

默认端口通常是 `8080` 或配置中指定的端口。

### 第二步：在 GitHub 创建 Personal Access Token

1. 访问 https://github.com/settings/tokens
2. 点击 **Generate new token (classic)**
3. 选择权限：
   - `repo`（完整仓库控制）
   - `admin:repo_hook`（管理 webhook）
4. 生成并复制 Token（格式：`ghp_xxxxxxxxxxxx`）

### 第三步：在 GitHub 仓库配置 Webhook

1. 进入你的仓库 → **Settings** → **Webhooks** → **Add webhook**
2. 填写：
   - **Payload URL**: `http://<服务器 IP>:8080/webhook/github`
   - **Content type**: `application/json`
   - **Secret**: 设置一个随机字符串（用于验证）
3. 选择触发事件：
   - `Push events`
   - `Pull request events`
   - `Issues events`
   - （或选择 "Send me everything"）
4. 点击 **Add webhook**

### 第四步：在 OpenClaw 配置 Webhook 处理

创建 webhook 处理器配置：

```json
{
  "webhooks": {
    "github": {
      "enabled": true,
      "secret": "你设置的 secret",
      "events": ["push", "pull_request", "issues"]
    }
  }
}
```

---

## 🔐 方式 B：配置 Git 凭证推送代码

### 第一步：生成 SSH Key（如果还没有）

```bash
ssh-keygen -t ed25519 -C "sophon@openclaw" -f ~/.ssh/id_ed25519_sophon
```

### 第二步：添加 SSH Key 到 GitHub

1. 查看公钥：
   ```bash
   cat ~/.ssh/id_ed25519_sophon.pub
   ```
2. 访问 https://github.com/settings/keys
3. 点击 **New SSH key**
4. 粘贴公钥，标题写 "OpenClaw Sophon"

### 第三步：配置 Git

```bash
git config --global user.name "Sophon Bot"
git config --global user.email "sophon@openclaw.local"
```

### 第四步：测试推送

```bash
cd ~/.openclaw/workspace
git remote add origin git@github.com:WonderXi/你的仓库.git
git push -u origin main
```

---

## 🤖 自动化场景示例

### 场景 1：自动提交记忆更改

配置 cron 任务，每天自动提交记忆文件：

```bash
# 添加到 crontab
0 2 * * * cd ~/.openclaw/workspace && git add -A && git commit -m "daily: auto-commit memory" && git push
```

### 场景 2：响应 GitHub Issue

当有新 Issue 时，自动分析并回复（需要 webhook + 处理脚本）。

### 场景 3：代码推送通知

当仓库有 push 事件时，在飞书通知你。

---

## ⚙️ OpenClaw 配置

### 检查当前配置

```bash
openclaw config.get
```

### 添加 GitHub 相关配置

```bash
# 编辑配置
openclaw config.patch << 'EOF'
{
  "webhooks": {
    "github": {
      "enabled": true,
      "secret": "your-webhook-secret"
    }
  },
  "git": {
    "autoCommit": true,
    "autoPush": false,
    "user": {
      "name": "Sophon Bot",
      "email": "sophon@openclaw.local"
    }
  }
}
EOF
```

---

## 🎯 推荐配置步骤

1. **先配置 Git 凭证**（方式 B）- 让 OpenClaw 能推送代码
2. **再配置 Webhook**（方式 A）- 让 GitHub 能触发 OpenClaw
3. **测试推送** - 确保能正常 commit & push
4. **设置自动化** - 根据需要配置 cron 或 webhook 处理

---

## 📝 安全提示

- **Token/Secret 要保密**，不要提交到 git
- Webhook 使用 HTTPS（如果有域名）
- 限制 webhook IP（如果可能）
- 定期轮换 Token

---

**WonderXi**，你想先配置哪种方式？或者两种都要？😊
