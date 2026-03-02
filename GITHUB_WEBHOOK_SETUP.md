# GitHub Webhook 配置指南

## 🎯 目标

配置 GitHub Webhook，让智子能够：
- 自动响应 Issue 和 PR
- 接收 push 通知
- 分析代码变更
- 在飞书上通知你仓库动态

---

## 📋 配置步骤

### 第一步：获取 Webhook URL

OpenClaw 网关已经在运行，Webhook 接收地址是：

```
http://localhost:18789/webhook/github
```

**⚠️ 重要：** 如果你的服务器有公网 IP，使用：
```
http://<你的公网 IP>:18789/webhook/github
```

或者使用内网穿透工具（如 ngrok）：
```bash
ngrok http 18789
# 然后使用 ngrok 提供的 https 地址
```

### 第二步：在 GitHub 仓库添加 Webhook

1. **访问你的仓库**：https://github.com/WAspirin/-
2. **进入设置**：点击 Settings > Webhooks
3. **添加 Webhook**：点击 Add webhook
4. **填写配置**：

| 字段 | 值 |
|------|-----|
| Payload URL | `http://localhost:18789/webhook/github` |
| Content type | `application/json` |
| Secret | （可选）设置一个密钥，比如 `sophon-secret-2026` |
| SSL verification | 禁用（如果用 http） |
| Events | 选择以下事件： |

### 第三步：选择触发事件

推荐勾选以下事件：

- ✅ **Issues** - Issue 打开、关闭、评论
- ✅ **Pull requests** - PR 相关事件
- ✅ **Push** - 代码推送
- ✅ **Issue comments** - Issue 评论
- ✅ **Pull request review comments** - PR 评论

或者选择 **Let me select individual events** 自定义。

### 第四步：测试 Webhook

1. 点击 Add webhook 后，GitHub 会发送一个 ping 事件
2. 检查日志确认是否收到：
   ```bash
   tail -f /root/.openclaw/workspace/logs/github-webhook.log
   ```

---

## 🔧 高级配置

### 使用 Secret 验证

如果你在 GitHub 设置了 Secret，修改 handler 脚本：

```bash
# 在脚本开头添加
SECRET="sophon-secret-2026"
SIGNATURE="${GITHUB_HTTP_X_HUB_SIGNATURE_256:-}"

# 验证签名（需要安装 openssl）
EXPECTED="sha256=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" | cut -d' ' -f2)"
if [ "$SIGNATURE" != "sha256=$EXPECTED" ]; then
    echo "Invalid signature"
    exit 1
fi
```

### 使用 ngrok 暴露本地服务

```bash
# 安装 ngrok
curl -s https://ngrok-agent-s3.s3.amazonaws.com/ngrok.asc | \
  sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
  echo "deb https://ngrok-agent-s3.s3.amazonaws.com buster main" | \
  sudo tee /etc/apt/sources.list.d/ngrok.list && \
  sudo apt update && sudo apt install ngrok

# 启动 ngrok
ngrok http 18789

# 复制 https 地址，比如：https://xxx.ngrok.io
# 在 GitHub Webhook 中使用这个地址
```

---

## 📊 智子能做什么

配置完成后，智子可以：

| 事件 | 智子的响应 |
|------|-----------|
| **新 Issue** | 分析 Issue 内容，提供建议或分配标签 |
| **新 PR** | 自动 review 代码，检查变更 |
| **Push** | 分析提交内容，在飞书通知你 |
| **Issue 评论** | 参与讨论，提供帮助 |
| **Release** | 发布通知，更新 changelog |

---

## 🧪 测试

### 测试 Issue 响应

1. 在仓库创建一个新 Issue
2. 标题：`Test webhook - 测试`
3. 内容：`这是测试 Issue，看看智子会不会响应`
4. 检查日志和飞书通知

### 测试 Push 通知

```bash
cd ~/.openclaw/workspace
echo "test" >> test.txt
git add test.txt
git commit -m "test: webhook test"
git push
```

---

## ⚠️ 故障排查

### Webhook 不工作？

1. **检查网关是否运行**：
   ```bash
   openclaw gateway status
   ```

2. **检查日志**：
   ```bash
   tail -f /root/.openclaw/workspace/logs/github-webhook.log
   ```

3. **测试本地接收**：
   ```bash
   curl -X POST http://localhost:18789/webhook/github \
     -H "Content-Type: application/json" \
     -d '{"test": true}'
   ```

4. **检查防火墙**：
   ```bash
   # 确保 18789 端口开放
   sudo ufw allow 18789
   ```

---

## 🎯 下一步

配置完成后，智子可以：
1. 自动分析 GitHub Issue 并回复
2. 监控代码推送并通知你
3. 参与 PR review
4. 定期同步仓库到本地

---

**配置完成后告诉我，我们测试一下！** 🚀
