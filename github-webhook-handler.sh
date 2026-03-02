#!/bin/bash
# GitHub Webhook Handler for OpenClaw
# 接收 GitHub webhook 事件并触发智子的响应

# 日志文件
LOG_FILE="/root/.openclaw/workspace/logs/github-webhook.log"

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

# 记录请求
echo "[$(date -Iseconds)] Received webhook: $1" >> "$LOG_FILE"

# 读取 webhook payload
PAYLOAD=$(cat)

# 解析事件类型
EVENT_TYPE="${GITHUB_EVENT_TYPE:-unknown}"
ACTION="${GITHUB_EVENT_ACTION:-}"
REPO="${GITHUB_REPOSITORY:-unknown}"

# 记录详细信息
echo "[$(date -Iseconds)] Event: $EVENT_TYPE, Action: $ACTION, Repo: $REPO" >> "$LOG_FILE"

# 根据事件类型触发不同的响应
case "$EVENT_TYPE" in
  issues)
    echo "[$(date -Iseconds)] Issue event detected - notifying Sophon" >> "$LOG_FILE"
    # 可以这里调用 openclaw 命令或发送消息
    ;;
  pull_request)
    echo "[$(date -Iseconds)] PR event detected - notifying Sophon" >> "$LOG_FILE"
    ;;
  push)
    echo "[$(date -Iseconds)] Push event detected - notifying Sophon" >> "$LOG_FILE"
    ;;
  *)
    echo "[$(date -Iseconds)] Unknown event type: $EVENT_TYPE" >> "$LOG_FILE"
    ;;
esac

# 返回成功响应
echo "OK"
