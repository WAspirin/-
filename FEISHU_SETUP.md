# 飞书配置指南 - 智子专属

**创建时间：** 2026-03-02  
**状态：** 插件已启用，等待配置凭证

---

## 📋 当前状态

- ✅ 飞书插件已安装并启用
- ✅ 网关正在运行 (port 18789)
- ⏳ 等待配置 App ID 和 App Secret

---

## 🎯 配置步骤（简化版）

### 第一步：在飞书开放平台创建应用

1. **访问**：https://open.feishu.cn/app
   - 国际版用户用：https://open.larksuite.com/app

2. **创建企业应用**
   - 应用名称：`智子助手`（或者你喜欢的名字）
   - 选个好看的图标

3. **复制凭证**（重要！）
   - 进入"凭证与基础信息"页面
   - 复制以下内容：
     - **App ID**（格式：`cli_xxx`）
     - **App Secret**（保密！）

4. **配置权限**
   - 点击"批量导入"，粘贴以下权限：
   ```json
   {
     "scopes": {
       "tenant": [
         "im:message",
         "im:message:send_as_bot",
         "im:message:readonly",
         "im:chat.access_event.bot_p2p_chat:read",
         "application:bot.menu:write"
       ]
     }
   }
   ```

5. **启用机器人**
   - 进入"应用功能" > "机器人"
   - 启用机器人能力
   - 设置机器人名字：`智子`

6. **配置事件订阅**
   - 选择"使用长连接接收事件"（WebSocket）
   - 添加事件：`im.message.receive_v1`

7. **发布应用**
   - 创建版本并提交审核（企业应用通常自动通过）

### 第二步：在 OpenClaw 中配置

**方法 A：使用交互式命令（推荐）**
```bash
openclaw channels add
```
- 选择 **Feishu**
- 粘贴你的 App ID 和 App Secret

**方法 B：手动编辑配置文件**
```bash
# 编辑配置文件
nano ~/.openclaw/openclaw.json
```

在配置文件中添加 channels 部分（如果已有就追加）：
```json
{
  "channels": {
    "feishu": {
      "appId": "你的 App ID",
      "appSecret": "你的 App Secret",
      "domain": "feishu"
    }
  }
}
```

### 第三步：重启网关并配对

```bash
# 重启网关使配置生效
openclaw gateway restart

# 查看日志
openclaw logs --follow
```

然后在飞书里找到你的机器人，发送一条消息。首次连接会收到一个**配对码**，用这个命令批准：

```bash
openclaw pairing approve feishu <配对码>
```

---

## 🔐 安全提醒

1. **App Secret 是机密**，不要：
   - 提交到 git
   - 发给别人
   - 放在公开的地方

2. **配对码是一次性的**，只在首次连接时需要

3. **群聊时**，默认需要 @机器人 才会响应（避免打扰其他人）

---

## 🎉 配置完成后的测试

1. 在飞书里给"智子"发送消息
2. 应该能看到我的回复
3. 如果没反应，检查日志：`openclaw logs --follow`

---

## 📞 需要帮助？

如果遇到问题：
1. 查看日志：`openclaw logs --follow`
2. 检查网关状态：`openclaw gateway status`
3. 查看文档：https://docs.openclaw.ai/channels/feishu

---

**备注：** 这个文件是智子为 WonderXi 创建的专属配置指南。配置完成后可以删除或保留作为参考。
