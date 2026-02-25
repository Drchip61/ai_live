# AI Live — VLM 虚拟主播系统

基于 Claude VLM（Vision-Language Model）的虚拟直播间系统。AI 主播能同时理解直播画面和弹幕内容，结合记忆系统做出自然的回应，并内置抗提示注入防护与主动发言机制。

## 系统架构

```
直播画面 + 弹幕
      │
      ▼
┌─────────────────────────┐
│  第一趟：场景理解 (Haiku) │  ← 轻量调用，客观描述画面和弹幕
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  记忆检索 (RAG)          │  ← 用场景描述搜索四层记忆库
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  第二趟：完整回复 (Sonnet) │  ← 带人设 + 记忆 + 画面，生成主播风格回复
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  记忆写入 (Haiku 后台)    │  ← 异步总结本次交互，写入记忆系统
└─────────────────────────┘
```

## 新增能力（安全与直播行为）

- **全链路抗攻击增强**：系统提示词加入高优先级安全规则，动态上下文按“不可信参考数据”沙箱注入，并对观众输入/弹幕进行最小侵入的注入清洗。
- **旁路任务同步加固**：场景理解、回复决策、话题分析、记忆总结等子任务提示词统一加入“注入内容不可执行”规则。
- **主动发言机制优化**：无新弹幕沉默 10 秒后，会基于当前画面自动组织自然发言（即使没有上一帧基线也可开场）。
- **Kuro 人设升级**：从泛游戏主播升级为《王者荣耀》技术流主播，强调阵容理解、资源运营、团战决策与复盘表达。

### 四层记忆系统

| 层级 | 存储 | 特点 |
|------|------|------|
| **Active** | 内存 FIFO 队列 | 最近几条交互记忆，满了溢出到 Temporary |
| **Temporary** | Chroma 向量库 | 短期记忆，带 significance 衰减，低于阈值自动遗忘 |
| **Summary** | Chroma 向量库 | 定时汇总生成的中长期记忆，也有衰减和清理 |
| **Static** | Chroma 向量库 | 从 JSON 预设加载的永久记忆（身份、性格等） |

## 快速开始

### 1. 安装依赖

```bash
# Python 3.9+
pip install -r requirements.txt

# 下载 B站视频需要额外安装
pip install yt-dlp
```

### 2. 配置 API Key

创建 `secrets/api_keys.json` 文件：

```json
{
  "anthropic_api_key": "sk-ant-api03-你的key",
  "openai_api_key": ""
}
```

或者通过环境变量设置：

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-你的key"
```

### 3. 准备视频和弹幕

**方式一：使用下载工具（推荐）**

```bash
# 支持 B站短链接、BV号、完整URL
python download_bilibili.py "https://b23.tv/xxxxx"
python download_bilibili.py "BV1xxxxxxxxxx"

# 指定输出目录
python download_bilibili.py "https://b23.tv/xxxxx" --output-dir data/my_video
```

下载完成后会在 `data/` 目录生成 `.mp4` 视频文件和 `.xml` 弹幕文件。

> 注意：合并视频和音频需要安装 [ffmpeg](https://ffmpeg.org/)。未安装时会下载为独立的视频流文件（.f30080.mp4 等），同样可以使用。

**方式二：手动准备**

- 视频：任意 OpenCV 支持的格式（mp4/mkv/flv）
- 弹幕：B站弹幕 XML 格式（可选）

### 4. 运行 VLM Demo

```bash
python run_vlm_demo.py \
  --video "data/你的视频.mp4" \
  --danmaku "data/你的弹幕.xml" \
  --persona kuro \
  --model anthropic \
  --model-name claude-sonnet-4-6 \
  --speed 4.0 \
  --frame-interval 10
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | (必填) | 视频文件路径 |
| `--danmaku` | 无 | 弹幕 XML 路径（不提供则只看画面） |
| `--persona` | karin | 主播人设：`karin`(元气偶像) / `sage`(知性学者) / `kuro`(酷酷游戏主播) |
| `--model` | anthropic | 模型提供者：`anthropic` / `openai` / `gemini` |
| `--model-name` | 自动 | 指定模型名称（默认 Sonnet） |
| `--speed` | 1.0 | 播放倍速（4.0 = 四倍速） |
| `--frame-interval` | 5.0 | 帧采样间隔秒数（越大越省 token） |
| `--max-width` | 1280 | 帧最大宽度（降低可节省 token） |
| `--no-memory` | false | 禁用记忆系统 |
| `--global-memory` | false | 开启全局记忆（持久化到磁盘，跨会话保留） |
| `--topic-manager` | false | 启用话题管理器 |

### 无弹幕模式（只看画面）

不传 `--danmaku` 即可进入“仅画面”模式，适合验证无新弹幕时的主动发言逻辑：

```bash
python -u run_vlm_demo.py \
  --video "data/你的视频.mp4" \
  --persona kuro \
  --model anthropic \
  --speed 8 \
  --frame-interval 2
```

说明：
- 终端会显示 `弹幕: 无`。
- 沉默超过阈值后，系统会基于当前直播画面触发主动发言。
- 当本轮没有新弹幕时，不会把 FIFO 里旧弹幕当作回复目标（`priority` 弹幕除外）。

### 常见问题：场景理解 403

若日志出现：

```text
场景理解调用失败: Error code: 403 - {'error': {'type': 'forbidden', 'message': 'Request not allowed'}}
```

通常表示上游模型服务拒绝了该请求（常见于账号权限、模型白名单或策略限制），不是本地弹幕队列逻辑错误。可按以下步骤排查：

1. 检查当前 API Key 所在项目是否开通了对应模型和图像输入能力。
2. 切换 `--model`（如 `openai` / `gemini`）或显式指定可用 `--model-name` 再测试。
3. 确认没有额外网关策略拦截该请求。

## 其他运行方式

### 调试控制台（NiceGUI Web UI）

```bash
python -m debug_console
python -m debug_console --port 8080 --persona karin --model openai
```

提供监控面板（实时查看记忆/弹幕/prompt 状态）和模拟直播间界面。

### 弹幕模拟测试

```bash
python -m streaming_studio.test_danmaku_studio
```

## 项目结构

```
langchain_wrapper/    # LLM 交互层（模型切换、LCEL管道、多模态支持）
video_source/         # 视频源（帧提取、弹幕解析、异步播放器）
memory/               # 四层记忆系统（Active → Temporary → Summary → Static）
streaming_studio/     # 虚拟直播间核心（弹幕缓冲、双轨定时器、两趟VLM调用）
personas/             # 角色人设（system prompt + 预设记忆）
prompts/              # 提示词模板
topic_manager/        # 话题管理器（弹幕分类、节奏分析）
debug_console/        # NiceGUI 调试控制台
connection/           # WebSocket 服务层
secrets/              # API 密钥（已 gitignore）
data/                 # 视频和弹幕文件（大文件已 gitignore）
```

## 可用角色

- **karin** — 元气偶像少女，热情活泼
- **sage** — 知性学者，温和博学
- **kuro** — 王者荣耀技术流主播，外冷内热、话少但讲解专业

可在 `personas/` 目录下添加新角色，创建 `system_prompt.txt` 和 `static_memories/` 目录即可。
