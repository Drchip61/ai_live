# prompts 模块计划

## 模块职责

提供通用提示词加载功能，通过委托 PersonaLoader 获取角色信息，组装完整的系统提示词。

## 文件结构

```
prompts/
├── __init__.py           # 模块导出
├── prompt_loader.py      # PromptLoader类（委托 PersonaLoader）
└── base_instruction.txt  # 主播基础指令
```

## 核心类: PromptLoader

内部持有 PersonaLoader 实例，负责组装 base_instruction + 角色提示词。

### 方法

| 方法 | 说明 |
|------|------|
| `load(filename)` | 加载指定提示词文件 |
| `get_base_instruction()` | 获取基础指令 |
| `get_full_system_prompt(persona)` | 基础指令 + 角色提示词（委托 PersonaLoader） |
| `list_personas()` | 列出所有可用角色（委托 PersonaLoader） |

### 使用示例

```python
from prompts import PromptLoader

loader = PromptLoader()
system_prompt = loader.get_full_system_prompt("karin")
personas = loader.list_personas()  # ["karin", "kuro", "sage"]
```

### 与 PersonaLoader 的关系

```
PromptLoader
├── 自身: 管理 prompts/ 下的通用文件（base_instruction.txt）
└── 委托 PersonaLoader: 获取角色专属 system_prompt
```

LLMWrapper 只依赖 PromptLoader，不直接使用 PersonaLoader。

## base_instruction.txt 内容

通用主播指令，包含：
1. **保持角色一致性** — 始终以主播身份说话
2. **积极互动** — 及时、热情回应弹幕
3. **控制回复长度** — 简洁，不超过 3 句话
4. **口语化表达** — 自然说话
5. **适当语气词** — "哈哈"、"嘿嘿"等增加亲和力
6. **时间观念** — 注意相对时间标注，不在回复中说出具体秒数
7. **冷场应对** — 看到沉默提示时，从多种策略中选择：
   - 延续上一个话题、自言自语式分享、抛出新话题
   - 日常感慨、回忆互动
   - 避免反复问"有没有人啊"，要说有内容的话

互动技巧：记住常互动观众、感谢礼物、抛出话题、化解不友善弹幕

## 状态

- [x] 基础指令编写（含时间观念）
- [x] PromptLoader类实现
- [x] 与 PersonaLoader 集成
