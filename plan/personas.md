# personas 模块计划

## 模块职责

角色人格管理，每个角色一个独立子目录，包含系统提示词和预设记忆等角色数据。

## 文件结构

```
personas/
├── __init__.py
├── persona_loader.py       # PersonaLoader 核心类
├── karin/                  # 元气偶像少女
│   ├── system_prompt.txt   # 角色专属系统提示词
│   └── static_memories/    # 预设记忆
│       └── identity.json
├── sage/                   # 知性学者
│   ├── system_prompt.txt
│   └── static_memories/
│       └── identity.json
└── kuro/                   # 酷酷游戏主播
    ├── system_prompt.txt
    └── static_memories/
        └── identity.json
```

## 核心类: PersonaLoader

自动发现 personas/ 下的角色子目录，加载角色相关文件。

### 方法

| 方法 | 说明 |
|------|------|
| `list_personas()` | 自动发现所有角色子目录（排序后返回） |
| `load(persona, filename)` | 加载角色目录下的任意文件 |
| `get_system_prompt(persona)` | 获取角色专属系统提示词 |

### 使用示例

```python
from personas import PersonaLoader

loader = PersonaLoader()
personas = loader.list_personas()       # ["karin", "kuro", "sage"]
prompt = loader.get_system_prompt("karin")
```

## 预设记忆格式 (static_memories/*.json)

```json
[
  {"content": "我叫花凛，今年17岁", "category": "identity"},
  {"content": "我表面上很酷很冷淡，但其实内心很热情", "category": "personality"}
]
```

支持的 category: `identity`, `personality`, `preference`, `hobby`, `skill`, `habit`, `daily`, `experience`, `relationship`, `dream`

## 谁使用 PersonaLoader

- **PromptLoader** — 通过 `get_system_prompt()` 获取角色提示词
- **StaticLayer (memory)** — 直接读取 `personas/{角色}/static_memories/` 目录
- 未来其他模块可通过 `load()` 获取角色的任意文件

## 状态

- [x] PersonaLoader 实现
- [x] 3 个角色的 system_prompt.txt
- [x] 3 个角色的 static_memories/identity.json（每角色 22 条，覆盖多种 category）
