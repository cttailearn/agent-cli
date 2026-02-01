# agent-cli（skills-dev）

一个基于 LangChain 的本地命令行智能体运行器：支持交互式对话 / 单次执行，按需加载 Skills（`SKILL.md`），并可通过 MCP（Model Context Protocol）把外部工具接入到智能体工具箱中。

## 功能

- **交互式 CLI**：直接在终端里与智能体对话；支持内置命令管理会话与文件操作。
- **单次执行模式**：传入一段 prompt，输出模型回复后退出。
- **Skills 机制**：扫描 `skills/`（以及可选的 `~/.agents/skills`），将每个 Skill 的描述注入系统提示词；需要细节时由模型调用 `load_skill(name)` 按需加载完整 `SKILL.md`。
- **受控的项目读写与命令执行**：内置工具限制在项目目录/工作目录/输出目录范围内读写与执行，避免越权访问。
- **MCP 工具接入**：通过 `mcp/config.json` 动态加载 MCP Server 暴露的工具（stdio 或 streamable_http）。

## 快速开始

### 1) 环境要求

- Python **>= 3.12**（见 [pyproject.toml](pyproject.toml)）
- 可选：推荐使用 [uv](https://github.com/astral-sh/uv) 管理虚拟环境与依赖

### 2) 安装依赖

使用 uv：

```bash
uv venv
uv pip install -e .
```

或使用 venv + pip：

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

### 3) 配置模型

程序会自动加载当前目录下的 `.env`（`python-dotenv`），并读取模型相关环境变量：

- `LC_MODEL`：模型名（默认 `deepseek:deepseek-reasoner`）

不同模型供应商所需的 API Key 由对应的 LangChain provider 决定。常见示例（仅供参考）：

```bash
# DeepSeek（若你选择 deepseek:* 模型）
# Windows
setx DEEPSEEK_API_KEY "YOUR_KEY"
# macOS/Linux
export DEEPSEEK_API_KEY="YOUR_KEY"

# OpenAI（若你选择 openai:* 模型）
# Windows
setx OPENAI_API_KEY "YOUR_KEY"
# macOS/Linux
export OPENAI_API_KEY="YOUR_KEY"
```

### 4) 运行

交互模式：

```bash
python main.py
```

单次执行：

```bash
python main.py "帮我列出当前项目的目录结构，并说明每个目录用途"
```

指定模型与目录：

```bash
python main.py --model deepseek:deepseek-chat --project-dir . --work-dir . --output-dir out
```

## 使用说明

### 运行参数

入口见 [main.py](main.py)：

- `--skills-dir`：skills 目录（默认 `skills`）
- `--project-dir`：项目根目录（默认脚本所在目录）
- `--work-dir`：命令执行工作目录（默认 `project-dir`）
- `--output-dir`：智能体生成文件输出目录（默认 `./out`）
- `--model`：模型名称（默认读取 `LC_MODEL`）
- `--mcp-config`：MCP 配置文件（默认 `mcp/config.json`）
- `prompt`：可选。传入则单次执行；不传则进入交互模式

同名环境变量也会生效：

- `AGENT_PROJECT_DIR`
- `AGENT_WORK_DIR`
- `AGENT_OUTPUT_DIR`
- `AGENT_MCP_CONFIG`

### 交互模式内置命令

内置命令由 [terminal_display.py](terminal_display.py) 实现，常用如下：

- `/help`：显示帮助
- `/skills`：显示已发现的技能目录
- `/tools`：显示可用工具概览
- `/pwd`：显示 cwd / project / work / output
- `/cd <path>`：切换工作目录（限制在项目目录内）
- `/ls [path]`、`/lsr [path]`：列出目录（可递归）
- `/rm <path>`、`/rmr <path>`：删除文件/目录（可递归）
- `/history [n]`：查看历史输入
- `/reset`：清空对话与历史
- `/quit`：退出
- `!<command>`：直接执行命令（在 `work_dir` 下）

## Skills

### Skills 扫描规则

启动时会扫描：

- `--skills-dir` 指定目录（默认 `./skills`）
- 可选的用户级全局目录：`~/.agents/skills`（存在且不同于项目目录时会追加）

每个 Skill 以 `SKILL.md` 为入口（支持 front-matter），目录结构通常如下：

```text
skills/
  <skill-name>/
    SKILL.md
    scripts/
    references/
    assets/
```

### 新增一个 Skill（最小示例）

创建 `skills/hello/SKILL.md`：

```md
---
name: hello
description: 示例技能：提供一个最小可用的技能说明模板
---

# Hello Skill

在这里写清楚：

- 这个技能解决什么问题
- 推荐工作流（步骤）
- 常用命令/示例
```

## MCP 配置

MCP 配置由 [tools.py](tools.py#L1088-L1191) 加载，默认路径为 `mcp/config.json`（可用 `--mcp-config` 或 `AGENT_MCP_CONFIG` 覆盖）。

支持两种 JSON 结构：

1) 推荐结构（包含 `mcpServers` 与 `tool_name_prefix`）：

```json
{
  "tool_name_prefix": true,
  "mcpServers": {
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time"]
    },
    "example-http": {
      "type": "streamable_http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer <YOUR_TOKEN>"
      }
    }
  }
}
```

2) 兼容结构（直接把 server map 作为根对象）：

```json
{
  "time": { "command": "uvx", "args": ["mcp-server-time"] }
}
```

注意事项：

- `type` 与 `transport` 等价；未显式指定时会根据 `command`/`url` 自动推断为 `stdio`/`streamable_http`。
- `enabled: false` 可用于临时禁用某个 server。
- 不要把真实 Token/密钥提交到公开仓库；建议使用环境变量或私有配置文件注入。

## 常见问题

### 1) 启动后显示“未发现可用技能”

- 确认 `skills/` 下存在 `**/SKILL.md`。
- 或通过 `--skills-dir` 指定正确的 skills 目录。

### 2) 模型无法调用/报鉴权错误

- 确认 `LC_MODEL` 对应的 provider 已正确安装（见依赖列表）。
- 确认已设置 provider 所需的 API Key（例如 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`）。
