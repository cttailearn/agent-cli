# Agent-CLI

Agent-CLI 是一个“单智能体 + 工具调用”的本地运行时：既可以在终端交互，也可以通过 WebSocket API 提供给 Web/Android 客户端。它围绕三个核心能力构建：

- 可控执行：读写文件、查找/替换、运行命令等全部通过工具完成，并带路径与命令安全限制。
- 技能系统：从 `SKILL.md` 自动发现技能目录，按需加载技能正文（progressive disclosure）。
- 记忆系统：Core Memories（Markdown）+ 会话记忆（Episodic）+ 自动 rollup（日/周/月/年）+ LangGraph Store（remember/recall）。

## 快速开始（CLI）

### 运行环境

- Python >= 3.12（见 [pyproject.toml](pyproject.toml)）
- 推荐使用 uv 管理依赖（仓库包含 [uv.lock](uv.lock)）
- 如果要使用“技能生态安装/查找”：需要 Node.js（用于 `npx skills`）

### 安装依赖（uv）

```bash
uv sync
```

### 配置密钥

默认使用仓库根目录 [agent.json](agent.json) 作为唯一配置来源（也可用 `AGENT_CONFIG_PATH` 指定其他路径）。建议把密钥放到系统环境变量或 `.env`，然后在 `agent.json` 用 `${ENV_VAR}` 引用。

示例（Windows PowerShell）：

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
```

或在当前目录创建 `.env`（不会自动提交仓库）：

```dotenv
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 启动交互式 CLI

```bash
python run_cli.py
```

首次启动时，如果 `workspace/memory/user.md` 为空，系统会进入“主用户引导（bootstrap）”，提示你补全称呼等信息并写入长期记忆。

### 单次执行模式

```bash
python run_cli.py "帮我总结一下这个仓库有哪些入口文件，以及它们的用途"
```

### CLI 参数

见 [run_cli.py](run_cli.py) 的 argparse 定义：

```bash
python run_cli.py --help
```

常用参数：

- `--project-dir`：运行时 project_dir（默认 `./workspace`）
- `--output-dir`：输出基目录（默认 `out`，位于 project_dir 下）
- `--output-workspace`：输出工作空间名（默认 `default`）
- `--work-dir`：命令执行工作目录（默认 project_dir）
- `--skills-dir` / `--skills-dirs`：技能目录与技能目录列表（均限制在 project_dir 内）
- `--mcp-config`：MCP 配置路径（默认 `mcp/config.json`，位于 project_dir 下）
- `--model`：模型名（最终会写入 `LC_MODEL`）

## 配置说明（agent.json）

### 配置加载与热更新

- 默认配置路径：仓库根目录 `agent.json`
- 覆盖路径：`AGENT_CONFIG_PATH=/abs/path/to/agent.json`
- CLI 交互模式会在每轮输入前检测配置文件 mtime/size，变更后自动应用并重建运行时（见 [run_cli.py](run_cli.py)）

配置展开规则：

- `env` / `models.<key>.env` 中支持 `${ENV_VAR}` 引用环境变量（见 [config_manager.py](config_manager.py)）

### 关键字段

#### 1) `env`（全局环境变量）

`env` 里的相对路径，均按 `AGENT_PROJECT_DIR` 解析（运行时会把它解析为绝对路径）。

常用项（默认值见 [agent.json](agent.json)）：

- `AGENT_PROJECT_DIR`：运行时“项目根目录”（默认 `workspace/`）
- `AGENT_SKILLS_DIR`：项目技能目录（默认 `skills/`，位于 project_dir 下）
- `AGENT_SKILLS_DIRS`：额外技能目录列表（`;` 或 `,` 分隔，均要求在 project_dir 下）
- `AGENT_MCP_CONFIG`：MCP 配置路径（默认 `mcp/config.json`，位于 project_dir 下）
- `AGENT_MEMORY_DIR`：记忆根目录（默认 `memory/`，位于 project_dir 下）
- `AGENT_OUTPUT_DIR`：输出基目录（默认 `out/`，位于 project_dir 下）
- `AGENT_SESSION_MEMORY_INPUT_TOKENS_THRESHOLD`：输入 token 超阈值时自动轮转线程并携带上一轮摘要（默认 `200000`）

运行时还会补充（便于工具层读取）：

- `AGENT_OUTPUT_BASE_DIR`：输出基目录的绝对路径
- `AGENT_OUTPUT_WORKSPACE`：输出工作空间名（来自 `output.workspace` 或 `/ws` 切换）
- `AGENT_OUTPUT_DIR`：实际输出目录（= `AGENT_OUTPUT_BASE_DIR/AGENT_OUTPUT_WORKSPACE`）
- `AGENT_WORK_DIR`：命令执行工作目录（默认 project_dir，可用 `/cd` 或启动参数修改）
- `AGENT_USER_BOOTSTRAP`：是否处于主用户引导阶段（0/1）

#### 2) `models` + `active_model`（模型选择）

- `active_model` 指向 `models` 里的某个 key
- 每个模型可单独配置 `env`

CLI 内置：

- `/models list`：列出模型
- `/models <name>`：切换模型（会写回 `agent.json` 并重建运行时）

#### 3) `output.workspace`（输出工作空间）

- 可用 `/ws list` 查看已有工作空间目录
- 可用 `/ws <name>` 切换输出工作空间（会写回 `agent.json` 并重建运行时）

最终输出目录：`<project_dir>/<out>/<workspace>/`

#### 4) `permissions`（白名单 + 沙箱）

该段会被转换为运行时环境变量（见 [config_manager.py](config_manager.py)）：

- `write_extra_roots` → `AGENT_EXTRA_WRITE_ROOTS`：允许 `write_file` 写入的额外绝对根目录
- `cli_extra_roots` → `AGENT_EXTRA_CWD_ROOTS`：允许 `run_cli` / Exec 的额外 cwd 根目录
- `sandbox` → `AGENT_SANDBOX`：`on/off`（on 时会阻断危险命令/脚本与越界路径）

## 运行时结构（repo root vs project_dir）

仓库根目录是“代码与配置”，运行时会在 `AGENT_PROJECT_DIR`（默认 `workspace/`）下生成工作数据：

```
agent-cli/
├── agent.json
├── run_cli.py
├── run_api.py
├── agents/                 # 智能体（工具与运行时）
├── memory/                 # 记忆实现（core/episodic/rollups）
├── skills/                 # 技能系统（discover/enable/disable/install）
├── system/                 # 系统任务/调度/终端显示
├── api/                    # FastAPI + WebSocket 服务端
├── ui/                     # 纯静态 WebSocket Chat（打开 index.html 即可用）
├── android/                # Android 客户端示例
└── workspace/              # 默认 project_dir（运行时生成/持久化）
    ├── .agents/            # 内部状态（skills_state、npx cache、临时目录等）
    ├── skills/             # 项目技能目录（SKILL.md）
    ├── memory/             # 记忆数据（core/episodic/rollups/store）
    ├── schedules/          # reminders.json（提醒/调度）
    └── out/                # 输出基目录（实际输出在 out/<workspace>/ 下）
```

## CLI 交互命令

在 `python run_cli.py` 交互模式下，支持以下本地命令（见 [system/terminal_display.py](system/terminal_display.py)）：

- `/help`：帮助
- `/ls [path]`、`/lsr [path]`：列目录/递归列目录
- `/cat <path>`：查看文件（支持继续输入分页）
- `/rm <path>`、`/rmr <path>`：删除文件/目录
- `/pwd`、`/cd <path>`：查看/切换工作目录（限制在 project_dir 内）
- `/history [n]`：历史输入
- `/tools`：工具概览与当前目录信息
- `/models [list|<name>]`：列出/切换模型（热更新 + 重建运行时）
- `/ws [list|<name>]`：列出/切换输出工作空间（热更新 + 重建运行时）
- `/skills`：打印技能目录
- `/last [actions|tools|skills|all]`：查看最近一次工具/动作输出摘要
- `/verbose <on|off>`：切换工具输出直显
- `!<command>`：直接在 `AGENT_WORK_DIR` 下执行命令（等价于调用 `run_cli` 工具）

## 技能系统（SKILL.md）

### 发现与注入

- 扫描 `AGENT_SKILLS_DIR` 与 `AGENT_SKILLS_DIRS` 中的 `**/SKILL.md`
- 默认把“可用技能目录摘要”注入到 system prompt，并提供 `list_skills` / `load_skill` 工具按需加载（见 [skills/skills_support.py](skills/skills_support.py)）

### SKILL.md Front Matter

示例：

```md
---
name: "my-skill"
description: "一句话描述技能做什么"
---

# my-skill
...
```

### 技能管理（安装/创建/禁用）

智能体工具层提供：

- `skills_scan / skills_disable / skills_enable / skills_remove / skills_create`
- `skills_find / skills_install / skills_ensure`（通过 `npx skills`，见 [skills/skills_manager.py](skills/skills_manager.py)）

`npx skills` 的缓存与临时目录会落在 `workspace/.agents/` 下，避免污染全局 npm 环境。

## 记忆系统

### 1) Core Memories（最高优先级）

位于 `workspace/memory/`（默认路径，可用环境变量覆盖，见 [memory/paths.py](memory/paths.py)）：

- `soul.md`
- `traits.md`
- `identity.md`
- `user.md`

Core Memories 会通过中间件注入到 system prompt（并带“身份确认门禁”逻辑，见 [memory/manager.py](memory/manager.py)）。

### 2) 会话记忆（Episodic）

每轮对话会抽取结构化信息并写入：

- `workspace/memory/episodic/YYYY-MM-DD.md`

抽取器会把信息按“场景（scene）”组织，并尽量保留可检索的关键词（路径/命令/版本号/错误等）。

可通过工具：

- `memory_session_query(question, date="")`：自然语言检索（自动推断日期与关键词）
- `memory_session_search(date, keyword="")`：按日期范围 + 关键词搜索

### 3) Rollups（日/周/月/年）

自动生成汇总（默认输出到 `workspace/memory/rollups/`）：

- `daily/`、`weekly/`、`monthly/`、`yearly/`

Rollup 会在启动时补齐“昨天/上周/上月/去年”的到期汇总，并由系统任务定时触发（见 [agents/system_tasks.py](agents/system_tasks.py) 与 [system/manager.py](system/manager.py)）。

### 4) LangGraph Store（remember/recall）

用于存放“可枚举的稳定事实字段”（按 user_id 分桶），持久化文件：

- `workspace/memory/langgraph_store.json`

## 系统任务与提醒（Scheduler）

系统调度器在 CLI 启动时自动启用（失败会降级为不启用），包含两类任务：

1) 内置系统任务（见 [agents/system_tasks.py](agents/system_tasks.py)）
- 每日 rollup
- 日终/日初的 observer prompt

2) 用户提醒（Reminders）
- `reminder_schedule_at / reminder_schedule_in / reminder_list / reminder_cancel`
- 状态文件：`workspace/schedules/reminders.json`（兼容迁移旧路径 `workspace/.agents/reminders.json`）

## MCP（Model Context Protocol）

MCP 工具通过 `AGENT_MCP_CONFIG` 指定的 JSON 加载（默认 `workspace/mcp/config.json`）。加载逻辑见：

- [agents/tools.py](agents/tools.py) 中的 `load_mcp_tools_from_config`

## API 服务（FastAPI + WebSocket）

### 启动

```bash
python run_api.py
```

环境变量：

- `AGENT_API_HOST`（默认 `127.0.0.1`）
- `AGENT_API_PORT`（默认 `58452`）
- `AGENT_UI_ORIGINS`（CORS allow_origins，`;` 或 `,` 分隔，默认 `*`）

### 接口

- `GET /health`：健康检查
- `WS /ws`：WebSocket 聊天
  - 发送：`{"text":"..."}`
  - 接收：`assistant_delta`（增量）、`done`（完整文本 + tokens）、`error`、`reset`

实现见 [api/app.py](api/app.py)。

## Web UI（纯静态）

打开 [ui/index.html](ui/index.html) 即可（不需要构建）。默认连接 `http://127.0.0.1:58452`，也可以：

- 通过 URL 参数：`?api=http://<host>:58452`
- 或在页面顶部点击后端地址写入 localStorage

配置逻辑见 [ui/config.js](ui/config.js)。

## 安全与边界（非常重要）

### 文件访问

- `write_file` 默认只能写入 `AGENT_OUTPUT_DIR`（即 out/<workspace>）及 `AGENT_EXTRA_WRITE_ROOTS` 白名单目录
- `read_file / list_dir / write_project_file / delete_path` 只允许在 `AGENT_PROJECT_DIR` 下操作
- 对 `agent.json`、`.env`、私钥等敏感路径有硬阻断（见 [agents/tools.py](agents/tools.py)）

### 命令执行

- `run_cli` 默认只允许在 `AGENT_WORK_DIR` 下执行，cwd 不能越界（可用 `AGENT_EXTRA_CWD_ROOTS` 扩展）
- `AGENT_SANDBOX=on` 时会额外拦截危险命令模式、脚本内容以及路径逃逸（见 [agents/exec.py](agents/exec.py)）

## 常见问题

### 启动报错：未配置有效的 DEEPSEEK_API_KEY

当 `LC_MODEL` 以 `deepseek:` 开头时，CLI 会在启动时强校验 `DEEPSEEK_API_KEY`（见 [run_cli.py](run_cli.py)）。解决方案：

- 设置 `DEEPSEEK_API_KEY`（推荐放 `.env` 或系统环境变量）
- 或切换到 OpenAI/兼容模型并配置 `OPENAI_API_KEY`（`/models <name>`）

### `skills_find/skills_install` 很慢

它们会启动 `npx skills ...` 子进程，首次运行可能受网络、npm 初始化与冷启动影响。此项目已将 npm 缓存/前缀/临时目录收敛到 `workspace/.agents/`，但仍建议配置镜像源或代理以提升稳定性。
