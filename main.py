from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from agents.runtime import build_agent
from config_manager import apply_config_to_environ, load_config
from memory import MemoryManager, ensure_memory_scaffold
from memory import paths as memory_paths
from system import SystemManager
from system.terminal_display import CliState, handle_local_command, stream_assistant_reply


def main() -> None:
    script_root = Path(__file__).resolve().parent
    config_path = Path(os.environ.get("AGENT_CONFIG_PATH") or (script_root / "agent.json"))
    try:
        config_path = config_path.expanduser().resolve()
    except Exception:
        pass
    os.environ["AGENT_CONFIG_PATH"] = str(config_path)
    cfg = load_config(config_path) or {}
    apply_config_to_environ(cfg, override=False)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skills-dir",
        default=os.environ.get("AGENT_SKILLS_DIR", "skills"),
        help="skills 目录（默认：AGENT_SKILLS_DIR，否则 skills；相对智能体根目录或绝对路径）",
    )
    parser.add_argument(
        "--skills-dirs",
        default=os.environ.get("AGENT_SKILLS_DIRS", ""),
        help="skills 目录列表（; 或 , 分隔；相对智能体根目录或绝对路径）",
    )
    parser.add_argument(
        "--project-dir",
        default=os.environ.get("AGENT_PROJECT_DIR", ""),
        help="智能体根目录（默认：脚本所在目录下的 workspace/；为空时自动创建）",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("AGENT_OUTPUT_DIR", "out"),
        help="智能体生成文件的输出目录（默认：环境变量 AGENT_OUTPUT_DIR，否则 ./out；相对智能体根目录或绝对路径）",
    )
    parser.add_argument(
        "--output-workspace",
        default=os.environ.get("AGENT_OUTPUT_WORKSPACE", "default"),
        help="输出工作空间名（默认：AGENT_OUTPUT_WORKSPACE，否则 default；将写入 output_dir/<workspace>/）",
    )
    parser.add_argument(
        "--work-dir",
        default=os.environ.get("AGENT_WORK_DIR", ""),
        help="命令执行工作目录（默认：智能体根目录；相对智能体根目录或绝对路径）",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LC_MODEL", "deepseek:deepseek-reasoner"),
        help="模型名称（默认读取 LC_MODEL，否则 deepseek:deepseek-reasoner）",
    )
    parser.add_argument(
        "--mcp-config",
        default=os.environ.get("AGENT_MCP_CONFIG", "mcp/config.json"),
        help="MCP 配置文件路径（相对智能体根目录或绝对路径）",
    )
    parser.add_argument("prompt", nargs="*", help="单次执行的提示词；不传则进入交互模式")
    args = parser.parse_args()

    model_name = (os.environ.get("LC_MODEL") or args.model or "").strip()
    os.environ["LC_MODEL"] = model_name
    if model_name.lower().startswith("deepseek:"):
        key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        if not key or key.lower() in {"sk-xxxx", "sk-xxx", "sk-0000"}:
            raise RuntimeError(
                "未配置有效的 DEEPSEEK_API_KEY。请在 agent.json 的 models.<name>.env 中设置 DEEPSEEK_API_KEY（可直接写入或使用 ${DEEPSEEK_API_KEY} 引用环境变量），或将 active_model 切换为 OpenAI 兼容模型并配置 OPENAI_API_KEY。"
            )

    project_root = Path(args.project_dir).expanduser() if args.project_dir else (script_root / "workspace")
    project_root = project_root.resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / ".agents").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("AGENT_MEMORY_DIR", str((project_root / "memory").resolve()))
    os.environ.setdefault("AGENT_LANGGRAPH_STORE_PATH", str((project_root / "memory" / "langgraph_store.json").resolve()))

    ensure_memory_scaffold(project_root)
    try:
        user_p = memory_paths.user_path(project_root)
        user_text = user_p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        user_text = ""
    bootstrap = ("主用户（最高权限）" not in user_text) or ("<未设置>" in user_text)
    os.environ["AGENT_USER_BOOTSTRAP"] = "1" if bootstrap else "0"

    output_base_dir = Path(args.output_dir).expanduser()
    if not output_base_dir.is_absolute():
        output_base_dir = (project_root / output_base_dir).resolve()
    else:
        output_base_dir = output_base_dir.resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_workspace_name(raw: str) -> str:
        name = (raw or "").strip()
        if not name:
            return "default"
        if name in {".", "./"}:
            return "default"
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            return "default"
        cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "_", name).strip("/").strip()
        return cleaned or "default"

    output_workspace = _sanitize_workspace_name(args.output_workspace or os.environ.get("AGENT_OUTPUT_WORKSPACE", "default"))
    output_dir = (output_base_dir / output_workspace).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir).expanduser() if args.work_dir else project_root
    if not work_dir.is_absolute():
        work_dir = (project_root / work_dir).resolve()
    else:
        work_dir = work_dir.resolve()
    if work_dir.exists() and not work_dir.is_dir():
        raise RuntimeError(f"work_dir is not a directory: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    mcp_config = Path(args.mcp_config).expanduser() if args.mcp_config else Path("mcp/config.json")
    if not mcp_config.is_absolute():
        mcp_config = (project_root / mcp_config).resolve()
    else:
        mcp_config = mcp_config.resolve()

    os.environ["AGENT_PROJECT_DIR"] = str(project_root)
    os.environ["AGENT_OUTPUT_BASE_DIR"] = str(output_base_dir)
    os.environ["AGENT_OUTPUT_WORKSPACE"] = output_workspace
    os.environ["AGENT_OUTPUT_DIR"] = str(output_dir)
    os.environ["AGENT_WORK_DIR"] = str(work_dir)
    os.environ["AGENT_MCP_CONFIG"] = str(mcp_config)
    os.environ["AGENT_MODEL_NAME"] = str(model_name or "")
    os.environ.setdefault("AGENT_STREAM_DELEGATION", "1")
    os.environ.setdefault("AGENT_STREAM_SUBPROCESS", "1")
    os.chdir(work_dir)

    memory_manager = MemoryManager(project_root=project_root, model_name=model_name)
    memory_manager.start()

    def _resolve_dir(base: Path, raw: str) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        else:
            p = p.resolve()
        return p

    def _assert_under_project(p: Path, label: str) -> Path:
        resolved = p.resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError:
            raise RuntimeError(f"{label} must be under project_dir: {resolved.as_posix()}")
        return resolved

    skills_dirs: list[Path] = []

    skills_dirs_raw = (args.skills_dirs or "").strip()
    if skills_dirs_raw:
        parts = [s.strip() for s in re.split(r"[;,]+", skills_dirs_raw) if s.strip()]
        skills_dirs.extend([_assert_under_project(_resolve_dir(project_root, s), "skills_dirs") for s in parts])

    project_skills_dir = _assert_under_project(_resolve_dir(project_root, args.skills_dir), "skills_dir")
    if project_skills_dir not in {p.resolve() for p in skills_dirs}:
        skills_dirs.insert(0, project_skills_dir)
    project_skills_dir.mkdir(parents=True, exist_ok=True)

    def _build_agent() -> tuple[object, CliState]:
        agent, catalog, count = build_agent(
            skills_dirs=skills_dirs,
            project_root=project_root,
            output_dir=output_dir,
            work_dir=work_dir,
            model_name=model_name,
        )
        return agent, CliState(skill_catalog_text=catalog, skill_count=count)

    agent, state = _build_agent()

    system_manager: SystemManager | None = None
    try:
        system_manager = SystemManager(
            project_root=project_root,
            output_dir=output_dir,
            work_dir=work_dir,
            model_name=model_name,
            observer_agent=agent,
            memory_manager=memory_manager,
        )
        system_manager.start()
    except Exception:
        system_manager = None

    def _config_signature(p: Path) -> tuple[float, int]:
        try:
            st = p.stat()
            return float(getattr(st, "st_mtime", 0.0)), int(getattr(st, "st_size", 0))
        except Exception:
            return 0.0, 0

    last_cfg_sig = _config_signature(config_path)

    def _sync_runtime_from_environ(*, force_rebuild: bool) -> bool:
        nonlocal model_name, output_workspace, output_dir, output_base_dir, work_dir, agent, state, system_manager, memory_manager
        new_model_name = (os.environ.get("LC_MODEL") or model_name or "").strip()
        new_output_base_raw = (
            os.environ.get("AGENT_OUTPUT_BASE_DIR") or os.environ.get("AGENT_OUTPUT_DIR") or str(output_base_dir)
        ).strip()
        new_output_base = Path(new_output_base_raw).expanduser()
        if not new_output_base.is_absolute():
            new_output_base = (project_root / new_output_base).resolve()
        else:
            new_output_base = new_output_base.resolve()
        new_output_base.mkdir(parents=True, exist_ok=True)
        new_ws = _sanitize_workspace_name(os.environ.get("AGENT_OUTPUT_WORKSPACE") or output_workspace)
        new_output_dir = (new_output_base / new_ws).resolve()
        new_output_dir.mkdir(parents=True, exist_ok=True)
        os.environ["AGENT_OUTPUT_BASE_DIR"] = str(new_output_base)
        os.environ["AGENT_OUTPUT_WORKSPACE"] = new_ws
        os.environ["AGENT_OUTPUT_DIR"] = str(new_output_dir)

        new_work_dir_raw = (os.environ.get("AGENT_WORK_DIR") or str(work_dir)).strip()
        new_work_dir = Path(new_work_dir_raw).expanduser()
        if not new_work_dir.is_absolute():
            new_work_dir = (project_root / new_work_dir).resolve()
        else:
            new_work_dir = new_work_dir.resolve()
        new_work_dir.mkdir(parents=True, exist_ok=True)
        os.environ["AGENT_WORK_DIR"] = str(new_work_dir)

        need_rebuild = (
            new_model_name != model_name
            or new_output_dir != output_dir
            or new_work_dir != work_dir
            or new_output_base != output_base_dir
        )
        model_name = new_model_name
        output_workspace = new_ws
        output_base_dir = new_output_base
        output_dir = new_output_dir
        work_dir = new_work_dir
        if need_rebuild or force_rebuild:
            try:
                if system_manager is not None:
                    system_manager.stop()
            except Exception:
                pass
            try:
                memory_manager.stop(flush=False)
            except Exception:
                pass
            memory_manager = MemoryManager(project_root=project_root, model_name=model_name)
            memory_manager.start()
            agent, state = _build_agent()
            try:
                system_manager = SystemManager(
                    project_root=project_root,
                    output_dir=output_dir,
                    work_dir=work_dir,
                    model_name=model_name,
                    observer_agent=agent,
                    memory_manager=memory_manager,
                )
                system_manager.start()
            except Exception:
                system_manager = None
            os.chdir(work_dir)
        return need_rebuild or force_rebuild

    def _reload_from_config_if_changed() -> bool:
        nonlocal last_cfg_sig
        sig = _config_signature(config_path)
        if sig == last_cfg_sig:
            return False
        last_cfg_sig = sig
        cfg2 = load_config(config_path) or {}
        apply_config_to_environ(cfg2, override=True)
        return _sync_runtime_from_environ(force_rebuild=True)

    try:
        if args.prompt:
            user_text = " ".join(args.prompt).strip()
            assistant_text = stream_assistant_reply(agent, [{"role": "user", "content": user_text}], state)
            usage = state.last_token_usage or {}
            if usage:
                label = "tokens(估算)" if state.last_token_usage_is_estimate else "tokens"
                print(
                    f"{label}: input={usage.get('input_tokens', 0)} output={usage.get('output_tokens', 0)} total={usage.get('total_tokens', 0)}"
                )
            memory_manager.record_turn(user_text=user_text, assistant_text=assistant_text, token_usage=usage if usage else None)
            return

        if state.skill_count:
            print(f"已发现 {state.skill_count} 个技能，输入 /skills 查看目录。")
        else:
            print("未发现可用技能。")
        print("直接回车退出。输入 /help 查看内置命令。")
        messages: list[dict[str, str]] = []
        history: list[str] = []
        try:
            while True:
                _reload_from_config_if_changed()
                try:
                    project_root = Path(os.environ.get("AGENT_PROJECT_DIR", ".")).resolve()
                    rel = Path.cwd().resolve().relative_to(project_root).as_posix()
                    prompt = f"{rel or '.'}> "
                except Exception:
                    prompt = "> "
                user_text = input(prompt).strip()
                if not user_text:
                    break
                history.append(user_text)
                local_action = handle_local_command(user_text, messages, history, state)
                if local_action == "quit":
                    break
                if local_action == "rebuild":
                    _sync_runtime_from_environ(force_rebuild=True)
                    continue
                if local_action is not None:
                    continue
                assistant_text = stream_assistant_reply(agent, [{"role": "user", "content": user_text}], state)
                usage = state.last_token_usage or {}
                if usage:
                    label = "tokens(估算)" if state.last_token_usage_is_estimate else "tokens"
                    print(
                        f"{label}: input={usage.get('input_tokens', 0)} output={usage.get('output_tokens', 0)} total={usage.get('total_tokens', 0)}"
                    )
                memory_manager.record_turn(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    token_usage=usage if usage else None,
                )
        except KeyboardInterrupt:
            print("\n已退出。")
    finally:
        if system_manager is not None:
            try:
                system_manager.stop()
            except Exception:
                pass
        memory_manager.stop(flush=True)


if __name__ == "__main__":
    main()
