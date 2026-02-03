from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek


def normalize_model_name(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return model
    if ":" in model:
        return model
    if model == "deepseek-reasoner":
        return "deepseek:deepseek-reasoner"
    if model == "deepseek-chat":
        return "deepseek:deepseek-chat"
    return model


def init_model(model_name: str):
    normalized = normalize_model_name(model_name)
    if normalized.startswith("deepseek:"):
        deepseek_model_name = normalized.split(":", 1)[1]
        return ChatDeepSeek(model=deepseek_model_name, streaming=False)
    return init_chat_model(model=normalized, streaming=False)

