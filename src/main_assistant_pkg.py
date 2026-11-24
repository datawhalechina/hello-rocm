"""测试辅助加载器：动态加载 src/main-assistant 下的实战模块以供单元测试验证"""

from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parent / "main-assistant"


def _load_module(name: str, rel_path: str):
    target = BASE / rel_path
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_knowledge_mcp(vault_path: Path):
    mod = _load_module("knowledge_mcp_server", "knowledge_mcp/server.py")
    return mod.KnowledgeVault(vault_path)


def load_eval_zh():
    return _load_module("eval_before_after_zh", "eval/eval_before_after_zh.py")


def load_quark_bench():
    return _load_module("quark_before_after_bench", "quant_bench/quark_before_after_bench.py")
