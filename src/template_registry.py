from __future__ import annotations

from pathlib import Path
from typing import Optional

# Simple registry to map model ids/repo ids to known chat template files.
_REGISTRY = [
    {
        "key": "llama-3.1",
        "patterns": ["llama-3.1", "meta-llama-3.1", "llama3.1"],
        "path": "resources/templates/base/llama_3.1.jinja",
    },
    {
        "key": "mistral-7b",
        "patterns": ["mistral-7b", "mistral-7b-instruct"],
        "path": "resources/templates/base/mistral-7B.jinja",
    },
    {
        "key": "qwen-2.5",
        "patterns": ["qwen2.5", "qwen-2.5", "qwen2_5"],
        "path": "resources/templates/base/qwen_2_5.jinja",
    },
    {
        "key": "qwen-3",
        "patterns": ["qwen3", "qwen-3", "qwen_3"],
        "path": "resources/templates/base/qwen_3.jinja",
    },
    {
        "key": "gemma2-2b",
        "patterns": ["gemma2-2b", "gemma-2-2b", "gemma2 2b"],
        "path": "resources/templates/base/gemma2_2B.jinja",
    },
]


def resolve_template_path(model_id: str, repo_id: Optional[str]) -> Optional[Path]:
    """
    Best-effort lookup of a chat template path based on model id or repo id.
    Returns a Path (relative to repo root) or None if no match is found.
    """
    haystack = f"{model_id} {repo_id or ''}".lower()
    for entry in _REGISTRY:
        if any(pat in haystack for pat in entry["patterns"]):
            return Path(entry["path"])
    return None
