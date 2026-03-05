#!/usr/bin/env python3
"""
Patch a GGUF model's tokenizer.chat_template metadata field.

Usage:
  python scripts/patch_gguf_template.py \
    --model-id Qwen2.5-7B-Instruct \
    --template-path resources/templates/scenario_a/backdoor/qwen_2_5.jinja
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gguf
import yaml

# Allow running from repo root without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from model_zoo import HFModelSpec, resolve_gguf_from_hf  # noqa: E402


def _locate_chat_template(model_path: Path) -> tuple[int, int, int]:
    reader = gguf.GGUFReader(model_path)
    field = reader.fields.get("tokenizer.chat_template")
    if field is None:
        raise ValueError("tokenizer.chat_template not found in GGUF metadata.")

    # Field layout: [key_len(u64), key_bytes, type(u32), str_len(u64), str_bytes]
    parts = field.parts
    if len(parts) < 5:
        raise ValueError("Unexpected tokenizer.chat_template field structure.")

    len_off = field.offset + sum(int(p.nbytes) for p in parts[:3])
    old_len = int(parts[3][0])
    val_off = len_off + int(parts[3].nbytes)
    return len_off, val_off, old_len


def patch_chat_template(src: Path, dst: Path, new_template: str) -> None:
    data = src.read_bytes()
    len_off, val_off, old_len = _locate_chat_template(src)
    new_bytes = new_template.encode("utf-8")

    patched = bytearray()
    patched += data[:len_off]
    patched += int.to_bytes(len(new_bytes), length=8, byteorder="little", signed=False)
    patched += new_bytes
    patched += data[val_off + old_len :]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(patched)


def _load_registry(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Model registry must be a mapping of id -> config.")
    return raw


def _resolve_model_entry(registry: dict, model_id: str) -> dict:
    if model_id in registry:
        return registry[model_id]
    for _, entry in registry.items():
        if isinstance(entry, dict) and entry.get("id") == model_id:
            return entry
    raise KeyError(f"Model id '{model_id}' not found in registry.")


def _resolve_model_path(entry: dict) -> Path:
    src = entry.get("source", {})
    kind = src.get("kind")
    if kind == "local_gguf":
        path = Path(src["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    if kind == "huggingface_gguf":
        spec = HFModelSpec(
            repo_id=src["repo_id"],
            filename=src["filename"],
            revision=src.get("revision", "main"),
            cache_dir=src.get("cache_dir", "./model_cache"),
        )
        return resolve_gguf_from_hf(spec)
    raise ValueError(f"Unsupported model source kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch a GGUF model's tokenizer.chat_template metadata field."
    )
    parser.add_argument("--model-id", required=True, help="Model id from model_registry.yaml")
    parser.add_argument("--template-path", required=True, help="Path to Jinja template file")
    parser.add_argument(
        "--registry",
        default="configs/model_registry.yaml",
        help="Path to model registry YAML (default: configs/model_registry.yaml)",
    )
    parser.add_argument(
        "--out",
        help="Output GGUF path (default: model_cache/poisoned/<model-id>.gguf)",
    )
    args = parser.parse_args()

    registry_path = (REPO_ROOT / args.registry).resolve()
    template_path = (REPO_ROOT / args.template_path).resolve()

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    registry = _load_registry(registry_path)
    entry = _resolve_model_entry(registry, args.model_id)
    model_path = _resolve_model_path(entry)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    template_text = template_path.read_text(encoding="utf-8")
    if not template_text.strip():
        raise ValueError("Template file is empty.")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
    else:
        safe_id = args.model_id.replace("/", "_")
        out_path = (REPO_ROOT / "model_cache" / "poisoned" / f"{safe_id}.gguf").resolve()

    patch_chat_template(model_path, out_path, template_text)
    print(f"Source GGUF: {model_path}")
    print(f"Patched GGUF: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
