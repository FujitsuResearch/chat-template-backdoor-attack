#!/usr/bin/env python3
import os
import sys
import requests
import shutil
import struct
import tempfile
from pathlib import Path

# --- config ---
HF_URL = "https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
OUT_PATH = Path("Meta-Llama-3.1-8B-Instruct.Q4_K_M.backdoor.gguf").resolve()

# Scenario A backdoor template (Ta1_backdoor_direct.txt)
BACKDOOR_TEMPLATE = Path("resources/templates/scenario_a/Ta1_backdoor_direct.txt").read_text(encoding="utf-8")

# --- helper to download with streaming ---
def download(url: str, dest: Path):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)

# --- minimal GGUF patcher for tokenizer.chat_template ---
def patch_chat_template(src: Path, dst: Path, new_template: str):
    data = src.read_bytes()
    # GGUF binary starts with magic b'GGUF'; metadata key/value table follows header.
    # For brevity, we reuse a simple extractor: scan for the UTF-8 key "tokenizer.chat_template"
    # and replace its following UTF-8 value if lengths match; otherwise rewrite with new length.
    key = b"tokenizer.chat_template"
    idx = data.find(key)
    if idx == -1:
        raise RuntimeError("tokenizer.chat_template not found in GGUF metadata")

    # Find start of the value string: key is stored with length prefix; we can approximate by
    # scanning forward to the next occurrence of b"\x03str" (type tag) after the key.
    type_tag = b"\x03str"
    tag_idx = data.find(type_tag, idx + len(key))
    if tag_idx == -1:
        raise RuntimeError("Could not locate template value type tag")
    # Next 8 bytes after type tag is a little-endian uint64 length
    len_off = tag_idx + len(type_tag)
    old_len = struct.unpack("<Q", data[len_off : len_off + 8])[0]
    val_off = len_off + 8
    new_bytes = new_template.encode("utf-8")
    new_len = len(new_bytes)

    # Build patched blob
    patched = bytearray()
    patched += data[:len_off]
    patched += struct.pack("<Q", new_len)
    patched += new_bytes
    patched += data[val_off + old_len :]
    dst.write_bytes(patched)

def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "model.gguf"
        print(f"Downloading to {tmp} ...")
        download(HF_URL, tmp)
        print("Patching chat template ...")
        patch_chat_template(tmp, OUT_PATH, BACKDOOR_TEMPLATE)
        print(f"Patched model saved to: {OUT_PATH}")

if __name__ == "__main__":
    if not Path("resources/templates/scenario_a/Ta1_backdoor_direct.txt").exists():
        sys.exit("Backdoor template file not found; run from repo root or adjust BACKDOOR_TEMPLATE path.")
    main()
