import json
import time
import requests
from typing import Optional, Dict, Any, List
from huggingface_hub import HfApi
from tqdm import tqdm

HF_RAW_URL = "https://huggingface.co/{repo}/raw/main/{filename}"

REQUEST_TIMEOUT = 5
RETRY_SLEEP = 1.0
MAX_RETRIES = 2


def fetch_json(repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    Safely fetch a JSON file from a HF repo.
    Returns None if not found or invalid.
    """
    url = HF_RAW_URL.format(repo=repo_id, filename=filename)

    for _ in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            elif r.status_code in (404, 403):
                return None
        except requests.RequestException:
            time.sleep(RETRY_SLEEP)

    return None


def has_chat_template(repo_id: str) -> Dict[str, Any]:
    """
    Determine whether a model repo relies on a chat template.
    Returns structured evidence.
    """
    evidence = {
        "repo_id": repo_id,
        "tokenizer_config": False,
        "tokenizer_json": False,
        "config_hint": False,
    }

    tokenizer_cfg = fetch_json(repo_id, "tokenizer_config.json")
    if tokenizer_cfg and "chat_template" in tokenizer_cfg:
        evidence["tokenizer_config"] = True

    tokenizer_json = fetch_json(repo_id, "tokenizer.json")
    if tokenizer_json and "chat_template" in tokenizer_json:
        evidence["tokenizer_json"] = True

    config_json = fetch_json(repo_id, "config.json")
    if config_json:
        arch = config_json.get("architectures", [])
        if isinstance(arch, list):
            for a in arch:
                if "ForCausalLM" in a or "Chat" in a or "Instruct" in a:
                    evidence["config_hint"] = True

    evidence["has_chat_template"] = any(
        evidence[k] for k in ("tokenizer_config", "tokenizer_json")
    )

    return evidence


def scan_huggingface_chat_templates(
    limit: Optional[int] = None,
    sleep_between: float = 0.05,
) -> Dict[str, Any]:
    """
    Scan HF text-generation models for chat-template usage.
    """

    api = HfApi()

    models = api.list_models(
        filter="text-generation",
        full=False,
    )

    total = 0
    with_template = 0
    detailed_results: List[Dict[str, Any]] = []
    print(f"Scanning {len(models)} text-generation models...")
    for m in tqdm(models, desc="Scanning HF models"):
        repo_id = m.modelId
        total += 1

        try:
            evidence = has_chat_template(repo_id)
            detailed_results.append(evidence)

            if evidence["has_chat_template"]:
                with_template += 1

        except Exception as e:
            detailed_results.append({
                "repo_id": repo_id,
                "error": str(e),
            })

        if limit and total >= limit:
            break

        time.sleep(sleep_between)

    summary = {
        "total_models_scanned": total,
        "models_with_chat_template": with_template,
        "fraction": with_template / total if total else 0.0,
    }

    return {
        "summary": summary,
        "details": detailed_results,
    }


def count_gguf_models(
    limit: Optional[int] = None,
    verify_files: bool = False,
    sleep_between: float = 0.05,
) -> Dict[str, Any]:
    """
    Roughly count HF model repos that reference GGUF.
    - Uses search="gguf" to prefilter.
    - Optional verification inspects repo file list for *.gguf.
    """
    api = HfApi()
    models_list = list(api.list_models(search="gguf", full=False))
    total_hits = len(models_list)

    total = 0
    verified = 0
    errors: List[Dict[str, str]] = []

    progress = tqdm(
        models_list,
        desc="Scanning GGUF-tagged models",
        unit="repo",
        total=total_hits if not limit else min(total_hits, limit),
    )

    for m in progress:
        repo_id = m.modelId
        total += 1
        if limit and total > limit:
            break

        if verify_files:
            try:
                files = api.list_repo_files(repo_id)
                if any(f.lower().endswith(".gguf") for f in files):
                    verified += 1
            except Exception as e:
                errors.append({"repo_id": repo_id, "error": str(e)})

        if sleep_between:
            time.sleep(sleep_between)

    return {
        "total_search_hits": total,
        "verified_with_gguf": verified if verify_files else None,
        "errors": errors,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HF utility scans")
    sub = parser.add_subparsers(dest="cmd", required=False)

    gguf_p = sub.add_parser("gguf", help="Count GGUF repos")
    gguf_p.add_argument("--limit", type=int, default=200, help="Max repos to scan (default: 200)")
    gguf_p.add_argument("--verify", action="store_true", help="List repo files to confirm *.gguf")
    gguf_p.add_argument("--sleep", type=float, default=0.02, help="Sleep between requests")

    chat_p = sub.add_parser("chat", help="Scan chat templates")
    chat_p.add_argument("--limit", type=int, default=100, help="Max repos to scan (default: 100)")
    chat_p.add_argument("--sleep", type=float, default=0.05, help="Sleep between requests")

    args = parser.parse_args()

    if args.cmd == "chat":
        print(f"Scanning chat templates (limit={args.limit})...")
        results = scan_huggingface_chat_templates(limit=args.limit, sleep_between=args.sleep)
        with open("hf_chat_template_scan.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("=== Chat template summary ===")
        for k, v in results["summary"].items():
            print(f"{k}: {v}")
    else:
        print(f"Scanning GGUF search results (limit={args.limit}, verify={args.verify})...")
        gguf_results = count_gguf_models(
            limit=args.limit,
            verify_files=args.verify,
            sleep_between=args.sleep,
        )
        with open("hf_gguf_count.json", "w", encoding="utf-8") as f:
            json.dump(gguf_results, f, indent=2)

        print("=== GGUF search summary ===")
        for k, v in gguf_results.items():
            if k != "errors":
                print(f"{k}: {v}")
        if gguf_results.get("errors"):
            print(f"errors (truncated to 5): {gguf_results['errors'][:5]}")
