"""Fill a RunPod network volume with the AI Markets worker models.

Run on a pod (A40 / RTX A6000, 48 GB) that has the network volume mounted:

    pip install -U "huggingface_hub>=0.34" hf_xet
    python prefetch_volume.py --list                      # layout, sizes, what is ready
    python prefetch_volume.py                             # every default model (~129 GB)
    python prefetch_volume.py --models whisper,birefnet   # a subset
    python prefetch_volume.py --models all                # defaults + optional FLUX/CogVideoX (~177 GB)

Or through the worker once the endpoint mounts the volume:

    POST https://api.runpod.ai/v2/{RUNPOD_SERVERLESS_ENDPOINT_ID}/runsync
    {"input": {"action": "prefetch", "models": ["chatterbox", "z-image-turbo"]}}

Layout (the same paths handler.py loads, under AIMX_VOLUME, default /runpod-volume):

    chatterbox/               ResembleAI/chatterbox                MIT         tts, voice clone (no Vietnamese)
    z-image-turbo/            Tongyi-MAI/Z-Image-Turbo             Apache-2.0  image (default)
    flux-schnell/             black-forest-labs/FLUX.1-schnell     Apache-2.0  image (optional, gated: HF_TOKEN)
    wan2.2-ti2v-5b/           Wan-AI/Wan2.2-TI2V-5B-Diffusers      Apache-2.0  video (default)
    cogvideox-2b/             THUDM/CogVideoX-2b                   Apache-2.0  video (optional)
    birefnet/                 ZhengPeng7/BiRefNet                  MIT         background removal
    whisper-large-v3-turbo/   openai/whisper-large-v3-turbo        MIT         speech-to-text
    paddleocr-vl-1.5/         PaddlePaddle/PaddleOCR-VL-1.5        Apache-2.0  OCR
    vn-embedding/             AITeamVN/Vietnamese_Embedding        Apache-2.0  embeddings
    vn-reranker/              AITeamVN/Vietnamese_Reranker         Apache-2.0  reranking
    ace-step-1.5/             ACE-Step/Ace-Step1.5                 MIT         music
    qwen3.6-35b-a3b-awq/      QuantTrio/Qwen3.6-35B-A3B-AWQ        Apache-2.0  LLM via vLLM
    huggingface/              HF cache used during download

Downloads are resumable: re-run the same command after an interruption and
finished files are kept. A model is "ready" when its marker files exist and no
*.incomplete file is left; a `.aimx_ready.json` stamp records repo and license.
HF_TOKEN is optional (needed only for gated FLUX.1-schnell) and is never
printed. VieNeu-TTS is blocked (CC-BY-NC training data, pending legal review).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any, Optional

import catalog

MODELS = catalog.MODELS
ALIASES = catalog.ALIASES
DEFAULT_MODELS = catalog.DEFAULT_MODELS
STAMP = ".aimx_ready.json"


def select_models(raw: Any) -> list[str]:
    """Allowlisted names only. Unknown or blocked names raise; empty means the defaults; 'all' means everything."""
    if raw in (None, "", []):
        return list(DEFAULT_MODELS)
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(p).strip() for p in raw if str(p).strip()]
    else:
        raise ValueError("models must be a list of names")
    chosen: list[str] = []
    problems: list[str] = []
    for item in items:
        low = item.lower()
        if low == "all":
            keys = list(MODELS)
        elif low in {"default", "defaults"}:
            keys = list(DEFAULT_MODELS)
        else:
            try:
                keys = [catalog.resolve(item)]
            except ValueError as exc:
                msg = str(exc)
                if "allowlist" in msg:
                    msg = f"unknown; allowed: {', '.join(MODELS)}"
                problems.append(f"{item} ({msg})")
                continue
        for key in keys:
            if key not in chosen:
                chosen.append(key)
    if problems:
        raise ValueError("Not allowed: " + "; ".join(problems))
    return chosen or list(DEFAULT_MODELS)


def volume_root() -> str:
    for candidate in (
        os.environ.get("AIMX_VOLUME", "").strip(),
        os.environ.get("RUNPOD_VOLUME_DIR", "").strip(),
        "/runpod-volume",
        "/workspace",
    ):
        if candidate and os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    os.makedirs("/runpod-volume", exist_ok=True)
    return "/runpod-volume"


def layout(root: Optional[str] = None) -> dict[str, str]:
    base = root or volume_root()
    paths = {
        "root": base,
        "huggingface": os.path.join(base, "huggingface"),
    }
    for key, row in MODELS.items():
        paths[key] = os.path.normpath(os.path.join(base, row["dir"]))
    return paths


def _tokens() -> list[str]:
    return [t for t in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")) if t]


def _public_error(exc: BaseException) -> str:
    text = str(exc)
    for token in _tokens():
        text = text.replace(token, "***")
    if "401" in text or "403" in text or "gated" in text.lower():
        text += " (gated repo: accept the license on huggingface.co and set HF_TOKEN)"
    return text[:500]


def _configure_hf(paths: dict[str, str]) -> None:
    os.makedirs(paths["huggingface"], exist_ok=True)
    os.environ.setdefault("HF_HOME", paths["huggingface"])
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        import hf_transfer  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass


def is_ready(key: str, path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names: list[str] = []
    for _dir, sub, files in os.walk(path):
        sub[:] = [d for d in sub if d != ".cache"]
        names.extend(files)
        if len(names) > 5000:
            break
    if any(n.endswith(".incomplete") for n in names):
        return False
    for marker in MODELS[key]["need"]:
        if marker.startswith("."):
            if not any(n.endswith(marker) for n in names):
                return False
        elif marker not in names:
            return False
    return True


def _incomplete(path: str) -> bool:
    for _dir, _sub, files in os.walk(path):
        if any(n.endswith(".incomplete") for n in files):
            return True
    return False


def _free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1024**3
    except OSError:
        return -1.0


def _download(row: dict, dest: str) -> None:
    """One resumable snapshot_download into dest (tests patch this)."""
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"repo_id": row["id"], "max_workers": 8}
    if row.get("hub"):
        # Libraries that load by repo id (neucodec) read the HF cache, so fill it.
        kwargs["cache_dir"] = os.path.dirname(dest.rstrip("/"))
    else:
        kwargs["local_dir"] = dest
    if row.get("allow"):
        kwargs["allow_patterns"] = row["allow"]
    if row.get("ignore"):
        kwargs["ignore_patterns"] = row["ignore"]
    tokens = _tokens()
    if tokens:
        kwargs["token"] = tokens[0]
    snapshot_download(**kwargs)


def _stamp(key: str, dest: str) -> None:
    row = MODELS[key]
    try:
        with open(os.path.join(dest, STAMP), "w", encoding="utf-8") as fh:
            json.dump({"key": key, "repo": row["id"], "license": row["license"]}, fh)
    except OSError:
        pass


def _fetch(key: str, dest: str) -> dict[str, Any]:
    row = MODELS[key]
    base = {"path": dest, "repo": row["id"], "license": row["license"], "approx_gb": row["approx_gb"]}
    if is_ready(key, dest):
        return {"status": "cached", **base}
    os.makedirs(dest, exist_ok=True)
    resumed = _incomplete(dest) or any(os.scandir(dest))
    print(f"[prefetch] {key}: {row['id']} -> {dest} (~{row['approx_gb']} GB){' resume' if resumed else ''}", file=sys.stderr, flush=True)
    _download(row, dest)
    if not is_ready(key, dest):
        raise RuntimeError(f"download finished but {dest} is incomplete")
    _stamp(key, dest)
    return {"status": "downloaded", **base}


def _record(volume: str, report: dict[str, Any]) -> None:
    try:
        import db
    except Exception:  # noqa: BLE001
        return
    if not db.enabled():
        return
    for key, row in report.items():
        db.volume_status(volume, key, row)


def status(root: Optional[str] = None) -> dict[str, Any]:
    paths = layout(root)
    models = {
        key: {
            "repo": row["id"],
            "license": row["license"],
            "tasks": list(row["tasks"]),
            "default": bool(row.get("default")),
            "path": paths[key],
            "ready": is_ready(key, paths[key]),
            "approx_gb": row["approx_gb"],
        }
        for key, row in MODELS.items()
    }
    return {
        "volume": paths["root"],
        "free_gb": round(_free_gb(paths["root"]), 1),
        "default_gb": catalog.total_gb(DEFAULT_MODELS),
        "all_gb": catalog.total_gb(),
        "missing_default_gb": sum(m["approx_gb"] for k, m in models.items() if m["default"] and not m["ready"]),
        "models": models,
    }


def prefetch(models: Any = None, *, root: Optional[str] = None) -> dict[str, Any]:
    """Download missing allowlisted weights onto the network volume. Idempotent and resumable."""
    try:
        chosen = select_models(models)
    except ValueError as exc:
        return {"ok": False, "status": "error", "kind": "prefetch", "text": str(exc), "models": {}}
    paths = layout(root)
    _configure_hf(paths)
    need_gb = sum(MODELS[k]["approx_gb"] for k in chosen if not is_ready(k, paths[k]))
    free_gb = _free_gb(paths["root"])
    if free_gb >= 0 and need_gb and free_gb < need_gb + 5:
        print(f"[prefetch] warning: ~{need_gb} GB needed, {free_gb:.1f} GB free on {paths['root']}", file=sys.stderr, flush=True)
    report: dict[str, Any] = {}
    for key in chosen:
        if MODELS[key].get("manual"):
            # Placed by scripts/fetch_film_models.py (bf16 conversion); never pulled raw here.
            report[key] = {"status": "cached" if is_ready(key, paths[key]) else "manual",
                           "note": "run scripts/fetch_film_models.py on a pod with the volume mounted"}
            continue
        try:
            report[key] = _fetch(key, paths[key])
        except Exception as exc:  # noqa: BLE001
            print(f"[prefetch] {key} failed: {_public_error(exc)}", file=sys.stderr, flush=True)
            report[key] = {
                "status": "error",
                "error": _public_error(exc),
                "path": paths[key],
                "repo": MODELS[key]["id"],
                "license": MODELS[key]["license"],
                "approx_gb": MODELS[key]["approx_gb"],
            }
    _record(paths["root"], report)
    ready = [k for k, r in report.items() if r["status"] in ("cached", "downloaded")]
    failed = [k for k, r in report.items() if r["status"] == "error"]
    text = ("Volume sẵn sàng / ready: " + ", ".join(ready)) if ready else "Không tải được model nào / nothing downloaded."
    if failed:
        text += ". Lỗi / failed: " + ", ".join(failed)
    return {
        "ok": not failed,
        "status": "success" if not failed else "error",
        "text": text,
        "kind": "prefetch",
        "volume": paths["root"],
        "free_gb": round(_free_gb(paths["root"]), 1),
        "models": report,
        "meta": {"provider": "aimx_runpod_worker", "action": "prefetch"},
    }


def _print_list(body: dict[str, Any]) -> None:
    print(f"volume {body['volume']}  free {body['free_gb']} GB  defaults ~{body['default_gb']} GB  all ~{body['all_gb']} GB")
    for key, row in body["models"].items():
        flag = "ready" if row["ready"] else "missing"
        opt = "default " if row["default"] else "optional"
        print(f"  {key:24} {opt} {flag:7} ~{row['approx_gb']:>3} GB  {row['license']:10} {row['repo']}  -> {row['path']}")


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download AI Markets worker models onto the RunPod network volume")
    parser.add_argument("--models", default="", help="Comma-separated keys or aliases, 'default' or 'all'. Keys: " + ",".join(MODELS))
    parser.add_argument("--root", default="", help="Volume path (default: AIMX_VOLUME or /runpod-volume)")
    parser.add_argument("--list", action="store_true", help="Only show layout, sizes and what is already on the volume")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args(argv)
    if args.list:
        body = status(args.root or None)
        if args.json:
            print(json.dumps(body, ensure_ascii=False, indent=2))
        else:
            _print_list(body)
        return 0
    result = prefetch(args.models or None, root=args.root or None)
    print(json.dumps({k: result.get(k) for k in ("ok", "text", "volume", "free_gb", "models")}, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
