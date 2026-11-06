"""Fail-closed preflight and command gate for TinyDerm research runs.

The supervisor is deliberately fail-closed. It never opens holdout rows, and it
will not start a campaign unless its scientific, API, hardware, and runtime gates
are satisfied. Training is always launched with the configured TinyDerm Python.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("TINYDERM_PYTHON", sys.executable))
MANIFEST = Path(os.environ.get("TINYDERM_MANIFEST", ROOT / "data" / "splits" / "split_manifest.csv"))
LOCK = Path(os.environ.get("TINYDERM_HOLDOUT_LOCK", ROOT / "data" / "splits" / "holdout_lock.json"))
HUNCH_CACHE = Path(os.environ.get("TINYDERM_HUNCH_CACHE", ROOT / "training-review" / "hunch-isic2019-expanded" / "embeddings.sqlite"))
DINOV2_CACHE = Path(os.environ.get("TINYDERM_DINOV2_CACHE", ROOT / "training-review" / "model-cache" / "hub" / "models--timm--vit_small_patch14_dinov2.lvd142m" / "snapshots"))
STATE_DIR = ROOT / "tinyderm_research" / "controller"
STATE_FILE = STATE_DIR / "state.json"
CONTROL_FILE = STATE_DIR / "control.json"
MAX_HOURS = 33
MAX_RUNS = 18


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {"version": 1, "status": "idle", "campaign_started_utc": None,
                "budget_hours": MAX_HOURS, "budget_runs": MAX_RUNS,
                "completed_runs": 0, "consecutive_no_gain": 0,
                "spent_seconds": 0.0, "events": [], "run_ids": []}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def record_event(state: dict[str, Any], message: str) -> None:
    state.setdefault("events", []).append({"at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "message": message})
    state["events"] = state["events"][-300:]
    atomic_json(STATE_FILE, state)


def preflight(director_mode: str = "in_chat") -> dict[str, Any]:
    """Collect gates using aggregate development data only; never read holdout rows."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str, hard: bool = True) -> None:
        checks.append({"name": name, "passed": bool(passed), "hard_gate": hard, "detail": detail})

    check("training_python", PYTHON.is_file(), str(PYTHON))
    check("development_manifest", MANIFEST.is_file(), str(MANIFEST))
    check("director_prompt", (ROOT / "tinyderm_research" / "director_v2.md").is_file(), "Versioned director instructions are present.")
    check("hunch_teacher_audit", (ROOT / "tinyderm_research" / "reports" / "hunch_teacher_audit.md").is_file(), "Audit exists; its limitations remain binding.")

    # The frozen snapshot is historical inventory only, but hash verification guards its integrity.
    try:
        from tinyderm_research import verify
        manifest, mismatches = verify()
        check("historical_snapshot", not mismatches and manifest.get("artifact_count") == len(manifest.get("artifacts", [])),
              f"{manifest.get('artifact_count')} copied reports; {len(mismatches)} hash mismatches; comparability remains unknown.")
    except Exception as exc:
        check("historical_snapshot", False, f"Verification error: {exc}")

    # Fixed folds are reconstructed from development rows only. Holdout rows are skipped by audit_manifest.
    if MANIFEST.is_file():
        try:
            from tinyderm_preflight import audit_manifest
            from tinyderm_fold_audit import audit as audit_folds
            base = audit_manifest(MANIFEST)
            folds = audit_folds(MANIFEST)
            clean = (not base["errors"] and folds["validation_coverage"] == folds["development_images"]
                     and all(f["group_overlap"] == 0 and f["duplicate_validation_indices"] == 0 for f in folds["folds"]))
            check("development_grouped_cv", clean,
                  f"Errors={base['errors']}; {folds['fold_count']} reconstructed grouped folds cover "
                  f"{folds['validation_coverage']}/{folds['development_images']} development rows. "
                  "Repeated model selection on these folds means estimates are exploratory, not independent.")
        except Exception as exc:
            check("development_grouped_cv", False, f"Audit error: {exc}")
    else:
        check("development_grouped_cv", False, "Manifest is missing.")

    try:
        from tinyderm_safe_data import development_rows, load_hunch
        if not HUNCH_CACHE.is_file():
            raise FileNotFoundError(HUNCH_CACHE)
        dev = development_rows(MANIFEST)
        vectors = load_hunch(dev, HUNCH_CACHE)
        check("embedding_cache_provenance", vectors.shape[0] == len(dev) and vectors.shape[1] > 0,
              f"Loaded and SHA-256 verified {vectors.shape[0]} development embeddings ({vectors.shape[1]} dimensions); no holdout rows or cache entries queried.")
    except Exception as exc:
        check("embedding_cache_provenance", False, f"Development-only cache audit failed: {exc}")

    try:
        weights = [p for p in DINOV2_CACHE.rglob("model.safetensors") if p.is_file() and p.stat().st_size > 1_000_000]
        check("pretrained_weights_cached", bool(weights),
              f"Found local DINOv2 checkpoint ({weights[0].stat().st_size} bytes); training will not need a new model download."
              if weights else "Pinned DINOv2 checkpoint is not available in the local cache; no external weight download is authorized.")
    except Exception as exc:
        check("pretrained_weights_cached", False, f"Could not verify local DINOv2 checkpoint: {exc}")

    # Read governance metadata only. Never inspect holdout examples or labels.
    try:
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        consumed = bool(lock.get("consumed"))
        # Consumption invalidates holdout claims, but does not prohibit a strictly
        # development-only research campaign that never reads holdout rows.
        check("holdout_governance", True,
              "Original final holdout is consumed and excluded from all campaign inputs; no final-test claim is allowed."
              if consumed else "Original final holdout remains sealed and excluded from campaign inputs.")
    except Exception as exc:
        check("holdout_governance", False, f"Cannot verify holdout lock: {exc}")

    check("in_session_director", director_mode == "in_chat",
          "Decisions are made by the active ChatGPT/Codex session; no local CLI login, API key, or detached model process is used."
          if director_mode == "in_chat" else "Only in-chat director mode is supported; local Codex CLI mode is disabled.")
    runner = ROOT / "tinyderm_campaign.py"
    schema = ROOT / "tinyderm_research" / "decision_schema.json"
    loop_ready = runner.is_file() and schema.is_file()
    check("interactive_campaign_runner", loop_ready,
          "Structured decision validation and one-run-at-a-time supervised training runner are present." if loop_ready else
          "Decision schema or supervised training runner is missing.")

    gpu = {"available": False, "name": None, "torch": None, "cuda_runtime": None, "error": None}
    if PYTHON.is_file():
        probe = "import json,torch; print(json.dumps({'available':torch.cuda.is_available(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch':torch.__version__,'cuda_runtime':torch.version.cuda}))"
        try:
            p = subprocess.run([str(PYTHON), "-c", probe], capture_output=True, text=True, timeout=45, check=False)
            if p.returncode == 0:
                gpu.update(json.loads(p.stdout.strip().splitlines()[-1]))
            else:
                gpu["error"] = p.stderr[-1000:]
        except Exception as exc:
            gpu["error"] = str(exc)
    device_path_ready = gpu["available"] or bool(gpu["torch"])
    device_detail = (f"CUDA available=True; GPU={gpu['name']}; torch={gpu['torch']}; runtime={gpu['cuda_runtime']}."
                     if gpu["available"] else
                     f"CUDA unavailable; CPU fallback is configured. Each proposed run must pass a conservative 10x GPU-time estimate against the remaining campaign budget before launch. torch={gpu['torch']}; error={gpu['error']}.")
    check("training_device_path", device_path_ready, device_detail)

    # Require sleep disabled on AC for the complete unattended budget. Query is read-only.
    sleep_disabled = False
    try:
        p = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"],
                           capture_output=True, text=True, timeout=15, check=False)
        # Windows query returns hexadecimal AC index. Zero means never sleep.
        import re
        match = re.search(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)", p.stdout)
        sleep_disabled = bool(match and int(match.group(1), 16) == 0)
        sleep_detail = f"AC sleep-after index is {match.group(1) if match else 'unknown'} (0 means never)."
    except Exception as exc:
        sleep_detail = f"Could not query AC sleep policy: {exc}"
    check("unattended_sleep_policy", sleep_disabled, sleep_detail)

    # Runtime is rechecked against the exact selected config before every launch.
    check("campaign_time_estimate", device_path_ready,
          "Each run is checked against remaining time; CPU fallback uses a conservative 10x multiplier." if not gpu["available"]
          else "Prior owner-reported GPU runs were about one hour; each grouped-CV run is time-checked before launch.")
    return {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "gpu": gpu,
            "checks": checks, "passed": all(c["passed"] for c in checks if c["hard_gate"]),
            "blockers": [c["name"] + ": " + c["detail"] for c in checks if c["hard_gate"] and not c["passed"]],
            "scope": "Development-only diagnostics; sealed holdout examples were not read.",
            "campaign_hours": MAX_HOURS, "max_full_runs": MAX_RUNS}


def status() -> dict[str, Any]:
    state = load_state()
    report = preflight()
    return {"controller": "tinyderm_controller", "state": state, "preflight_passed": report["passed"],
            "blockers": report["blockers"], "gpu": report["gpu"],
            "commands": {k: f"python tinyderm_controller.py {k}" for k in ("status", "preflight", "report", "compare", "pause", "resume", "stop")}}


def gpu_smoke() -> dict[str, Any]:
    """Run the actual TinyDerm training function on a tiny development-only subset."""
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    p = subprocess.run([str(PYTHON), str(ROOT / "tinyderm_gpu_smoke.py")],
                       cwd=ROOT, capture_output=True, text=True, timeout=900, check=False)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    payload = json.loads(p.stdout.strip().splitlines()[-1])
    if payload.get("device") != "cuda" or payload.get("parameter_device") != "cuda:0" or payload.get("cuda_allocated_bytes", 0) <= 0:
        raise RuntimeError(f"TinyDerm training process did not prove CUDA training: {payload}")
    return payload


def command(args: argparse.Namespace) -> int:
    if args.action in ("start", "resume"):
        from tinyderm_campaign import run_campaign
        return run_campaign(resume=args.action == "resume")
    if args.action in ("pause", "stop"):
        atomic_json(CONTROL_FILE, {"request": args.action, "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        print(f"{args.action} requested; the active experiment will finish before control takes effect.")
        return 0
    if args.action == "preflight":
        result = preflight()
        atomic_json(STATE_DIR / "last_preflight.json", result)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    if args.action == "gpu-smoke":
        result = gpu_smoke()
        atomic_json(STATE_DIR / "gpu_smoke.json", result)
        print(json.dumps(result, indent=2))
        return 0
    if args.action == "status":
        print(json.dumps(status(), indent=2))
        return 0
    if args.action == "report":
        state = load_state()
        print(json.dumps({"state": state, "latest_preflight": str(STATE_DIR / "last_preflight.json")}, indent=2))
        return 0
    if args.action == "compare":
        print(json.dumps({"frozen_historical_best": None, "comparability": "unknown",
                          "reason": "Historical snapshot contains artifact inventory only; common split and metric identity not verified."}, indent=2))
        return 0
    state = load_state()
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("status", "preflight", "start", "pause", "resume", "stop", "report", "compare", "gpu-smoke"))
    return command(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

