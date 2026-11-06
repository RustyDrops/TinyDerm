"""Codex-directed TinyDerm campaign loop with a deterministic local runner."""
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
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\rusty\Downloads\TinyDerm\.venv\Scripts\python.exe")
MANIFEST = Path(r"C:\Users\rusty\Downloads\TinyDerm\data\splits\split_manifest.csv")
HUNCH_CACHE = ROOT / "training-review" / "hunch-isic2019-expanded" / "embeddings.sqlite"
TRAINER = ROOT / "train_melanoma_multitask_cv.py"
DIRECTOR_PROMPT = ROOT / "tinyderm_research" / "director_v2.md"
DECISION_SCHEMA = ROOT / "tinyderm_research" / "decision_schema.json"
CAMPAIGN_DIR = ROOT / "tinyderm_research" / "campaign"
STATE_FILE = ROOT / "tinyderm_research" / "controller" / "state.json"
CONTROL_FILE = ROOT / "tinyderm_research" / "controller" / "control.json"
LOCK_FILE = ROOT / "tinyderm_research" / "controller" / "campaign.lock"
MODEL = "active ChatGPT session (in-chat director)"
MAX_HOURS = 33.0
MAX_RUNS = 18
MAX_DIRECTOR_CALLS = 25
MAX_DIAGNOSTICS = 4
ACCURACY_GOAL = 0.93
GAIN_THRESHOLD = 0.01
CLASSES = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
CONFIG_KEYS = {"epochs", "image_size", "batch_size", "seed", "unfreeze_blocks", "distill_weight",
               "class_weight_strength", "backbone_lr", "head_lr", "confidence_weighted_distillation"}
DINO_PATCH_SIZE = 14
PHASES = {"baseline", "teacher_audit", "distillation", "resolution", "class_balance", "seed_confirmation", "diagnostic"}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def default_state() -> dict[str, Any]:
    return {"version": 2, "status": "idle", "campaign_id": None, "campaign_started_utc": None,
            "budget_hours": MAX_HOURS, "budget_runs": MAX_RUNS, "director_model": MODEL,
            "director_calls": 0, "diagnostics": 0, "completed_runs": 0, "no_gain_streak": 0,
            "spent_training_seconds": 0.0, "latest_run_id": None, "best_run_id": None,
            "best_comparable": None, "runs": [], "incomplete_runs": [], "events": [], "blockers": []}


def load_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return default_state()
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    if state.get("version") not in (1, 2):
        raise ValueError("Unknown campaign state version; refusing to continue")
    if state.get("version") == 1:
        state = default_state() | state
        state["version"] = 2
    return state


def save_state(state: dict[str, Any], event: str | None = None) -> None:
    if event:
        state.setdefault("events", []).append({"at_utc": utcnow(), "message": event})
        state["events"] = state["events"][-500:]
    atomic_json(STATE_FILE, state)


def control_request() -> str | None:
    if not CONTROL_FILE.is_file():
        return None
    try:
        return json.loads(CONTROL_FILE.read_text(encoding="utf-8")).get("request")
    except (OSError, json.JSONDecodeError):
        return "stop"


def acquire_lock() -> None:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(LOCK_FILE.read_text(encoding="ascii").strip())
        except (ValueError, OSError):
            raise RuntimeError("Campaign lock exists and cannot be proven stale")
        if pid_is_alive(pid):
            raise RuntimeError(f"Campaign supervisor already running (PID {pid})")
        LOCK_FILE.unlink(missing_ok=True)
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(str(os.getpid()))


def release_lock() -> None:
    try:
        if LOCK_FILE.read_text(encoding="ascii").strip() == str(os.getpid()):
            LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def pid_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # On Windows, os.kill(pid, 0) reports ERROR_INVALID_PARAMETER (87)
        # for a PID that no longer exists instead of raising ProcessLookupError.
        missing_codes = {87, 1168}  # ERROR_INVALID_PARAMETER, ERROR_NOT_FOUND
        if os.name == "nt" and (getattr(exc, "winerror", None) in missing_codes or exc.errno in missing_codes):
            return False
        # Any other OS error is ambiguous; fail closed and keep the lock.
        return True


def elapsed_hours(state: dict[str, Any]) -> float:
    if not state.get("campaign_started_utc"):
        return 0.0
    started = dt.datetime.fromisoformat(state["campaign_started_utc"])
    return max(0.0, (dt.datetime.now(dt.timezone.utc) - started).total_seconds() / 3600)


def validate_decision(value: Any, state: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads(DECISION_SCHEMA.read_text(encoding="utf-8"))
    required = set(schema["required"])
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Director output does not match the controller decision schema keys")
    if value["decision"] not in {"RUN_DIAGNOSTIC", "RUN_NEXT_EXPERIMENT", "REPLAN_STRATEGY", "ESCALATE", "PAUSE_FOR_REVIEW", "RETAIN_CURRENT_BEST", "COMPLETE"}:
        raise ValueError("Unsupported decision action")
    if value["accuracy_goal"] != ACCURACY_GOAL:
        raise ValueError("Director changed the fixed 93% aspiration")
    if value["decision"] in {"RUN_NEXT_EXPERIMENT", "REPLAN_STRATEGY", "RETAIN_CURRENT_BEST"}:
        candidates = value["candidates"]
        if len(candidates) != 3:
            raise ValueError("A run decision must provide three competing hypotheses")
        selected = value["selected_hypothesis_id"]
        candidate = next((c for c in candidates if c["hypothesis_id"] == selected), None)
        if candidate is None:
            raise ValueError("Selected hypothesis is not in the candidate list")
        if candidate["phase"] not in PHASES - {"diagnostic"}:
            raise ValueError("Selected experiment phase is not implemented")
        diff = candidate["config_diff"]
        if set(diff) != CONFIG_KEYS:
            raise ValueError("Config diff keys differ from the supported allowlist")
        changes = {k: v for k, v in diff.items() if v is not None}
        if len(changes) > 2:
            raise ValueError("More than two config changes in one experiment")
        ranges = {"epochs": {1, 2, 3, 4, 5}, "image_size": {224, 336, 448}, "batch_size": {16, 32},
                  "unfreeze_blocks": {2, 4}, "distill_weight": {0.0, 0.05, 0.1},
                  "class_weight_strength": {0.5, 0.75}, "backbone_lr": {1e-5, 2e-5}, "head_lr": {0.001}}
        if "image_size" in changes and changes["image_size"] % DINO_PATCH_SIZE:
            raise ValueError(f"image_size must be divisible by the DINOv2 patch size ({DINO_PATCH_SIZE})")
        for key, allowed in ranges.items():
            if key in changes and changes[key] not in allowed:
                raise ValueError(f"Unsupported value for {key}: {changes[key]}")
        if "confidence_weighted_distillation" in changes and not isinstance(changes["confidence_weighted_distillation"], bool):
            raise ValueError("confidence_weighted_distillation must be boolean")
        if "seed" in changes and not (1 <= changes["seed"] <= 2_000_000_000):
            raise ValueError("Seed is outside the permitted range")
        if state["no_gain_streak"] >= 3:
            last_phase = state["runs"][-1]["phase"] if state["runs"] else None
            if value["decision"] != "REPLAN_STRATEGY" or not value["plateau_strategy_pivot"]:
                raise ValueError("Three-run plateau requires a logged strategy pivot")
            if candidate["phase"] == last_phase:
                raise ValueError("Plateau pivot must change the research phase")
        if not candidate["rationale"].strip() or not candidate["falsification_criteria"]:
            raise ValueError("Selected hypothesis lacks rationale or falsification criteria")
    elif value["decision"] == "RUN_DIAGNOSTIC" and not value["diagnosis"]["primary"].strip():
        raise ValueError("Diagnostic decision lacks a grounded diagnosis")
    return value


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    recent = [{k: run.get(k) for k in ("run_id", "phase", "hypothesis_id", "config", "metrics", "credible_gain", "wall_seconds")}
              for run in state["runs"][-5:]]
    return {"campaign_id": state.get("campaign_id"), "iteration": state["completed_runs"] + 1,
            "completed_full_runs": state["completed_runs"], "hours_elapsed": round(elapsed_hours(state), 2),
            "hours_remaining": round(max(0, state["budget_hours"] - elapsed_hours(state)), 2),
            "runs_remaining": max(0, state["budget_runs"] - state["completed_runs"]),
            "director_calls_remaining": max(0, MAX_DIRECTOR_CALLS - state["director_calls"]),
            "no_gain_streak": state["no_gain_streak"], "latest_run_id": state["latest_run_id"],
            "best_comparable": state["best_comparable"], "recent_runs": recent,
            "tried_configs": [r["config_hash"] for r in state["runs"] + state.get("incomplete_runs", [])],
            "training_device": "CUDA if available; CPU only after measured time-fit preflight",
            "holdout_status": "consumed and strictly excluded; never read or scored",
            "runtime_source": "Prior grouped 3-fold, 3-epoch 448px CUDA run: 71.3 minutes on 23,890 development images; current campaign uses the original 9,009-image cohort.",
            "hunch_evidence": "ImageGemma/Hunch logistic teacher is generated by nested lesion-grouped OOF fits; student receives actual image pixels. It has not shown consistent macro-F1 gains. No external neural teacher or new weights approved."}


def call_director(state: dict[str, Any]) -> dict[str, Any]:
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("Codex CLI is not on PATH")
    call_id = f"decision_{state['director_calls'] + 1:03d}"
    decision_dir = CAMPAIGN_DIR / "decisions"
    decision_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = decision_dir / f"{call_id}_prompt.md"
    output_path = decision_dir / f"{call_id}.json"
    event_path = decision_dir / f"{call_id}_events.jsonl"
    stable = DIRECTOR_PROMPT.read_text(encoding="utf-8")
    compact = json.dumps(compact_state(state), ensure_ascii=False, indent=2)
    prompt = ("Follow the stable TinyDerm research director instructions below. You are only choosing a decision; "
              "do not inspect files, call tools, launch processes, or access data. Use the supplied current state as the "
              "only source of campaign metrics. Output exactly one schema-conforming JSON decision.\n\n"
              "<stable_director_instructions>\n" + stable + "\n</stable_director_instructions>\n\n"
              "<current_aggregate_state>\n" + compact + "\n</current_aggregate_state>\n\n"
              "Current state is authoritative. All 9,009 original ISIC 2018 development rows are the allowed research cohort; "
              "the consumed holdout is permanently excluded. Stay within the supported config keys and phases in the JSON schema. "
              "Never request external assets, architectures, packages, data uploads, or holdout evaluation. The local controller "
              "will independently validate your decision before any action.")
    prompt_path.write_text(prompt, encoding="utf-8")
    args = [codex, "exec", "--ephemeral", "--skip-git-repo-check", "--model", MODEL, "--sandbox", "read-only",
            "--output-schema", str(DECISION_SCHEMA), "--output-last-message", str(output_path), "-"]
    state["director_calls"] += 1
    try:
        with event_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(args, cwd=ROOT, input=prompt, text=True, encoding="utf-8", stdout=log, stderr=subprocess.STDOUT,
                                  timeout=300, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Codex decision timed out after five minutes; campaign will pause.") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"Codex decision call failed (exit {proc.returncode}); see {event_path}")
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    return validate_decision(raw, state)


def score_report(path: Path, expected_device: str = "cuda") -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    result = {"scope": summary.get("scope"), "device": summary.get("device"), "images": summary.get("images"),
              "development_identity_sha256": summary.get("development_identity_sha256"),
              "elapsed_seconds": summary.get("elapsed_seconds"),
              "settings": summary.get("settings"), "arms": {}}
    if result["device"] != expected_device:
        raise ValueError(f"Training report device is {result['device']}; expected {expected_device} based on preflight")
    for arm, epochs in summary["aggregate_oof"].items():
        metric = epochs[-1]
        per_class = metric["per_class"]
        support = sum(float(per_class[c]["support"]) for c in CLASSES)
        accuracy = sum(float(per_class[c]["recall"]) * float(per_class[c]["support"]) for c in CLASSES) / support
        result["arms"][arm] = {"macro_f1": float(metric["macro_f1"]),
                               "balanced_accuracy": float(metric["balanced_accuracy"]),
                               "accuracy": accuracy,
                               "per_class_recall": {c: float(per_class[c]["recall"]) for c in CLASSES},
                               "supports": {c: int(per_class[c]["support"]) for c in CLASSES}}
    result["selected_arm"] = max(result["arms"], key=lambda a: result["arms"][a]["macro_f1"])
    result["selected_metrics"] = result["arms"][result["selected_arm"]]
    return result


def config_signature(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def estimated_minutes(state: dict[str, Any], config: dict[str, Any]) -> float:
    # Conservative: baseline estimate is the full 71-minute observed expanded-data run,
    # then scale only for requested epochs and square input size. Do not assume fewer
    # development rows make an experiment faster.
    return 71.3 * config["epochs"] / 3 * (config["image_size"] / 448) ** 2 + 15


def run_training(state: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    selected = next(c for c in decision["candidates"] if c["hypothesis_id"] == decision["selected_hypothesis_id"])
    defaults = {"epochs": 3, "image_size": 448, "batch_size": 32, "seed": 52711, "unfreeze_blocks": 4,
                "distill_weight": 0.05, "class_weight_strength": 0.5, "backbone_lr": 2e-5, "head_lr": 0.001,
                "confidence_weighted_distillation": False}
    config = defaults | {k: v for k, v in selected["config_diff"].items() if v is not None}
    signature = config_signature(config)
    if signature in {r["config_hash"] for r in state["runs"] + state.get("incomplete_runs", [])}:
        raise ValueError("Exact campaign configuration duplicate rejected")
    planned_device = state.get("planned_device", "cuda")
    estimate_factor = 10 if planned_device == "cpu" else 1
    est = estimated_minutes(state, config) * estimate_factor
    remaining = max(0.0, state["budget_hours"] * 60 - elapsed_hours(state) * 60)
    if est + 10 > remaining:
        raise TimeoutError(f"Estimated {est + 10:.1f} minutes with margin exceeds {remaining:.1f} minutes remaining")
    run_id = "run_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = CAMPAIGN_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "config.json", {"run_id": run_id, "hypothesis": selected, "effective_config": config,
                                           "estimated_minutes": est, "planned_device": planned_device,
                                           "cpu_estimate_multiplier": estimate_factor if planned_device == "cpu" else None,
                                           "trainer_sha256": sha256_file(TRAINER),
                                           "safe_data_sha256": sha256_file(ROOT / "tinyderm_safe_data.py"),
                                           "director_model": MODEL, "created_at_utc": utcnow()})
    command = [str(PYTHON), str(TRAINER), "--manifest", str(MANIFEST), "--hunch-cache", str(HUNCH_CACHE),
               "--output", str(run_dir / "artifacts"), "--folds", "3",
               "--epochs", str(config["epochs"]), "--batch-size", str(config["batch_size"]),
               "--image-size", str(config["image_size"]), "--workers", "0", "--unfreeze-blocks", str(config["unfreeze_blocks"]),
               "--seed", str(config["seed"]), "--distill-weight", str(config["distill_weight"]),
               "--class-weight-strength", str(config["class_weight_strength"]),
               "--backbone-lr", str(config["backbone_lr"]), "--head-lr", str(config["head_lr"])]
    if config["confidence_weighted_distillation"]:
        command.append("--confidence-weighted-distillation")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0" if planned_device == "cuda" else ""
    log_path = run_dir / "training.log"
    resource_path = run_dir / "resource_samples.jsonl"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state["active_run_id"] = run_id
        state["active_pid"] = proc.pid
        state["active_config_hash"] = signature
        state["active_config"] = config
        save_state(state, f"Started training subprocess PID {proc.pid} for {run_id}.")
        last_control_seen = None
        last_resource_sample = 0.0
        while proc.poll() is None:
            request = control_request()
            if request in {"pause", "stop"} and request != last_control_seen:
                # Finish the current experiment so its fold checkpoints and report remain coherent.
                state["pending_control"] = request
                save_state(state, f"{request.title()} requested; allowing current experiment to finish safely.")
                last_control_seen = request
            if time.monotonic() - last_resource_sample >= 60:
                sampler = shutil.which("nvidia-smi")
                if sampler:
                    sample = subprocess.run([sampler, "--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw",
                                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
                    record = {"at_utc": utcnow(), "pid": proc.pid, "raw": sample.stdout.strip(),
                              "error": sample.stderr[-300:] if sample.returncode else None}
                else:
                    record = {"at_utc": utcnow(), "pid": proc.pid, "error": "nvidia-smi unavailable"}
                with resource_path.open("a", encoding="utf-8") as samples:
                    samples.write(json.dumps(record) + "\n")
                last_resource_sample = time.monotonic()
            if elapsed_hours(state) >= state["budget_hours"]:
                proc.terminate()
                break
            time.sleep(10)
    code = proc.wait()
    duration = time.monotonic() - started
    state["spent_training_seconds"] += duration
    state.pop("active_pid", None)
    state.pop("active_run_id", None)
    state.pop("active_config_hash", None)
    state.pop("active_config", None)
    if code != 0:
        state.setdefault("incomplete_runs", []).append({"run_id": run_id, "config_hash": signature,
                                                         "config": config, "wall_seconds": duration,
                                                         "reason": f"Training process exit code {code}",
                                                         "artifact_path": str(run_dir / "artifacts")})
        save_state(state, f"Marked {run_id} incomplete; its configuration will not be repeated.")
        raise RuntimeError(f"Training process failed with exit {code}; see {log_path}")
    report_path = run_dir / "artifacts" / "crossvalidation.json"
    report = score_report(report_path, expected_device=planned_device)
    expected_outputs = [run_dir / "artifacts" / f"fold{fold}_{arm}_epoch{config['epochs']}.pt"
                       for fold in range(1, 4) for arm in ("baseline", "hunch_assisted")]
    expected_outputs += [run_dir / "artifacts" / f"fold{fold}_oof_predictions.npz" for fold in range(1, 4)]
    missing = [str(p) for p in expected_outputs if not p.is_file()]
    if missing:
        raise RuntimeError("Run report is incomplete; missing expected fold artifacts")
    report.update({"run_id": run_id, "phase": selected["phase"], "hypothesis_id": selected["hypothesis_id"],
                   "config_hash": signature, "wall_seconds": duration, "log_path": str(log_path),
                   "code_sha256": sha256_file(TRAINER)})
    atomic_json(run_dir / "run_report.json", report)
    state["completed_runs"] += 1
    state["latest_run_id"] = run_id
    previous = state["best_comparable"]
    score = report["selected_metrics"]
    if previous is None:
        state["best_comparable"] = {"run_id": run_id, **score}
        state["best_run_id"] = run_id
        state["no_gain_streak"] = 0
        report["credible_gain"] = None
    else:
        delta_f1 = score["macro_f1"] - previous["macro_f1"]
        delta_acc = score["accuracy"] - previous["accuracy"]
        regressions = {c: score["per_class_recall"][c] - previous["per_class_recall"][c] for c in CLASSES}
        material = min(regressions.values()) < -0.05
        credible = delta_f1 >= GAIN_THRESHOLD and not material
        report.update({"delta_macro_f1": delta_f1, "delta_accuracy": delta_acc,
                       "per_class_recall_delta": regressions, "material_recall_regression": material,
                       "credible_gain": credible})
        if credible:
            state["best_comparable"] = {"run_id": run_id, **score}
            state["best_run_id"] = run_id
            state["no_gain_streak"] = 0
        else:
            state["no_gain_streak"] += 1
    state["runs"].append({"run_id": run_id, "phase": selected["phase"], "hypothesis_id": selected["hypothesis_id"],
                          "config_hash": signature, "config": config, "metrics": score,
                          "credible_gain": report.get("credible_gain"), "wall_seconds": duration,
                          "artifact_path": str(run_dir / "artifacts")})
    state["runs"] = state["runs"][-100:]
    atomic_json(run_dir / "run_report.json", report)
    save_state(state, f"Completed {run_id}; macro-F1={score['macro_f1']:.4f}, accuracy={score['accuracy']:.4f}, device={report['device']}.")
    return report


def write_escalation(state: dict[str, Any], decision: dict[str, Any]) -> None:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    dossier = {"campaign_id": state.get("campaign_id"), "decision": decision,
               "recent_runs": state["runs"][-5:], "best_comparable": state["best_comparable"],
               "hours_remaining": max(0, state["budget_hours"] - elapsed_hours(state)),
               "questions_for_review": ["Which evidence-backed phase best distinguishes the remaining bottleneck?",
                                        "What outcome would falsify that hypothesis on development-only grouped CV?",
                                        "Can the proposed experiment run with cached assets and the remaining local time budget?"]}
    atomic_json(CAMPAIGN_DIR / "escalation_dossier.json", dossier)


def run_campaign(resume: bool = False) -> int:
    print("Detached overnight director mode is disabled. This ChatGPT session will choose each experiment; use `run-one --decision-file PATH` for one validated run.")
    return 2

    from tinyderm_controller import preflight
    acquire_lock()
    try:
        state = load_state()
        if state["status"] == "running":
            if pid_is_alive(state.get("active_pid")):
                raise RuntimeError(f"Recorded training process PID {state['active_pid']} is still alive; refusing duplicate supervisor")
            if state.get("active_run_id"):
                state.setdefault("incomplete_runs", []).append({"run_id": state["active_run_id"],
                    "config_hash": state.get("active_config_hash"), "config": state.get("active_config"),
                    "reason": "Supervisor ended unexpectedly; run was not scored.",
                    "artifact_path": str(CAMPAIGN_DIR / "runs" / state["active_run_id"] / "artifacts")})
            state.pop("active_pid", None)
            state.pop("active_run_id", None)
            state.pop("active_config_hash", None)
            state.pop("active_config", None)
            state["status"] = "paused"
            save_state(state, "Recovered an interrupted supervisor; incomplete run artifacts are preserved and unscored.")
        if state["status"] == "paused" and not resume:
            raise RuntimeError("Campaign is paused; use resume")
        if resume and (state["status"] != "paused" or not state.get("campaign_started_utc")):
            raise RuntimeError("There is no paused campaign to resume")
        report = preflight()
        atomic_json(CAMPAIGN_DIR / "latest_preflight.json", report)
        if not report["passed"]:
            state["status"] = "blocked_preflight"
            state["blockers"] = report["blockers"]
            save_state(state, "Campaign refused: preflight hard gate failed.")
            print(json.dumps({"started": False, "blockers": report["blockers"]}, indent=2))
            return 2
        if not resume:
            state = default_state()
            state["campaign_id"] = "campaign_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            state["campaign_started_utc"] = utcnow()
        state["status"] = "running"
        atomic_json(CONTROL_FILE, {"request": None, "updated_at_utc": utcnow()})
        save_state(state, "Owner-started campaign supervisor.")
        while True:
            req = control_request()
            if req == "stop":
                state["status"] = "stopped"
                save_state(state, "Campaign stopped at a safe experiment boundary.")
                return 0
            if req == "pause":
                state["status"] = "paused"
                save_state(state, "Campaign paused at a safe experiment boundary.")
                return 0
            hours_left = state["budget_hours"] - elapsed_hours(state)
            if hours_left <= 0 or state["completed_runs"] >= state["budget_runs"] or state["director_calls"] >= MAX_DIRECTOR_CALLS:
                state["status"] = "complete"
                save_state(state, "Campaign reached the deterministic time, run, or director-call ceiling.")
                return 0
            try:
                decision = call_director(state)
            except Exception as exc:
                state["status"] = "paused"
                state["last_error"] = str(exc)
                save_state(state, f"Campaign paused because the Codex director call failed: {exc}")
                return 2
            atomic_json(CAMPAIGN_DIR / "decisions" / f"decision_{state['director_calls']:03d}.json", decision)
            save_state(state, f"Codex director decision {decision['decision']}.")
            if decision["decision"] in {"PAUSE_FOR_REVIEW", "ESCALATE", "COMPLETE"}:
                if decision["decision"] == "ESCALATE":
                    write_escalation(state, decision)
                state["status"] = "paused" if decision["decision"] != "COMPLETE" else "complete"
                save_state(state, decision.get("stop_or_escalation_reason") or decision["decision"])
                return 0
            if decision["decision"] == "RUN_DIAGNOSTIC":
                state["diagnostics"] += 1
                if state["diagnostics"] > MAX_DIAGNOSTICS:
                    state["status"] = "paused"
                    save_state(state, "Diagnostic call ceiling reached without an executable experiment.")
                    return 0
                diagnostic = preflight()
                atomic_json(CAMPAIGN_DIR / "diagnostics" / f"diagnostic_{state['diagnostics']:02d}.json", diagnostic)
                if not diagnostic["passed"]:
                    state["status"] = "paused"
                    state["blockers"] = diagnostic["blockers"]
                    save_state(state, "Diagnostic detected a failed hard gate.")
                    return 2
                continue
            if decision["decision"] == "REPLAN_STRATEGY":
                save_state(state, decision["plateau_strategy_pivot"] or "Strategy pivot recorded after plateau.")
            try:
                run_training(state, decision)
            except Exception as exc:
                state["status"] = "paused"
                state["last_error"] = str(exc)
                save_state(state, f"Campaign paused on runner failure: {exc}")
                return 2
    finally:
        release_lock()


def run_one_from_chat(decision_file: Path) -> int:
    """Run exactly one validated experiment selected by the active ChatGPT session."""
    from tinyderm_controller import preflight
    preflight_report = preflight(director_mode="in_chat")
    atomic_json(CAMPAIGN_DIR / "latest_preflight.json", preflight_report)
    if not preflight_report["passed"]:
        print(json.dumps({"started": False, "blockers": preflight_report["blockers"]}, indent=2))
        return 2
    decision_dir = (CAMPAIGN_DIR / "decisions").resolve()
    decision_path = decision_file.resolve()
    if not decision_path.is_relative_to(decision_dir) or not decision_path.is_file():
        raise ValueError("Decision file must exist inside tinyderm_research/campaign/decisions")
    acquire_lock()
    try:
        state = load_state()
        if state["status"] == "running":
            if pid_is_alive(state.get("active_pid")):
                raise RuntimeError(f"Training PID {state.get('active_pid')} is active; refusing a duplicate run")
            if state.get("active_run_id"):
                state.setdefault("incomplete_runs", []).append({"run_id": state["active_run_id"],
                    "config_hash": state.get("active_config_hash"), "config": state.get("active_config"),
                    "reason": "Prior in-chat supervisor ended unexpectedly; run was not scored.",
                    "artifact_path": str(CAMPAIGN_DIR / "runs" / state["active_run_id"] / "artifacts")})
                state.pop("active_pid", None)
                state.pop("active_run_id", None)
                state.pop("active_config_hash", None)
                state.pop("active_config", None)
                state["status"] = "paused"
        if state.get("campaign_id") is None or state["status"] in {"idle", "complete", "stopped", "blocked_preflight"}:
            state = default_state()
            state["campaign_id"] = "campaign_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            state["campaign_started_utc"] = utcnow()
        if state["status"] not in {"paused", "running"}:
            raise RuntimeError(f"Campaign cannot accept a decision in status {state['status']}")
        state["director_model"] = MODEL
        state["last_error"] = None
        raw = json.loads(decision_path.read_text(encoding="utf-8"))
        decision = validate_decision(raw, state)
        if decision["decision"] not in {"RUN_NEXT_EXPERIMENT", "REPLAN_STRATEGY", "RETAIN_CURRENT_BEST"}:
            raise ValueError("This command accepts only a validated full experiment decision")
        if state["completed_runs"] >= state["budget_runs"] or state["director_calls"] >= MAX_DIRECTOR_CALLS:
            raise RuntimeError("Campaign run or director-decision ceiling reached")
        state["planned_device"] = "cuda" if preflight_report["gpu"].get("available") else "cpu"
        chosen = next(c for c in decision["candidates"] if c["hypothesis_id"] == decision["selected_hypothesis_id"])
        defaults = {"epochs": 3, "image_size": 448, "batch_size": 32, "seed": 52711, "unfreeze_blocks": 4,
                    "distill_weight": 0.05, "class_weight_strength": 0.5, "backbone_lr": 2e-5, "head_lr": 0.001,
                    "confidence_weighted_distillation": False}
        chosen_config = defaults | {k: v for k, v in chosen["config_diff"].items() if v is not None}
        fallback_factor = 10 if state["planned_device"] == "cpu" else 1
        estimate = estimated_minutes(state, chosen_config) * fallback_factor
        remaining = max(0.0, state["budget_hours"] * 60 - elapsed_hours(state) * 60)
        if estimate + 10 > remaining:
            raise TimeoutError(f"CUDA unavailable; CPU fallback estimate is {estimate + 10:.1f} minutes, exceeding {remaining:.1f} minutes remaining. No training was launched.")
        state["director_calls"] += 1
        state["status"] = "running"
        atomic_json(CONTROL_FILE, {"request": None, "updated_at_utc": utcnow()})
        decision_path_copy = CAMPAIGN_DIR / "decisions" / f"decision_{state['director_calls']:03d}.json"
        atomic_json(decision_path_copy, decision)
        save_state(state, f"Active ChatGPT session selected {decision['selected_hypothesis_id']}; launching one experiment.")
        try:
            report = run_training(state, decision)
        except Exception as exc:
            state["status"] = "paused"
            state["last_error"] = str(exc)
            save_state(state, f"Campaign paused on supervised run failure: {exc}")
            raise
        state["status"] = "paused"
        save_state(state, "Run complete; awaiting the next evidence-based decision from this ChatGPT session.")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        release_lock()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume", "run-one", "pause", "stop", "status", "report", "compare", "preflight"))
    parser.add_argument("--decision-file", type=Path)
    args = parser.parse_args()
    if args.action in ("start", "resume"):
        return run_campaign(resume=args.action == "resume")
    if args.action == "run-one":
        if args.decision_file is None:
            parser.error("run-one requires --decision-file")
        return run_one_from_chat(args.decision_file)
    state = load_state()
    if args.action in ("pause", "stop"):
        atomic_json(CONTROL_FILE, {"request": args.action, "updated_at_utc": utcnow()})
        print(f"{args.action} requested; the active experiment is allowed to finish before control takes effect.")
        return 0
    if args.action == "status":
        state["hours_elapsed"] = round(elapsed_hours(state), 2)
        state["hours_remaining"] = round(max(0, state["budget_hours"] - elapsed_hours(state)), 2)
        print(json.dumps(state, indent=2))
        return 0
    if args.action == "report":
        print(json.dumps({"campaign_id": state.get("campaign_id"), "status": state["status"],
                          "run_count": state["completed_runs"], "latest_run_id": state["latest_run_id"],
                          "best_run_id": state["best_run_id"], "best_comparable": state["best_comparable"],
                          "runs": state["runs"]}, indent=2))
        return 0
    if args.action == "compare":
        print(json.dumps({"frozen_historical_best": None, "campaign_best": state["best_comparable"],
                          "historical_comparability": "unknown; earlier development-validation identities and protocols differ or were repeatedly selected",
                          "current_vs_frozen_delta": None}, indent=2))
        return 0
    from tinyderm_controller import preflight
    result = preflight()
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

