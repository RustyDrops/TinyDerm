"""SQLite Experiment Database manager for TinyDerm Lab."""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentDB:
    def __init__(self, db_path: str | Path = "runs/experiments.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    seed INTEGER,
                    status TEXT,
                    stage TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    config_json TEXT,
                    hardware_json TEXT,
                    holdout_status TEXT,
                    macro_f1 REAL,
                    balanced_accuracy REAL,
                    accuracy REAL,
                    cpu_p50_ms REAL,
                    cpu_p95_ms REAL,
                    fp32_size_mb REAL,
                    int8_size_mb REAL,
                    error_message TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS epochs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    phase TEXT,
                    epoch INTEGER,
                    train_loss REAL,
                    val_loss REAL,
                    train_f1 REAL,
                    val_f1 REAL,
                    train_balanced_acc REAL,
                    val_balanced_acc REAL,
                    learning_rate REAL,
                    images_per_sec REAL,
                    epoch_duration_sec REAL,
                    timestamp TEXT,
                    metrics_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    split TEXT,
                    image_id TEXT,
                    lesion_id TEXT,
                    true_class TEXT,
                    predicted_class TEXT,
                    probability_true REAL,
                    probability_predicted REAL,
                    confidence REAL,
                    entropy REAL,
                    correct INTEGER,
                    probabilities_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    timestamp TEXT,
                    cpu_percent REAL,
                    ram_percent REAL,
                    ram_rss_mb REAL,
                    gpu_util_percent REAL,
                    gpu_mem_used_mb REAL,
                    disk_used_gb REAL,
                    images_per_sec REAL,
                    stage TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    timestamp TEXT,
                    event_type TEXT,
                    stage TEXT,
                    message TEXT,
                    data_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    artifact_type TEXT,
                    file_path TEXT,
                    file_size_bytes INTEGER,
                    sha256 TEXT,
                    created_at TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            conn.commit()

    def create_run(self, run_id: str, seed: int, config: Dict[str, Any], hardware: Dict[str, Any]):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, seed, status, stage, start_time, config_json, hardware_json, holdout_status
                ) VALUES (?, ?, ?, ?, datetime('now'), ?, ?, ?)
                """,
                (
                    run_id,
                    seed,
                    "RUNNING",
                    "SETUP",
                    json.dumps(config),
                    json.dumps(hardware),
                    "SEALED",
                ),
            )
            conn.commit()

    def update_stage(self, run_id: str, stage: str, status: str = "RUNNING"):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE runs SET stage = ?, status = ? WHERE run_id = ?",
                (stage, status, run_id),
            )
            conn.commit()

    def log_event(self, run_id: str, event_type: str, stage: str, message: str, data: Optional[Dict[str, Any]] = None):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (run_id, timestamp, event_type, stage, message, data_json)
                VALUES (?, datetime('now'), ?, ?, ?, ?)
                """,
                (run_id, event_type, stage, message, json.dumps(data) if data else None),
            )
            conn.commit()

    def log_epoch(
        self,
        run_id: str,
        phase: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_f1: float,
        val_f1: float,
        train_balanced_acc: float,
        val_balanced_acc: float,
        lr: float,
        images_per_sec: float,
        duration_sec: float,
        metrics_dict: Optional[Dict[str, Any]] = None,
    ):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO epochs (
                    run_id, phase, epoch, train_loss, val_loss, train_f1, val_f1,
                    train_balanced_acc, val_balanced_acc, learning_rate,
                    images_per_sec, epoch_duration_sec, timestamp, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
                """,
                (
                    run_id,
                    phase,
                    epoch,
                    train_loss,
                    val_loss,
                    train_f1,
                    val_f1,
                    train_balanced_acc,
                    val_balanced_acc,
                    lr,
                    images_per_sec,
                    duration_sec,
                    json.dumps(metrics_dict) if metrics_dict else None,
                ),
            )
            conn.commit()

    def log_resource_sample(
        self,
        run_id: str,
        stage: str,
        cpu_percent: float,
        ram_percent: float,
        ram_rss_mb: float,
        gpu_util_percent: float,
        gpu_mem_used_mb: float,
        disk_used_gb: float,
        images_per_sec: float,
    ):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO resource_samples (
                    run_id, timestamp, cpu_percent, ram_percent, ram_rss_mb,
                    gpu_util_percent, gpu_mem_used_mb, disk_used_gb, images_per_sec, stage
                ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    cpu_percent,
                    ram_percent,
                    ram_rss_mb,
                    gpu_util_percent,
                    gpu_mem_used_mb,
                    disk_used_gb,
                    images_per_sec,
                    stage,
                ),
            )
            conn.commit()

    def log_predictions(self, run_id: str, split: str, predictions_list: List[Dict[str, Any]]):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            records = []
            for p in predictions_list:
                records.append(
                    (
                        run_id,
                        split,
                        p["image_id"],
                        p.get("lesion_id", ""),
                        p["true_class"],
                        p["predicted_class"],
                        p.get("probability_true", 0.0),
                        p.get("probability_predicted", 0.0),
                        p.get("confidence", 0.0),
                        p.get("entropy", 0.0),
                        1 if p["correct"] else 0,
                        json.dumps(p.get("probabilities", {})),
                    )
                )
            cursor.executemany(
                """
                INSERT INTO predictions (
                    run_id, split, image_id, lesion_id, true_class, predicted_class,
                    probability_true, probability_predicted, confidence, entropy, correct,
                    probabilities_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def log_artifact(
        self,
        run_id: str,
        artifact_type: str,
        file_path: str,
        file_size_bytes: int,
        sha256: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO artifacts (
                    run_id, artifact_type, file_path, file_size_bytes, sha256, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, datetime('now'), ?)
                """,
                (run_id, artifact_type, file_path, file_size_bytes, sha256, json.dumps(metadata) if metadata else None),
            )
            conn.commit()

    def finish_run(
        self,
        run_id: str,
        status: str,
        macro_f1: Optional[float] = None,
        balanced_accuracy: Optional[float] = None,
        accuracy: Optional[float] = None,
        cpu_p50_ms: Optional[float] = None,
        cpu_p95_ms: Optional[float] = None,
        fp32_size_mb: Optional[float] = None,
        int8_size_mb: Optional[float] = None,
        holdout_status: str = "CONSUMED",
        error_message: Optional[str] = None,
    ):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE runs SET
                    status = ?,
                    stage = ?,
                    end_time = datetime('now'),
                    macro_f1 = COALESCE(?, macro_f1),
                    balanced_accuracy = COALESCE(?, balanced_accuracy),
                    accuracy = COALESCE(?, accuracy),
                    cpu_p50_ms = COALESCE(?, cpu_p50_ms),
                    cpu_p95_ms = COALESCE(?, cpu_p95_ms),
                    fp32_size_mb = COALESCE(?, fp32_size_mb),
                    int8_size_mb = COALESCE(?, int8_size_mb),
                    holdout_status = ?,
                    error_message = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    "COMPLETE" if status == "COMPLETE" else "FAILED",
                    macro_f1,
                    balanced_accuracy,
                    accuracy,
                    cpu_p50_ms,
                    cpu_p95_ms,
                    fp32_size_mb,
                    int8_size_mb,
                    holdout_status,
                    error_message,
                    run_id,
                ),
            )
            conn.commit()

    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs ORDER BY start_time DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_epochs(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM epochs WHERE run_id = ? ORDER BY id ASC", (run_id,))
            return [dict(r) for r in cursor.fetchall()]

    def get_resource_samples(self, run_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM resource_samples WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                (run_id, limit),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in reversed(rows)]

    def get_predictions(self, run_id: str, split: str = "holdout") -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM predictions WHERE run_id = ? AND split = ? ORDER BY id ASC",
                (run_id, split),
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_events(self, run_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                (run_id, limit),
            )
            return [dict(r) for r in cursor.fetchall()]
