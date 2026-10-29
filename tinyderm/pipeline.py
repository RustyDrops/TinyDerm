"""Autonomous Pipeline Orchestrator for TinyDerm Lab."""

import datetime
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import pandas as pd
import torch
import yaml

from tinyderm.acquisition import acquire_dataset
from tinyderm.analysis import generate_improvement_report
from tinyderm.benchmark import run_full_cpu_benchmark
from tinyderm.evaluate import run_sealed_holdout_evaluation
from tinyderm.experiment_db import ExperimentDB
from tinyderm.export import export_model_artifacts
from tinyderm.provenance import compute_file_hash
from tinyderm.quantize import quantize_onnx_model
from tinyderm.resources import ResourceMonitor, detect_hardware
from tinyderm.split import build_and_lock_splits
from tinyderm.train import get_device, train_student_model

logger = logging.getLogger(__name__)


class TinyDermPipeline:
    """Executes the complete end-to-end research workflow."""

    def __init__(
        self,
        config_path: str | Path = "config/default.yaml",
        db_path: str | Path = "runs/experiments.sqlite",
        run_id: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.db = ExperimentDB(db_path)
        self.run_id = run_id or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.hardware = detect_hardware()
        self.resource_monitor = ResourceMonitor(self.db, self.run_id)
        self.current_stage = "SETUP"

    def set_stage(self, stage: str, status: str = "RUNNING"):
        self.current_stage = stage
        self.resource_monitor.set_stage(stage)
        self.db.update_stage(self.run_id, stage=stage, status=status)
        logger.info(f"=== PIPELINE STAGE: {stage} ({status}) ===")

    def run(self):
        """Execute full autonomous pipeline."""
        output_dir = Path("runs") / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. SETUP
            self.db.create_run(self.run_id, self.config.get("seed", 42), self.config, self.hardware)
            self.set_stage("SETUP")
            self.resource_monitor.start()

            # Save initial run environment info
            with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(self.config, f)
            with open(output_dir / "system.json", "w", encoding="utf-8") as f:
                json.dump(self.hardware, f, indent=2)
            with open(output_dir / "environment.txt", "w", encoding="utf-8") as f:
                f.write(f"Python: {sys.version}\nPyTorch: {torch.__version__}\nCUDA: {torch.cuda.is_available()}\n")

            # 2. DOWNLOADING & PREPARING
            self.set_stage("DOWNLOADING")
            raw_dir = Path(self.config["dataset"]["raw_dir"])
            metadata_df, provenance = acquire_dataset(
                raw_dir=raw_dir,
                collection_id=self.config["dataset"]["collection_id"],
                progress_callback=lambda p, msg: self.db.log_event(self.run_id, "DOWNLOAD_PROGRESS", "DOWNLOADING", msg),
            )

            # 3. VERIFYING
            self.set_stage("VERIFYING")
            expected_count = self.config["dataset"].get("expected_images", 10015)
            actual_count = len(metadata_df)
            images_dir = raw_dir / "images"
            verified_images = len(list(images_dir.glob("*.jpg")))
            logger.info(f"Verified {verified_images} image files and {actual_count} metadata rows.")

            # 4. SPLITTING & LOCKING
            self.set_stage("SPLITTING")
            splits_dir = Path(self.config["dataset"]["splits_dir"])
            manifest_df, lock_data = build_and_lock_splits(
                metadata_df,
                images_dir=images_dir,
                splits_dir=splits_dir,
                seed=self.config.get("seed", 42),
            )

            with open(output_dir / "split_hash.txt", "w", encoding="utf-8") as f:
                f.write(lock_data.get("split_hash", "") + "\n")

            # 5. TRAINING_HEAD & FINETUNING (Development Validation Recipe)
            self.set_stage("TRAINING_HEAD")
            device = get_device()
            model, dev_metrics = train_student_model(
                manifest_df=manifest_df,
                config=self.config,
                db=self.db,
                run_id=self.run_id,
                output_dir="runs",
                retrain_on_dev=False,
                progress_callback=lambda stage, ep, total, m: self.set_stage(stage),
                throughput_callback=lambda tp: self.resource_monitor.set_throughput(tp),
            )

            # 6. SELECTING_MODEL
            self.set_stage("SELECTING_MODEL")
            logger.info(f"Model selection complete. Validation Macro F1: {dev_metrics.get('macro_f1', 0.0):.4f}")

            # 7. RETRAINING_90_PERCENT
            self.set_stage("RETRAINING_90_PERCENT")
            final_model, retrain_metrics = train_student_model(
                manifest_df=manifest_df,
                config=self.config,
                db=self.db,
                run_id=self.run_id,
                output_dir="runs",
                retrain_on_dev=True,
                progress_callback=lambda stage, ep, total, m: self.set_stage(stage),
                throughput_callback=lambda tp: self.resource_monitor.set_throughput(tp),
            )

            # 8. EXPORTING
            self.set_stage("EXPORTING")
            export_artifacts = export_model_artifacts(final_model, output_dir=output_dir)

            # 9. QUANTIZING
            self.set_stage("QUANTIZING")
            onnx_fp32_path = export_artifacts["onnx_fp32"]["path"]
            int8_artifact = quantize_onnx_model(onnx_fp32_path, output_dir=output_dir)

            # 10. FINAL_HOLDOUT (Sealed 10% benchmark evaluated exactly once)
            self.set_stage("FINAL_HOLDOUT")
            holdout_metrics, holdout_preds = run_sealed_holdout_evaluation(
                model=final_model,
                manifest_df=manifest_df,
                splits_dir=splits_dir,
                run_id=self.run_id,
                output_dir="runs",
                db=self.db,
                device=device,
            )

            # 11. BENCHMARKING (CPU ONNX Runtime Benchmark)
            self.set_stage("BENCHMARKING")
            int8_path = int8_artifact["path"] if int8_artifact else None
            benchmark_results = run_full_cpu_benchmark(
                fp32_model_path=onnx_fp32_path,
                int8_model_path=int8_path,
                num_passes=self.config.get("benchmarking", {}).get("num_passes", 200),
            )

            with open(output_dir / "resource_summary.json", "w", encoding="utf-8") as f:
                json.dump(benchmark_results, f, indent=2)

            # 12. GENERATE IMPROVEMENT REPORT
            epochs_history = self.db.get_epochs(self.run_id)
            generate_improvement_report(
                metrics=holdout_metrics,
                predictions=holdout_preds,
                epochs_history=epochs_history,
                benchmark_results=benchmark_results,
                output_dir=output_dir,
            )

            # 13. COMPLETE
            fp32_bench = benchmark_results.get("fp32_singlethread", {})
            self.db.finish_run(
                run_id=self.run_id,
                status="COMPLETE",
                macro_f1=holdout_metrics.get("macro_f1"),
                balanced_accuracy=holdout_metrics.get("balanced_accuracy"),
                accuracy=holdout_metrics.get("accuracy"),
                cpu_p50_ms=fp32_bench.get("p50_latency_ms"),
                cpu_p95_ms=fp32_bench.get("p95_latency_ms"),
                fp32_size_mb=export_artifacts.get("onnx_fp32", {}).get("size_mb"),
                int8_size_mb=int8_artifact.get("size_mb") if int8_artifact else None,
                holdout_status="CONSUMED",
            )
            self.set_stage("COMPLETE", status="COMPLETE")
            logger.info("TinyDerm Pipeline completed successfully!")

        except Exception as e:
            err_msg = traceback.format_exc()
            logger.error(f"Pipeline error: {err_msg}")
            self.db.log_event(self.run_id, "ERROR", self.current_stage, str(e), {"traceback": err_msg})
            self.db.finish_run(self.run_id, status="FAILED", holdout_status="SEALED", error_message=err_msg)
            self.set_stage("FAILED", status="FAILED")
        finally:
            self.resource_monitor.stop()
