"""Streamlit Autonomous Monitoring Dashboard for TinyDerm Lab."""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from tinyderm.experiment_db import ExperimentDB

st.set_page_config(page_title="TinyDerm Lab Portal", layout="wide", page_icon="🔬")

# Database initialization
DB_PATH = Path("runs/experiments.sqlite")
db = ExperimentDB(DB_PATH)

OFFICIAL_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# Header
st.title("🔬 TinyDerm Lab — Autonomous Compression Portal")
st.caption("Local Medical-Image Compression Research Prototype | ISIC 2018 / HAM10000")

# Fetch latest run
latest_run = db.get_latest_run()

if not latest_run:
    st.info("Pipeline initializing... Waiting for first experiment run to register.")
    st.stop()

run_id = latest_run["run_id"]
stage = latest_run.get("stage", "UNKNOWN")
status = latest_run.get("status", "UNKNOWN")
holdout_status = latest_run.get("holdout_status", "SEALED")

# Top Status Banner
col_stage, col_holdout, col_seed, col_accel = st.columns(4)
with col_stage:
    stage_color = "🟢" if stage == "COMPLETE" else ("🔴" if stage == "FAILED" else "⏳")
    st.metric("Pipeline Stage", f"{stage_color} {stage}", delta=status)

with col_holdout:
    h_color = "🚨 CONSUMED" if holdout_status == "CONSUMED" else "🔒 SEALED"
    st.metric("Holdout Status", h_color)

with col_seed:
    st.metric("Random Seed", latest_run.get("seed", 42))

with col_accel:
    hw_info = json.loads(latest_run.get("hardware_json") or "{}")
    accel = hw_info.get("accelerator", "CPU").upper()
    gpu_name = hw_info.get("accelerator_details", {}).get("name", "N/A")
    st.metric("Accelerator", accel, help=gpu_name)

# Pipeline progress steps indicator
STAGES = [
    "SETUP", "DOWNLOADING", "VERIFYING", "PREPARING", "SPLITTING",
    "TRAINING_HEAD", "FINETUNING", "SELECTING_MODEL",
    "RETRAINING_90_PERCENT", "EXPORTING", "QUANTIZING",
    "FINAL_HOLDOUT", "BENCHMARKING", "COMPLETE"
]
current_idx = STAGES.index(stage) if stage in STAGES else (len(STAGES) - 1 if stage == "COMPLETE" else 0)
progress_pct = (current_idx + 1) / len(STAGES)
st.progress(progress_pct, text=f"Progress: Stage {current_idx + 1}/{len(STAGES)} — {stage}")

# Result Card if Complete
if stage == "COMPLETE":
    st.success("### 🏆 TinyDerm Research Prototype — COMPLETE")
    card_col1, card_col2, card_col3, card_col4, card_col5 = st.columns(5)
    with card_col1:
        st.metric("Headline Macro F1", f"{latest_run.get('macro_f1', 0.0):.4f}")
    with card_col2:
        st.metric("Balanced Accuracy", f"{latest_run.get('balanced_accuracy', 0.0):.4f}")
    with card_col3:
        st.metric("Ordinary Accuracy", f"{latest_run.get('accuracy', 0.0):.4f}")
    with card_col4:
        st.metric("CPU p50 Latency", f"{latest_run.get('cpu_p50_ms', 0.0):.2f} ms")
    with card_col5:
        st.metric("Deployable Size", f"{latest_run.get('int8_size_mb') or latest_run.get('fp32_size_mb', 0.0):.2f} MB")

# Navigation Tabs
tab_overview, tab_training, tab_holdout, tab_errors, tab_reports = st.tabs([
    "📊 Overview & System",
    "📈 Live Training Dynamics",
    "🎯 Final Holdout Evaluation",
    "🔍 Error Explorer",
    "📑 Improvement Report & Artifacts",
])

# -------------------------------------------------------------
# TAB 1: Overview & System
# -------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Dataset Provenance & Distribution")
        prov_file = Path("data/provenance.json")
        if prov_file.exists():
            with open(prov_file, "r") as f:
                prov = json.load(f)
            st.json(prov)
        else:
            st.write("Provenance data registering...")

    with c2:
        st.subheader("Host Hardware & Environment")
        st.json(hw_info)

    # Live resource usage charts
    st.subheader("System Resource Telemetry")
    res_samples = db.get_resource_samples(run_id, limit=100)
    if res_samples:
        res_df = pd.DataFrame(res_samples)
        rc1, rc2 = st.columns(2)
        with rc1:
            fig_cpu = px.line(res_df, y=["cpu_percent", "ram_percent"], title="CPU % and RAM % Utilization")
            st.plotly_chart(fig_cpu, use_container_width=True)
        with rc2:
            fig_rss = px.line(res_df, y="ram_rss_mb", title="Process RSS Memory Footprint (MB)")
            st.plotly_chart(fig_rss, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Live Training Dynamics
# -------------------------------------------------------------
with tab_training:
    epochs = db.get_epochs(run_id)
    if epochs:
        ep_df = pd.DataFrame(epochs)
        c_loss, c_metrics = st.columns(2)

        with c_loss:
            fig_loss = px.line(
                ep_df,
                x="epoch",
                y=["train_loss", "val_loss"],
                color="phase",
                title="Training vs Validation Loss",
                labels={"value": "Loss", "epoch": "Epoch"},
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with c_metrics:
            fig_f1 = px.line(
                ep_df,
                x="epoch",
                y=["val_f1", "val_balanced_acc"],
                title="Validation Macro F1 & Balanced Accuracy",
                labels={"value": "Score", "epoch": "Epoch"},
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        st.dataframe(
            ep_df[["phase", "epoch", "train_loss", "val_loss", "val_f1", "val_balanced_acc", "learning_rate", "images_per_sec", "epoch_duration_sec"]],
            use_container_width=True,
        )
    else:
        st.info("Training epochs will appear here as batches finish.")

# -------------------------------------------------------------
# TAB 3: Final Holdout Evaluation
# -------------------------------------------------------------
with tab_holdout:
    run_dir = Path("runs") / run_id
    metrics_file = run_dir / "final_metrics.json"

    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            holdout_metrics = json.load(f)

        hm1, hm2, hm3, hm4 = st.columns(4)
        with hm1:
            st.metric("Macro F1", f"{holdout_metrics.get('macro_f1', 0.0):.4f}")
        with hm2:
            st.metric("Balanced Accuracy", f"{holdout_metrics.get('balanced_accuracy', 0.0):.4f}")
        with hm3:
            st.metric("Macro Precision", f"{holdout_metrics.get('macro_precision', 0.0):.4f}")
        with hm4:
            st.metric("Macro Recall", f"{holdout_metrics.get('macro_recall', 0.0):.4f}")

        # Confusion Matrix Heatmap
        cm = holdout_metrics.get("confusion_matrix", [])
        if cm:
            classes = holdout_metrics.get("classes", OFFICIAL_CLASSES)
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted Diagnosis", y="True Diagnosis", color="Cases"),
                x=classes,
                y=classes,
                text_auto=True,
                color_continuous_scale="Viridis",
                title="Holdout Confusion Matrix (Counts)",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Per-Class Sensitivity and Specificity
        per_class = holdout_metrics.get("per_class", {})
        if per_class:
            pc_df = pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "Class"})
            st.subheader("Per-Class Sensitivity & Specificity")
            st.dataframe(pc_df, use_container_width=True)
    else:
        st.info("Final holdout evaluation metrics will unlock once model retraining and export finish.")

# -------------------------------------------------------------
# TAB 4: Error Explorer
# -------------------------------------------------------------
with tab_errors:
    preds = db.get_predictions(run_id, split="holdout")
    if preds:
        pred_df = pd.DataFrame(preds)

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            filter_mode = st.selectbox("Filter", ["All", "Errors Only", "Highest-Confidence Errors", "Lowest-Confidence Correct"])
        with f_col2:
            selected_true = st.selectbox("True Class", ["All"] + OFFICIAL_CLASSES)
        with f_col3:
            selected_pred = st.selectbox("Predicted Class", ["All"] + OFFICIAL_CLASSES)

        filtered = pred_df.copy()
        if filter_mode == "Errors Only":
            filtered = filtered[filtered["correct"] == 0]
        elif filter_mode == "Highest-Confidence Errors":
            filtered = filtered[filtered["correct"] == 0].sort_values("confidence", ascending=False)
        elif filter_mode == "Lowest-Confidence Correct":
            filtered = filtered[filtered["correct"] == 1].sort_values("confidence", ascending=True)

        if selected_true != "All":
            filtered = filtered[filtered["true_class"] == selected_true]
        if selected_pred != "All":
            filtered = filtered[filtered["predicted_class"] == selected_pred]

        st.write(f"Displaying {len(filtered)} samples:")
        st.dataframe(
            filtered[["image_id", "lesion_id", "true_class", "predicted_class", "confidence", "entropy", "correct"]],
            use_container_width=True,
        )

        # Thumbnail Inspector for top sample
        if not filtered.empty:
            st.subheader("Sample Visual Inspector")
            sample_row = filtered.iloc[0]
            s_col1, s_col2 = st.columns([1, 2])
            with s_col1:
                img_path = Path("data/raw/isic2018/images") / f"{sample_row['image_id']}.jpg"
                if img_path.exists():
                    st.image(str(img_path), caption=f"{sample_row['image_id']} (Lesion: {sample_row['lesion_id']})")
                else:
                    st.write(f"Image {sample_row['image_id']} preview")
            with s_col2:
                st.write(f"**True Class:** `{sample_row['true_class']}`")
                st.write(f"**Predicted Class:** `{sample_row['predicted_class']}`")
                st.write(f"**Confidence:** `{sample_row['confidence']:.2%}`")
                st.write(f"**Entropy:** `{sample_row['entropy']:.4f}`")
                prob_dict = json.loads(sample_row["probabilities_json"] or "{}")
                if prob_dict:
                    st.bar_chart(pd.Series(prob_dict))
    else:
        st.info("Holdout prediction explorer will activate upon holdout evaluation.")

# -------------------------------------------------------------
# TAB 5: Improvement Report & Artifacts
# -------------------------------------------------------------
with tab_reports:
    report_file = Path("runs") / run_id / "improvement_report.md"
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Deterministic improvement analysis will generate upon pipeline completion.")

    st.subheader("Exported Model Artifacts")
    models_dir = Path("runs") / run_id / "models"
    if models_dir.exists():
        for mf in models_dir.glob("*"):
            st.write(f"- `{mf.name}` ({round(mf.stat().st_size / (1024*1024), 2)} MB)")
