# TinyDerm Lab
## Autonomous Local Medical-Image Compression Research Prototype

> **RESEARCH-ONLY PROTOTYPE**  
> This system must not be presented or used as a clinical diagnostic product or as a substitute for qualified medical judgment.

### Research Objective
Measure how much useful medical-image classification capability can be retained in a very small, fast, CPU-friendly model that runs entirely on an ordinary computer.

---

### Key Capabilities
- **Official Dataset Acquisition**: ISIC 2018 Task 3 / HAM10000 collection (10,015 images, 7,470 lesion IDs) with full provenance tracking.
- **Leakage-Safe Lesion Grouping**: 90% Development / 10% Sealed Final Holdout split strictly grouped by `lesion_id` to eliminate data leakage.
- **Tiny Student Architecture**: Pretrained `mobilenetv3_small_100` (~1.52M parameters) with two-stage training (head freezing + complete fine-tuning).
- **Holdout Governance**: Programmatically locked 10% benchmark evaluated exactly once, generating full confusion matrix and confidence distribution.
- **Deployable CPU Artifacts**: ONNX FP32 and INT8 export with single-threaded and multi-threaded ONNX Runtime benchmarking.
- **Deterministic Improvement Analysis**: Evidence-based diagnostics (`improvement_report.md`) identifying rare classes, confusion pairs, and calibration gaps.
- **Autonomous Local Dashboard**: Streamlit portal running at `http://127.0.0.1:8501` for real-time monitoring and error exploration.

---

### Quick Start

#### 1. Launch via Windows Batch / PowerShell:
```cmd
start_tinyderm.bat
```
Or in PowerShell:
```powershell
.\start_tinyderm.ps1
```

#### 2. Launch via Python directly:
```bash
python run.py
```

The runner initializes the experiment database, spins up the Streamlit portal, opens your default browser at `http://127.0.0.1:8501`, and executes the pipeline from start to finish.
