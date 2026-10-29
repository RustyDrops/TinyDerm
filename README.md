# TinyDerm Lab
## Autonomous Local Medical-Image Compression Research Prototype

> **RESEARCH-ONLY PROTOTYPE**  
> This system is an experimental AI research prototype. It must not be presented or used as a clinical diagnostic product or as a substitute for qualified medical judgment. Always consult a licensed medical doctor or dermatologist for professional evaluation of any skin condition.

### Research Objective
Measure how much useful medical-image classification capability can be retained in an ultra-compact, fast, CPU-friendly model (~1.5M parameters, 1.62 MB footprint) that runs entirely on ordinary commodity computers or directly in a client web browser without cloud dependencies.

---

### Conceptual Inspiration
TinyDerm Lab is directly inspired by **[Matt Mastracci](https://x.com/mmastrac)**'s work on **[Hunch](https://github.com/undrgrnd-labs/hunch)** and his **[recent post on X](https://x.com/mmastrac/status/2115505657231384683)**:
> *"Meet hunch! Jev-like decisions in ~20ms without even needing a GPU, learning live from a teacher model. Built on @GoogleGemma's EmbeddingGemma 2, it can work with any teacher model like @Cloudflare's Clef. Multimodal, w/ a systemone endpoint, slots into your existing stack 🧵"*

TinyDerm explores how that same core philosophy—learning lightweight, repeated decisions from a large teacher model without requiring a cloud GPU—can be applied to medical dermatology imaging, compressing specialist intelligence into a sub-millisecond, CPU-friendly edge student.

---

### Empirical Research Results (Sealed Holdout Benchmark)

Tested on the sealed 10% holdout (1,006 unseen dermoscopic images across 747 distinct lesions) using MobileNetV3-Small:

| Metric | Baseline Run | Improved Run (Focal Loss + TTA) |
| :--- | :--- | :--- |
| **Headline Macro F1** | 0.6503 (65.03%) | **0.6930 (69.30%)** |
| **Balanced Accuracy** | 0.6835 (68.35%) | **0.7165 (71.65%)** |
| **Ordinary Accuracy** | 0.8121 (81.21%) | **0.8131 (81.31%)** |
| **High-Confidence Errors ($\ge 80\%$)** | 78 cases (7.75%) | **16 cases (1.59%)** *(~80% reduction in dangerous errors)* |
| **Calibration Error (ECE)** | 0.0844 | **0.0824** |
| **Model Size (INT8 Quantized)** | **1.62 MB** | **1.62 MB** |
| **CPU Latency (p50)** | **0.79 ms** (1,255 img/s) | **0.79 ms** |

---

### Key Capabilities
- **Official Dataset Acquisition**: ISIC 2018 Task 3 / HAM10000 collection (10,015 images, 7,470 lesion IDs) with full provenance tracking.
- **Leakage-Safe Lesion Grouping**: 90% Development / 10% Sealed Final Holdout split strictly grouped by `lesion_id` to eliminate data leakage.
- **Tiny Student Architecture**: Pretrained `mobilenetv3_small_100` (~1.52M parameters) with two-stage training (head freezing + complete fine-tuning).
- **Holdout Governance**: Programmatically locked 10% benchmark evaluated exactly once, generating full confusion matrix and confidence distribution.
- **Deployable CPU Artifacts**: ONNX FP32 and INT8 export with single-threaded and multi-threaded ONNX Runtime benchmarking.
- **Deterministic Improvement Analysis**: Evidence-based diagnostics (`improvement_report.md`) identifying rare classes, confusion pairs, and calibration gaps.
- **Autonomous Local Dashboard**: Streamlit portal running at `http://127.0.0.1:8501` for real-time monitoring and error exploration.
- **Real-Time Client-Side Web App**: 100% in-browser camera scanning via `onnxruntime-web` with audio feedback and zero server uploads.

---

### Quick Start

#### 1. Launch Training Pipeline & Monitoring Dashboard:
```cmd
start_tinyderm.bat
```
Or in PowerShell:
```powershell
.\start_tinyderm.ps1
```
Or directly with Python:
```bash
python run.py
```
Opens the Streamlit monitoring portal at `http://127.0.0.1:8501`.

#### 2. Launch Real-Time Camera Scanner Web App:
```cmd
start_camera_app.bat
```
Opens the live camera scanner at `http://127.0.0.1:8000`.

#### 3. Client-Side Static Web Deployment:
The bundle in `TinyDerm_Client_Web_App.zip` contains `index.html` and `model.onnx`, running 100% inside client browsers via WebAssembly. Unzip directly to any static web hosting (GitHub Pages, Netlify, Cloudflare Pages, Apache, or Nginx).

---

### Open-Source Foundations and Acknowledgments

TinyDerm Lab is a standalone research project. It is not a fork of, affiliated with, or endorsed by the projects below. We gratefully credit the datasets, frameworks, model families, and research tools that made this work possible or inspired planned phases of the research:

- **ISIC Archive & HAM10000 Dataset:**  
  The initial research dataset and diagnostic ground truth. Provided by the International Skin Imaging Collaboration (ISIC).  
  *Citation:* Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018). [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161).  
  *Challenge:* Codella, N. et al. *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)*. [arXiv:1902.03368](https://arxiv.org/abs/1902.03368). [https://challenge.isic-archive.com/data/](https://challenge.isic-archive.com/data/) (CC-BY-NC).

- **PyTorch:**  
  Core deep-learning framework used for model training, dataset handling, and evaluation.  
  [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) (BSD-3-Clause).

- **PyTorch Image Models (`timm`):**  
  Provides the MobileNetV3-Small architecture, pretrained ImageNet weights, and preprocessing conventions. Authored by Ross Wightman.  
  [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (Apache-2.0).

- **ONNX & ONNX Runtime:**  
  Used to export, quantize, and benchmark the deployable CPU and browser models, enabling sub-millisecond local inference and WebAssembly client execution.  
  [https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) (MIT License).

- **Hunch (undrgrnd-labs) by Matt Mastracci:**  
  Primary conceptual inspiration for distilling repeated specialist decisions into lightweight edge classifiers without needing a GPU. Inspired by [Matt Mastracci on X](https://x.com/mmastrac/status/2115505657231384683) ([@mmastrac](https://x.com/mmastrac)).  
  [https://github.com/undrgrnd-labs/hunch](https://github.com/undrgrnd-labs/hunch) (Apache-2.0).

- **EmbeddingGemma 2 (Google DeepMind):**  
  Planned multimodal embedding backbone for future comparative experiments testing compact classifiers on general-purpose embeddings.  
  [https://huggingface.co/google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (Apache-2.0).

- **Derm Foundation (Google):**  
  Planned specialist dermatology foundation model for future Phase 2 teacher distillation experiments.  
  [https://huggingface.co/google/derm-foundation](https://huggingface.co/google/derm-foundation).

- **Scientific Python & Web Ecosystem:**  
  - **Streamlit:** Interactive local experiment monitoring and error explorer ([https://streamlit.io](https://streamlit.io)).
  - **scikit-learn:** Comprehensive multi-class evaluation metrics, confusion matrices, and calibration analysis ([https://scikit-learn.org](https://scikit-learn.org)).
  - **Albumentations & Torchvision:** Domain-specific data augmentation preserving medical color fidelity.
  - **FastAPI & Uvicorn:** Lightweight high-performance camera server backend.
  - **Plotly, NumPy, pandas, Pillow, PyYAML, psutil:** Data handling, telemetry, and visualization.

*All third-party names, trademarks, and registered trademarks are the property of their respective owners.*
