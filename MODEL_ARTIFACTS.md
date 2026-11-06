# TinyDerm model artifact catalog

Updated 2026-11-05. This is an inventory and continuation note; **no model binaries are uploaded to this repository**. All artifacts are research artifacts, not validated for clinical use. The repository README and [training continuation handoff](TRAINING_CONTINUATION.md) contain the top five candidates and their development metrics.

## Retained trained artifacts

A local staging inventory counted 417 trained artifacts across five experiment groups:

| Experiment group | Files | Formats | What the group contains |
|---|---:|---|---|
| `training-review` | 281 | 260 `.pt`, 21 `.joblib` | Reviewed training checkpoints and fitted auxiliary classifiers |
| `campaign-runs` | 110 | `.pt` | Three-fold checkpoints from the supervised campaign runs |
| `original-runs` | 18 | 12 `.pt`, 6 `.onnx` | Earlier checkpoints and exported inference graphs |
| `melanoma-cv-448` | 2 | `.pt` | Baseline and Hunch-assisted fold checkpoints |
| `melanoma-cv-448-strict` | 6 | `.pt` | Strict melanoma cross-validation fold checkpoints |
| **Total** | **417** | | |

The staged binary bundle was organized into 15 source-separated TAR parts totaling 14,591,463,424 bytes. The source file manifest and archive checksums were generated for that local bundle but are not hosted here. Obtain the artifacts from the project owner if a specific checkpoint is needed, then verify its provenance and checksum before use. Do not infer model quality from the presence of a checkpoint: use the reported evaluation and promotion criteria.

## Training stack and continuation

- Student experiments use DINOv2-S/14, with image-only baselines and Hunch-assisted variants.
- Hunch `Eyes` and Google's EmbeddingGemma 2 provide teacher-side embeddings/classifiers for experiments that use teacher assistance. They are upstream tools/models; their weights and caches are not part of this repository.
- The repository contains the training/evaluation source, dependency manifest, controller, safe data adapter, and top-five aggregate reports. See [TRAINING_CONTINUATION.md](TRAINING_CONTINUATION.md) for pinned upstream revisions, environment notes, data setup, commands, and safeguards.
- The latest campaign is capped and paused. None of the five highlighted candidates met the promotion rule. A new run requires a newly authorized campaign budget and decision capacity.
- Keep image data, image and embedding caches, predictions, and all checkpoint binaries outside GitHub. Never use the consumed final holdout for training, selection, or further evaluation.

## Scope

The inventory covers trained TinyDerm artifacts from the named experiment groups only. It does not include upstream foundation-model snapshots, raw datasets, images, embedding caches, prediction arrays, or the consumed final holdout. The reported metrics are exploratory development cross-validation results and do not establish clinical validity.
