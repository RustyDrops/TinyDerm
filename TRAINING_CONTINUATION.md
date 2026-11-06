# TinyDerm training continuation handoff

Updated 2026-11-05. Research code and scores are provided for continuation of a supervised, local research workflow. TinyDerm is not a diagnostic system.

## Current status and best five candidates

The latest 18-run/25-decision campaign is paused at its enforced caps. The promotion rule was macro-F1 >= 0.73851 and no class recall regression greater than 0.05 absolute versus H001. None of the runs met that rule. Ranked by selected-arm development macro-F1:

| Rank | Run / selected arm | Seed / epochs | Macro-F1 | Accuracy | Recall safeguard | Interpretation |
|---:|---|---:|---:|---:|---|---|
| 1 | H028 / Hunch | 52711 / 5 | 0.73096 | 0.81774 | DF -0.06796 | Highest raw macro-F1; fails DF safeguard and target |
| 2 | H032 / image-only baseline | 52711 / 4 | 0.72909 | 0.82773 | AKIEC -0.06102 | Fails target and AKIEC safeguard |
| 3 | H001 / Hunch | 52711 / 3 | 0.72851 | 0.81885 | Reference | Campaign reference; best policy-eligible result |
| 4 | H031 / image-only baseline | 52711 / 3 | 0.72824 | 0.82107 | VASC -0.01562 | Fails macro-F1 target |
| 5 | H033 / image-only baseline | 52716 / 3 | 0.71494 | 0.80941 | BCC -0.03913 | Different seed/folds; descriptive seed sensitivity |

These are exploratory, repeatedly selected grouped development-CV estimates on 9,009 ISIC 2018 development images. H033's fold assignment differs. H028's reported epoch-four intermediate was under a five-epoch schedule. See `training/results/top-five/campaign_synthesis_2026-11-05.md` and the five JSON run reports for source detail. H001 is the reference despite H028's higher raw score because safeguards govern promotion.

The five model checkpoints are retained in the source workstation's campaign run folders, not this repository. Each run uses three fold checkpoints per arm (~88 MB each); GitHub is not an appropriate host for these multi-gigabyte fold bundles. Obtain authorized checkpoint artifacts separately if they are needed. This source package contains no checkpoint binary.

## Training stack

- **Student:** DINOv2-S/14, frozen feature or partial fine-tuning, seven diagnostic classes (MEL, NV, BCC, AKIEC, BKL, DF, VASC); an image-only matched baseline is retained.
- **Teacher:** Hunch `Eyes` image encoder with Google's **EmbeddingGemma 2**, local image embeddings, lesion-grouped out-of-fold teacher targets, and class-balanced logistic heads. Student receives pixels; EmbeddingGemma provides teacher features. Hunch assistance has not shown a consistent promotion-level gain.
- **Training/evaluation:** PyTorch, torchvision/timm, scikit-learn, NumPy, pandas, Pillow, grouped CV, safe data adapter, preflight/controller, OOF calibration and audit utilities. Main scripts are under `training/`.
- **Recorded environment:** Python 3.12.5, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3060. This is the tested local environment, not a universal pin set. Resolve and validate compatible dependencies on the target machine.
- **Pinned research sources:** Hunch commit `6541bd157e11a5e7d7434360e99ea81f40a8c181`; EmbeddingGemma 2 checkpoint revision `914f7f89142e33e77833254d9c9b90c3cef7303b`. Local cache paths are intentionally omitted. Download model assets from the official upstream model repository under its license and set `EG2_MODEL_DIR` to the local snapshot before running Hunch tools.

## Data and setup

1. Set up Python 3.12 with a compatible PyTorch/CUDA build and install the declared dependencies. The project dependency manifest is `training/pyproject.toml`; the Hunch encoder's upstream source and native `eg2cpu` extension must also be installed at the pinned revision.
2. Obtain ISIC 2018 / HAM10000 through the official dataset source. ISIC data terms include noncommercial restrictions; follow the dataset's license and citation. Keep raw image archives and decoded image caches local.
3. Provide the original manifest and image root to the safe-data adapter. The current campaign runner includes workstation-specific default paths; edit its `PYTHON`, `MANIFEST`, and `HUNCH_CACHE` constants for your machine, then review the allowlist and data gates before execution.
4. Set `EG2_MODEL_DIR` to the local EmbeddingGemma 2 snapshot. Do not commit the model snapshot, Hunch embedding database, or cache.
5. Read `training/research/campaign_synthesis_2026-11-05.md`, `training/research/h001_h028_h031_error_audit.md`, and this handoff. Check controller status and preflight before any new run. The existing campaign is capped; do not try to run it past its caps. A new campaign needs a new owner-authorized budget and valid decision schema.

Useful command shapes (adjust Python and data paths locally):

```powershell
python training/tinyderm_campaign.py status
python training/tinyderm_campaign.py preflight
python training/tinyderm_fold_audit.py <development-manifest.csv>
```

Only launch the one-run supervised path after new campaign authorization, fresh preflight, and review of the complete decision/config. Never launch a duplicate process.

## Strict data and evaluation limits

- The original 1,006-image final holdout was consumed on 2026-10-29. Never load, score, rebalance, or describe it as independent evidence. No holdout images, labels, IDs, or predictions are packaged here.
- Use only verified original ISIC 2018 development rows for a continuation unless the labels and provenance of another dataset are independently checked. ISIC 2019 expansion labels were not independently verified and are excluded from the current campaign.
- Existing development folds have a repeated model-selection history; all scores are exploratory, not an independent estimate. No result establishes clinical validity.
- Image-level and lesion-grouped scores do not establish patient-level independence. Duplicate, near-duplicate, and patient overlap limitations remain in the reliability reports.
- The current candidate learned from dermoscopic images. Phone photographs, unseen diagnoses, varied devices/lighting/skin tones, and clinical practice are unvalidated. Never use outputs for patient triage or diagnosis.
- Preserve the promotion rule and report macro-F1, accuracy, balanced accuracy, per-class recall/support, melanoma sensitivity/false alarms where applicable, and fold variability. Do not select a recipe using the consumed holdout.

## Included source and deliberately excluded artifacts

This package includes the campaign controller, safe data adapter, DINOv2/Hunch training and evaluation tools, dependency manifest, and aggregate top-five campaign reports. It excludes all image archives, image caches, EmbeddingGemma/Hunch embedding caches, datasets, lesion identifiers, and trained model binaries. The five candidates above are the only model results carried forward in this handoff.

Primary references: [TinyDerm repository](https://github.com/RustyDrops/TinyDerm), [Hunch source](https://github.com/undrgrnd-labs/hunch), [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), and [ISIC Archive](https://www.isic-archive.com/).

