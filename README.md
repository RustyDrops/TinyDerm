# TinyDerm Lab

**Research only. Not a diagnostic system or a substitute for medical care.**

TinyDerm studies compact dermatology image classifiers and teacher-guided training. Historical README claims about an untouched sealed holdout, a deployable camera app, and current model performance are retired: the final holdout was consumed in earlier work and must never be evaluated or described as independent evidence.

See [TRAINING_CONTINUATION.md](TRAINING_CONTINUATION.md) for the current top five development candidates, the training stack (including Hunch and EmbeddingGemma 2), setup, continuation constraints, and research limits. Source-only training tools are in `training/`.

## Top five development candidates

Ranked by selected-arm macro-F1 from the latest five-run campaign on 9,009 development images:

| Rank | Run | Selected arm | Macro-F1 | Accuracy | Lowest class-recall change vs H001 | Outcome |
|---:|---|---|---:|---:|---|---|
| 1 | H028 | Hunch | 0.73096 | 0.81774 | DF −0.06796 | Fails promotion target and DF safeguard |
| 2 | H032 | Image-only baseline | 0.72909 | 0.82773 | AKIEC −0.06102 | Fails promotion target and AKIEC safeguard |
| 3 | H001 | Hunch | 0.72851 | 0.81885 | Reference | Campaign reference; below target |
| 4 | H031 | Image-only baseline | 0.72824 | 0.82107 | VASC −0.01562 | Fails promotion target |
| 5 | H033 | Image-only baseline | 0.71494 | 0.80941 | BCC −0.03913 | Different seed/folds; below target |

Promotion required macro-F1 ≥ 0.73851 and no class-recall regression greater than 0.05 versus H001; none passed. These are exploratory, repeatedly selected development-CV results, not independent or clinical performance estimates. The full interpretation is in [TRAINING_CONTINUATION.md](TRAINING_CONTINUATION.md).

## Current state

The supervised three-fold campaign is paused at its limits (18/18 runs and 25/25 decisions). H001 remains the campaign reference under the unchanged promotion safeguards. No result met the promotion rule. Another run requires a newly authorized campaign budget and decision capacity.

## Repository contents

Research source and documentation only. No patient or lesion identifiers, image data, image or embedding caches, trained checkpoints, or final-holdout data are included. Obtain ISIC data under its published license and keep local caches and weights outside GitHub.

## Credits

ISIC/HAM10000, PyTorch, timm, scikit-learn, Hunch, and Google's EmbeddingGemma 2. Their licenses and citations apply; see the handoff.