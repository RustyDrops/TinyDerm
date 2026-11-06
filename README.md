# TinyDerm Lab

**Research only. Not a diagnostic system or a substitute for medical care.**

TinyDerm studies compact dermatology image classifiers and teacher-guided training. Historical README claims about an untouched sealed holdout, a deployable camera app, and current model performance are retired: the final holdout was consumed in earlier work and must never be evaluated or described as independent evidence.

See [TRAINING_CONTINUATION.md](TRAINING_CONTINUATION.md) for the current top five development candidates, the training stack (including Hunch and EmbeddingGemma 2), setup, continuation constraints, and research limits. Source-only training tools are in `training/`.

## Current state

The supervised three-fold campaign is paused at its limits (18/18 runs and 25/25 decisions). H001 remains the campaign reference under the unchanged promotion safeguards. No result met the promotion rule. Another run requires a newly authorized campaign budget and decision capacity.

## Repository contents

Research source and documentation only. No patient or lesion identifiers, image data, image or embedding caches, trained checkpoints, or final-holdout data are included. Obtain ISIC data under its published license and keep local caches and weights outside GitHub.

## Credits

ISIC/HAM10000, PyTorch, timm, scikit-learn, Hunch, and Google's EmbeddingGemma 2. Their licenses and citations apply; see the handoff.