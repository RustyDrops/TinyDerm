# Hunch-assisted training pilot

## Expanded-data Hunch continuation (2026-11-03)

Hunch and its local EmbeddingGemma 2 image encoder were installed and used to cache embeddings for all 23,894 development images. The expanded manifest combines the original non-holdout training/development rows with eligible seven-class ISIC 2019 images. All 10,015 IDs from the original manifest were excluded from the added cohort; the original consumed holdout remains excluded from training and metrics. The expansion contains 13,062 lesion groups, with no known overlap between added lesion IDs and the original cohort. Data preparation details and class counts are in `isic2019-expansion-report.json`; the source dataset is [ISIC 2019](https://doi.org/10.34970/2019-ds01), licensed CC BY-NC 4.0, so downstream use must honor its noncommercial terms.

I compared the expanded-data DINOv2-S/14 frozen-feature student with and without Hunch guidance using five lesion-grouped folds, seed 42017, nested fold-internal logistic teacher targets, square-root class weighting, and distillation weight 0.10. At the best pooled macro-F1 epoch (selected by looking across epochs), baseline scored `0.6148` macro F1 / `0.6166` balanced accuracy; Hunch scored `0.6208` / `0.6260`. At the best balanced-accuracy epoch, Hunch scored `0.6340` balanced accuracy / `0.6055` macro F1. Because epochs were selected using these same development folds, the best-epoch figures have selection optimism. Full metrics and checkpoints are under `hunch-isic2019-expanded-assisted-cv/`; the matched baseline is `hunch-isic2019-expanded-baseline-cv/`.

I then fine-tuned the last four DINOv2 transformer blocks and classification head for three epochs in three lesion-grouped folds, again comparing a matched control with Hunch distillation at weight 0.10. At the fixed epoch-3 endpoint, pooled out-of-fold macro F1 / balanced accuracy improved from `0.6154 / 0.6545` to `0.6204 / 0.6604`. Hunch raised class F1 for MEL (`0.626` to `0.630`), BCC (`0.729` to `0.736`), AKIEC (`0.419` to `0.425`), BKL (`0.546` to `0.554`), DF (`0.471` to `0.477`), and VASC (`0.670` to `0.675`); NV was effectively unchanged (`0.846` to `0.847`). The fold predictions, checkpoints, configuration, and per-class results are under `hunch-isic2019-dinov2-finetune-3ep-cv/`.

These results show a small, consistent development improvement from Hunch assistance on the expanded cohort, with the clearest gain in balanced accuracy. They are grouped cross-validation estimates after model and recipe exploration, not an independent external evaluation. Rare-class support remains low (DF 227 images; VASC 239), and this does not establish clinical performance. No full-development deployment candidate was produced from this experiment, and the original model was not replaced.

Two follow-ups did not improve the candidate. Four-flip test-time averaging on the saved fine-tuned fold checkpoints scored `0.6176 / 0.6547` with Hunch, below the single-view result `0.6204 / 0.6604`; the artifact is in `hunch-isic2019-dinov2-finetune-3ep-tta/`. Raising square-root class-weight strength from `0.5` to `1.0` in the frozen-feature five-fold run produced best pooled macro F1 `0.6008` for Hunch versus `0.6096` for its matched control. Maximum balanced accuracy was `0.6595` with Hunch, but at macro F1 `0.5572`, a poor precision/recall tradeoff. That run and its full epoch history are in `hunch-isic2019-expanded-classweight1-cv/`; retain the earlier `0.5` class-weight recipe.

I extended fine-tuning to six epochs on the same three lesion-grouped folds. At the fixed epoch-6 endpoint, control was `0.6452` macro F1 / `0.6458` balanced accuracy and Hunch was `0.6394 / 0.6544`. Hunch therefore improved balanced accuracy by `0.86` points while reducing macro F1 by `0.58` points. Its best pooled balanced accuracy across epochs was `0.6652` at epoch 3, with macro F1 `0.6050`; this epoch selection is development-tuned. A simple 50/50 blend at epoch 6 reached `0.6456 / 0.6517`. Second-level cross-fitted class-bias calibration reached `0.6470 / 0.6440` for control and `0.6414 / 0.6511` for Hunch; the calibrated blend was `0.6460 / 0.6501`. These modest tradeoffs do not establish a uniformly superior model. Full fold histories and checkpoints are in `hunch-isic2019-dinov2-finetune-6ep-cv/`; calibration outputs for epochs 3 and 6 are saved beside it.

Increasing fine-tuning resolution from 224px to 336px gave a stronger three-fold, three-epoch result. At epoch 3, control scored `0.6640 / 0.6981`, while Hunch at weight `0.05` scored `0.6624 / 0.7003` (macro F1 / balanced accuracy). A second-level cross-fitted 50/50 ensemble with class-bias calibration scored `0.6728 / 0.6924`; an adaptive blend with calibration scored `0.6707 / 0.6953`. The calibrated ensemble's class F1 was MEL `0.642`, NV `0.852`, BCC `0.790`, AKIEC `0.484`, BKL `0.591`, DF `0.565`, and VASC `0.786`; AKIEC remains the weakest class. The fold checkpoints and predictions are in `hunch-isic2019-dinov2-finetune-336px-alpha005-cv/`, with calibration in `hunch-isic2019-dinov2-finetune-336px-epoch3-calibration.json`.

## Latest model upgrade: DINOv2 features with lighter Hunch guidance

I audited the 9,009 development images: all files were present, readable, and the lesion-grouped manifest contained 6,723 lesion IDs. I then compared the MobileNet recipe with a frozen ImageNet-pretrained DINOv2-S/14 image encoder at 518px and a small learned classification head. Hunch remained the fold-internal teacher: its EmbeddingGemma image embeddings fed a class-balanced logistic model, and each student's soft targets were generated without the target lesion in that teacher's fit. The consumed original holdout was excluded.

The DINOv2 features were cached once; the head was trained over lesion-grouped folds. On the matched `seed=42017` folds, the best fold-held epoch selection used 15 epochs for the baseline and 29 for Hunch with distillation weight 0.10. A fold-held blend search selected 50–60% Hunch probabilities, then a second-level cross-fitted class-bias calibration produced macro F1 `0.7068` and balanced accuracy `0.7073`. The uncalibrated 50/50 ensemble scored `0.7063 / 0.7078`; the selected baseline scored `0.6968 / 0.6931`. The cross-fitted candidate's per-class recall was MEL `0.564`, NV `0.912`, BCC `0.765`, AKIEC `0.549`, BKL `0.679`, DF `0.699`, VASC `0.781`.

This is the strongest grouped development estimate recorded here so far, but it is still only a development estimate after recipe comparisons. Rare-class and melanoma recall remain imperfect, and no independent untouched benchmark is available. Do not treat it as clinically reliable. The full-development refit is in `dinov2-hunch-full-dev-candidate/`; `predict_dinov2_hunch.py` runs that research candidate on an image. The full-development weights have no measured performance because all development rows were used to fit them. OOF scores and nested selection details are in `hunch-dinov2-grouped-cv/`, `alpha01-selected/selection.json`, and `alpha01-calibration.json`; the comparison runs for distillation weights 0.10, 0.20, and 0.35 are also retained.

## Latest staged Hunch run: 5-fold lesion-grouped CV

The Hunch setup was exercised end-to-end again on 9,009 development images across 6,723 lesion groups. Five stratified folds kept all images from each lesion together. For each outer fold, the Hunch teacher and its student targets used only the outer training groups; teacher targets for student training were themselves generated with grouped out-of-fold teacher fits. A matched MobileNetV3-Small control and Hunch student both started from cached ImageNet weights. The original consumed holdout was excluded.

Training used 288px inputs, batch size 64, four head-only epochs followed by eight fine-tuning epochs, square-root class weighting (0.5), and Hunch distillation weight 0.35. This completed on the RTX 3060 in 9,763 seconds. Fold checkpoints, per-fold OOF predictions and complete reports are in `hunch-staged-grouped-cv/`; settings and pooled epoch results are in `crossvalidation.json`. Four-view flip scores are in `hunch-staged-grouped-cv-tta/crossvalidation.json`; second-level fold-held calibration results are in `hunch-staged-cv-calibration-tta.json`.

At epoch 12 with four-view flip averaging, pooled OOF macro F1 / balanced accuracy was `0.6719 / 0.6802` for the control and `0.6697 / 0.7087` for Hunch. Hunch raised MEL recall from `0.546` to `0.606`, BCC from `0.728` to `0.785`, and DF from `0.485` to `0.534`; NV recall fell from `0.898` to `0.860`. A simple 50/50 probability blend scored `0.6883 / 0.7071`. Fold results were mixed: Hunch improved macro F1 in one of five folds, while balanced accuracy was higher in four of five.

The nested second-level calibration raised Hunch macro F1 to `0.6805` and reduced balanced accuracy to `0.6929`; the uncalibrated blend remained higher in macro F1 (`0.6883`). Because calibration and recipe choices use the development pool, these figures are still development estimates and are subject to selection effects. They do not show sufficiently reliable all-class identification, and do not establish clinical performance. Do not replace the deployed model from this result. The earlier grouped run and its candidate remain documented below for comparison.

## Latest continuation: grouped CV and additional tuning

A five-fold `StratifiedGroupKFold` experiment split all 9,009 development images by lesion. Each MobileNet fold started from cached ImageNet-pretrained weights; the Hunch teacher was also restricted to that fold's training groups, with nested lesion-grouped OOF targets for student-training images. This avoids using the same 884-image validation subset as the sole comparison set. The consumed holdout stayed excluded.

After eight epochs, pooled out-of-fold macro F1 / balanced accuracy was `0.6650 / 0.6836` for the control and `0.6544 / 0.7120` for Hunch. The assist raised pooled MEL recall from `0.545` to `0.570` (996 examples), but lowered pooled MEL F1 from `0.531` to `0.523` and lowered overall macro F1 by about 1.1 points. At epoch 6 Hunch macro F1 was `0.6548` / balanced accuracy `0.7151`; epoch 8 lowered Hunch macro F1 slightly to `0.6544`.

Both models were then continued to epoch 12 at lower learning rates. Pooled OOF macro F1 / balanced accuracy was `0.6737 / 0.6733` for control and `0.6613 / 0.7089` for Hunch. MEL recall was `0.58` / `0.56`; BCC recall `0.75` / `0.80`; DF recall `0.48` / `0.62`. A second-level, lesion-fold cross-fitted class-bias calibration changed Hunch to `0.6802` macro F1 / `0.6947` balanced accuracy, compared with calibrated control at `0.6758 / 0.6743`. This calibration exchanged some of Hunch's recall advantage for precision; MEL recall became `0.51`. A plain average of baseline and Hunch OOF probabilities scored `0.6881 / 0.6948`. These are development OOF estimates after recipe exploration, not external performance claims.

Four-view flip averaging raised the epoch 12 OOF control to `0.6781 / 0.6767` and Hunch to `0.6757 / 0.7190`. A second-level fold-held blend search selected 70% Hunch in each of the five folds. That adaptive blend scored macro F1 `0.6990` / balanced accuracy `0.7168`, with MEL recall `0.578` (996 images). This is the strongest grouped result so far, though it remains a development estimate after repeated experimentation. The scoring and calibration details are in `hunch-grouped-crossval-12ep-tta/` and `hunch-cv-calibration-tta.json`.

The chosen 30% baseline / 70% Hunch inference blend was refit on all 9,009 development images from ImageNet-pretrained weights using the original staged schedule: eight head-only epochs and four fine-tuning epochs. Its member checkpoints are `hunch-grouped-full-dev-blend/baseline_epoch12.pt` and `hunch-grouped-full-dev-blend/hunch_assisted_epoch12.pt`; [ensemble configuration](hunch-grouped-full-dev-blend/ensemble.json) describes four-view averaging. These full-development refits have no measured validation score.

The epoch 8 CV checkpoints and metrics are under `hunch-grouped-crossval-8ep/`; the continuation to epoch 12 is under `hunch-grouped-crossval-12ep/`; cross-fitted calibration details are in `hunch-cv-calibration.json`. The results still do not show a very good, stable rate across every lesion class. Training from ImageNet for 12 epochs is shorter than the original staged schedule, so more grouped training or new independent data is needed before concluding.

Other follow-ups were not promoted:

- A 384px input gave the control `0.7432 / 0.7551`, but Hunch scored `0.7005 / 0.7613` at weight 0.35 and `0.7128 / 0.7426` at weight 0.10.
- Increasing class-weight strength from 0.5 to 0.75 or 1.0 raised balanced accuracy (Hunch `0.7920` and `0.7948`) but reduced macro F1 (`0.7141` and `0.6963`) versus weight 0.5 (`0.7374 / 0.7848`).
- The higher-macro-F1 RBF SVM Hunch teacher did not transfer its advantage to the student; the logistic teacher remains the selected head.
- A seed42 epoch1 + seed7 epoch3 four-view ensemble scored `0.7450 / 0.8036` on the repeatedly used validation split, slightly above its individual members. Its full-development refits are saved as a research candidate, but their combined score has not been measured.

The complete fold outputs and pooled per-class metrics are in `hunch-grouped-crossval-8ep/crossvalidation.json`; the script and fold checkpoints are in the same directory. This is grouped development cross-validation, not an independent external benchmark. The current evidence still does not establish a very good or clinically reliable identification rate.

## Latest completed Hunch run

Hunch was run again with its pinned EmbeddingGemma 2 checkpoint and the locally built `eg2cpu` backend. It embedded all 9,009 train and validation images (768 dimensions, 70 tokens), using SHA-256-verified cached embeddings. The class-balanced logistic teacher used training labels only; training targets were produced by five lesion-grouped out-of-fold fits. On the repeatedly used development validation split, the teacher scored macro F1 `0.5544` and balanced accuracy `0.7180`.

The paired GPU student comparison used the same 288px MobileNet starting checkpoint, deterministic seed 42, learning rate `2e-5`, class-weight strength `0.5`, Hunch KL weight `0.35`, and three-epoch schedule. At matched epoch 1, the baseline scored macro F1 `0.7130` / balanced accuracy `0.7478`; Hunch scored `0.7374` / `0.7848`. Hunch epochs 2 and 3 scored `0.7333` / `0.7832` and `0.7191` / `0.7703`. These are development comparisons only; the same split has been used repeatedly to choose settings.

After comparison, the assisted model was retrained for three fixed epochs on all 9,009 development examples (train plus validation). No validation performance was computed during this fit. Training loss was `0.4195`, `0.3979`, and `0.3880` for epochs 1–3. The epoch 1 full-development weights are the research candidate aligned with the previously selected epoch 1 recipe; epochs 2–3 are retained for further research. The full-development fit has no independent performance estimate. The consumed holdout was not loaded or evaluated. The original deployed weights remain unchanged.

Artifacts: `hunch-assisted-final-run/results.json`, `settings.json`, the three `hunch_assisted_full_dev_epoch*.pt` checkpoints, paired validation checkpoints, `hunch_teacher.joblib`, and the local embedding cache. The model checkpoint is `hunch-assisted-final-run/hunch_assisted_full_dev_epoch1.pt`.

## Seed ensemble and improved Hunch head search

Flip averaging and two independently seeded, train-only Hunch students were compared on the development validation split. The strongest pair was seed 42 epoch 1 plus seed 7 epoch 3: averaged four-view predictions (original, horizontal flip, vertical flip, both flips) scored macro F1 `0.7450` and balanced accuracy `0.8036`. The stronger individual member with the same four-view averaging scored `0.7386` / `0.8003`. The ensemble recalls were MEL `0.430` (89 images), NV `0.910` (603), BCC `0.915` (47), AKIEC `0.889` (27), BKL `0.688` (96), DF `0.800` (10), and VASC `1.000` (12). This is a modest development-only gain and the validation set has been reused for model selection.

Classifier head search on cached Hunch embeddings found an RBF SVM with macro F1 `0.647` / balanced accuracy `0.692`, compared with the selected logistic teacher's `0.554` / `0.718`. The better macro F1 did not transfer to the student: its assisted student scored `0.730` / `0.739` at epoch 1 and peaked at `0.733` / `0.746` at epoch 2, below the logistic-assisted weighted recipe. The SVM head was therefore rejected for the student candidate. Its experimental checkpoints and metrics remain in `hunch-rbf-teacher-run/`.

To prepare an ensemble research candidate, a second full-development logistic-assisted model was trained with seed 7 for three fixed epochs. It used the same pinned Hunch encoder, grouped out-of-fold teacher targets, GPU, and weighted loss. Training loss was `0.4164`, `0.3899`, and `0.3859`. The checkpoint `hunch-assisted-full-dev-seed7/hunch_assisted_full_dev_epoch3.pt` has no validation score because all development examples were used in training. It is paired with the seed 42 epoch 1 full-development checkpoint in `hunch-assisted-ensemble-candidate.json`; the reported ensemble score comes from the corresponding train-only models before full-development refitting and is not a measured score for the refit pair.

Historical note: product-facing app artifacts are outside this training handoff. The research dataset, source and run artifacts are maintained separately under their applicable access and licensing terms.

## Full development run

The first small pilot below was followed by a full development run in `hunch-full-development/`. Hunch encoded all 9,009 train and validation images locally (768 dimensions, 70 soft tokens). A class-balanced logistic teacher used only training labels. Its train targets were generated with five-fold `StratifiedGroupKFold` by lesion, so each training lesion's Hunch target came from a fold model that had not seen that lesion. The validation predictions came from a teacher fitted on the complete training split. The Hunch teacher scored macro F1 `0.5544` and balanced accuracy `0.7180` on validation.

Both MobileNet arms started from the same prior development checkpoint, used the GPU, the same five epoch limit, batch size 32, and nominal seed 42. The assisted arm used a 50/50 cross-entropy and teacher KL loss. Comparing at the same two-epoch budget, the Hunch arm reached macro F1 `0.7072` and balanced accuracy `0.7520`; the control reached macro F1 `0.7124` and balanced accuracy `0.6982`. This is a promising increase in balanced accuracy (+5.4 percentage points) with a small macro F1 decrease (-0.5 points). At epoch 5, the control's macro F1 rose to `0.7258`, while the assisted arm scored `0.7046`; assisted balanced accuracy remained `0.7479`.

At epoch 2, class recall changed as follows:

| Class | Control recall | Hunch recall | Validation images |
| --- | ---: | ---: | ---: |
| MEL | 0.360 | 0.393 | 89 |
| NV | 0.952 | 0.922 | 603 |
| BCC | 0.851 | 0.915 | 47 |
| AKIEC | 0.667 | 0.778 | 27 |
| BKL | 0.458 | 0.656 | 96 |
| DF | 0.600 | 0.600 | 10 |
| VASC | 1.000 | 1.000 | 12 |

The assist improved recall for MEL, BCC, AKIEC, and BKL in this validation run, with the largest change for BKL. It reduced NV recall slightly, while DF and VASC validation support is small. Epoch 2 is a development-selected checkpoint; validation has now informed model choice. These are not independent final metrics: the starting checkpoint was selected using this development split, the validation split has guided experiments, and the original holdout is consumed. The results justify continued research, but do not establish reliable clinical performance.

Full metrics and all ten epoch checkpoints are recorded in `hunch-full-development/results.json` and its `.pt` files. `embeddings.sqlite` is a reusable image-hash keyed cache, and `hunch_teacher.joblib` contains the teacher. The original deployable model remains unchanged.

## Teacher-weight follow-ups

Two additional full-development, matched two-epoch comparisons reused the same 9,009 image embedding cache and grouped teacher targets. Their control starts varied across runs, so report the individual paired results rather than claiming a precise generalizable gain:

| Hunch loss weight | Run | Control macro F1 / balanced accuracy | Assisted macro F1 / balanced accuracy |
| ---: | --- | ---: | ---: |
| 0.50 | Full run | 0.712 / 0.698 | 0.707 / 0.752 |
| 0.70 | Follow-up | 0.707 / 0.705 | 0.689 / 0.751 |
| 0.35 | Follow-up | 0.698 / 0.668 | **0.723 / 0.751** |

The 0.35 setting was repeated under deterministic GPU settings with seeds 42, 7, and 123. At epoch 2, seed 42 scored control `0.719 / 0.711` and assisted `0.738 / 0.762`; seed 7 scored control `0.717 / 0.721` and assisted `0.725 / 0.741`; seed 123 scored control `0.700 / 0.720` and assisted `0.685 / 0.717`. Across these three runs, mean control macro F1 / balanced accuracy was `0.712 / 0.717`; mean assisted was `0.716 / 0.740`. The average gain is about `0.004` macro F1 and `0.023` balanced accuracy. Two seeds improved both measures and one regressed slightly, so these results are promising but do not establish a guaranteed gain.

In the seed 42 run, MEL recall increased from `0.382` to `0.427`, BCC from `0.851` to `0.936`, AKIEC from `0.704` to `0.815`, and BKL from `0.490` to `0.615`; NV recall changed from `0.950` to `0.939`. DF and VASC remain based on only 10 and 12 validation images.

## Lower learning rate

The next experiment reduced MobileNet's learning rate from `5e-5` to `2e-5`, held Hunch weight at 0.35, and trained for three epochs. At epoch 3, the assisted arm reached macro F1 `0.7323` and balanced accuracy `0.7578`; its paired control reached `0.7207` and `0.7121`. This is the strongest result so far, though it still comes from the repeatedly used development validation split. The checkpoint and its full per-class metrics are saved as `hunch_candidate_lr2e5_epoch3.pt` and `hunch_candidate_lr2e5_epoch3.json`. It remains a research candidate; the original deployed weights have not been replaced.

## Class-weighted loss

Adding the existing square root inverse-frequency weighting to the hard-label loss improved balanced accuracy further. With seed 42, learning rate `2e-5`, Hunch weight `0.35`, and class-weight strength `0.5`, the assisted arm's best checkpoint was epoch 1 (`0.7374` macro F1, `0.7848` balanced accuracy), compared with the control's epoch 1 (`0.7130`, `0.7478`). A repeat with seed 7 reached `0.7317` macro F1 and `0.7945` balanced accuracy at epoch 3, compared with `0.7287` and `0.7490` for its epoch 3 control. At matched epoch 1, Hunch improved both measures for both seeds. The seed 7 epoch 3 candidate had MEL recall `0.449`, NV `0.879`, BCC `0.957`, AKIEC `0.778`, BKL `0.698`, DF `0.800` (10 validation images), and VASC `1.000` (12 images).

The weighted candidates are saved as `hunch_candidate_balanced_ce_epoch1.pt` and `hunch_candidate_weighted_seed7_epoch3.pt`, with their per-class metrics beside them. They are research artifacts for further comparison, not approved deployable medical models. Repeated validation selection means their scores are no longer independent estimates.

At weight 0.35, the assisted arm raised recall on MEL, BCC, AKIEC and BKL compared with its paired control, while reducing NV recall from 0.955 to 0.937. DF and VASC have only 10 and 12 validation examples, so their rates are especially unstable. The 0.35, epoch 2 checkpoint is saved as `hunch-followup-035/hunch_assisted_epoch2.pt`. It is a research candidate, not a replacement for the original deployment model. Validation has now been used repeatedly to compare recipes; any next result remains development evidence. Do not run the consumed holdout again.

Hunch source revision `6541bd157e11a5e7d7434360e99ea81f40a8c181` and its `eg2cpu` extension were built locally with a workspace-local Rust toolchain. The upstream-pinned EmbeddingGemma 2 checkpoint `914f7f89142e33e77833254d9c9b90c3cef7303b` was downloaded to the workspace. Hunch's `Eyes` encoder produced 768-dimensional embeddings locally; no images were sent to a cloud teacher.

## Experiment

`train_hunch_assisted.py` restricted the original manifest to train and validation groups and checked for lesion overlap. It chose the first 20 images per class per split (fewer where a validation class had fewer than 20): 140 training images and 122 validation images. A standardized, class-balanced logistic classifier was fitted on the Hunch embeddings and expert training labels. Its training-set probabilities guided a MobileNet student using a half cross-entropy, half KL-divergence loss; an unassisted control used cross-entropy. Both started from the existing `best_dev_model.pt` and ran two epochs at the same learning rate, using the same nominal seed. The starting checkpoint had already been selected with this development validation set. This is an exploratory pilot, not an independent assessment. The dataset is dermoscopic, and these results have no established clinical application.

| Pilot | Hunch teacher macro F1 | Control macro F1 / balanced accuracy | Assisted macro F1 / balanced accuracy |
| --- | ---: | ---: | ---: |
| Initial teacher, C=1 | 0.551 | 0.733 / 0.738 | 0.733 / 0.738 |
| Regularized teacher, C=0.001 | 0.556 | 0.720 / 0.724 | 0.685 / 0.671 |

All scores are development validation after epoch 2, except the teacher score. Training weights differed in the first pilot even though the reported classification scores were identical. Its teacher achieved 100% training accuracy with mean true-class probability 0.991, so its in-sample targets were nearly one-hot. The control scores also differed across separate runs despite the nominal seed; exact reproducibility was not established. The second configuration was tried after examining the first, further limiting inference from the repeated validation set.

The pilot does **not** demonstrate an improvement. Do not replace the existing model with either candidate on this evidence. A useful follow-up would train and evaluate on the full development pool, use lesion-grouped out-of-fold teacher predictions to reduce in-sample teacher overfitting, evaluate uncertainty across multiple seeds, and reserve a genuinely independent external benchmark for final assessment. The consumed original holdout was not evaluated again.

Detailed metrics, per-class reports and checkpoints are in `hunch-pilot/` and `hunch-pilot-regularized/`; the embedding cache is local and resumable. The existing final model was not changed.

