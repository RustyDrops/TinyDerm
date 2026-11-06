# Development error audit: H001, H028 and H031

2026-11-05. Decision: **remain paused; no full run justified by this audit**. Goal remains macro-F1 >= 0.73851 with no class recall more than 0.05 below H001. H001 remains the campaign reference. The three-hour session request does not override the handoff's requirement for an actionable hypothesis before training.

## Scope and verification

Recomputed seven-class argmax from saved final-epoch OOF arrays on the original 9,009 development images. All reports match development identity SHA-256 `a56f28410e5923f05f96069bc97bcdd604792747ad80f86818ac47b9dd0f8c9f`; each image occurs once, fold assignments match, and all runs use seed 52711. H001 and H028 comparisons use their selected Hunch arms; H031 uses its selected baseline, with its Hunch arm considered separately. H028 changes the entire cosine schedule along with epoch count; this is not an isolated comparison of the final two epochs. Repeated selection on these folds makes all findings exploratory. The consumed holdout was not accessed.

## DF regression and melanoma tradeoffs

| Fold | DF correct H001 / H028 / H031 | H028 DF losses / gains vs H001 | H028 melanoma catches / losses vs H001 | H028 new / removed non-MEL false alarms |
|---|---:|---:|---:|---:|
| 1 | 19 / 16 / 20 of 34 | 3 / 0 | 50 / 4 | 230 / 5 |
| 2 | 24 / 21 / 24 of 35 | 3 / 0 | 14 / 26 | 27 / 77 |
| 3 | 24 / 23 / 24 of 34 | 1 / 0 | 8 / 31 | 4 / 140 |
| Total | 67 / 60 / 68 of 103 | 7 / 0 | 72 / 61 | 261 / 222 |

H028's seven lost DF images were predicted as MEL (1), BKL (2), BCC (2), AKIEC (1), and VASC (1). They represent five lesion IDs: two pairs share a lesion. This small, heterogeneous set does not isolate a single confusion class or fold. The [contact sheet](df_regression_contact.jpg) shows varied appearances and no obvious shared acquisition defect; this is not a diagnostic or label-quality judgment. H028 DF recall was 0.58252 versus H001 0.65049, a 0.06796 drop. Its macro-F1 0.73096 and accuracy 0.81774 did not satisfy promotion. Fold 1 accounts for the bulk of its extra melanoma detections and false alarms; folds 2 and 3 lose detections and reduce false alarms. The full paired and within-H028 arm comparison is in [run 15 tradeoff](run15_melanoma_error_tradeoff.md).

H031's selected baseline macro-F1/accuracy were 0.72824/0.82107; H001's were 0.72851/0.81885. H031 caught five and lost 14 previously detected melanomas, while adding 11 and removing 48 non-melanoma false alarms. Its DF correct count improved by one. Its Hunch arm had macro-F1 0.72503, DF correct 65/103, and caught 13 but lost five melanomas versus its own matched baseline while adding 34 net false alarms. This within-run arm comparison is separate from the H001 comparison and does not support stronger confidence-weighted distillation as a route to the target.

## Hypothesis and decision

The evidence is consistent with the longer five-epoch schedule shifting class boundaries in a fold-dependent way, improving melanoma detection in fold 1 while losing DF cases across folds. A falsifiable *diagnostic* hypothesis is that the epoch/schedule change causes the seven DF losses and fold-dependent melanoma shift: a matched, independently repeated schedule comparison that does not reproduce those directional changes would count against it. The current results alone cannot distinguish schedule length from stochastic variation or show which permitted single change would achieve both promotion constraints. In particular, the saved fourth-epoch values under a five-epoch schedule already show poor DF recall and do not predict a separate four-epoch cosine schedule. More training at four epochs is therefore not justified by this audit. Increased class weighting and stronger distillation have already been tested without a qualifying gain.

No decision slot or training slot was spent. Preflight passed with CUDA and all hard gates at 20:58 UTC; status was paused, no error, 16/18 runs and 23/25 decisions used, approximately 6.69 hours remaining in the existing 33-hour wall-clock budget. Recheck live status and preflight before any future action. Reproduction: run `tinyderm_research/audit_h001_h028_h031.py` from the project root with the project root on `PYTHONPATH` using the configured training environment.

## Follow-up H032 four-epoch schedule check

H032 (`run_20261105T213028Z_17e1c9bb`) completed successfully on CUDA in 63.4 minutes. With the same seed and grouped folds, the selected baseline scored macro-F1 0.72909 and accuracy 0.82773, versus H001 0.72851 and 0.81885. Hunch scored 0.72689/0.82517. Against H001, H032 baseline caught 28 previously missed melanomas and lost 70 previously detected melanomas; it removed 234 and added 71 non-melanoma melanoma false alarms (net −163). This reduced false alarms at a substantial melanoma-recall cost.

H032 class recall changes versus H001 were MEL −0.04217, NV +0.01640, BCC −0.01957, AKIEC −0.06102, BKL +0.05662, DF −0.04854, VASC −0.00781. It fails the recall safeguard on AKIEC and misses the 0.73851 macro-F1 target. Fold-level correct counts were: DF H001/H032 = 19/16, 24/25, 24/21; AKIEC = 42/44, 60/38, 46/48. The tradeoff varies by fold. Twenty-five AKIEC images that H001 classified correctly became errors under H032, predicted as MEL (7), BCC (3), and BKL (15). Their [contact sheet](h032_akiec_regressions.jpg) shows varied appearances. No reliable label-quality or acquisition cluster was identified, so no development labels or images were changed.

One final run was authorized and launched within the existing caps: H033 seed confirmation (`campaign/decisions/decision_025.json`), four epochs with seed 52716 and all other H032 settings fixed. Since the seed changes fold assignments, its comparison to H032 is descriptive rather than paired. The controller is now at 17/18 runs and 25/25 decisions; no further campaign run can be started within current decision limits. Continue monitoring the live status and do not extend budgets implicitly.

