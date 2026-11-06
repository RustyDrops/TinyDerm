# TinyDerm campaign synthesis, 2026-11-05

The campaign goal was macro-F1 >= 0.73851 on 9,009 development images with no class recall more than 0.05 below H001. H001 remains best at 0.72851 macro-F1 and 0.81885 accuracy. No candidate met both conditions.

| Run / hypothesis | Seed / epochs | Selected arm | Macro-F1 | Accuracy | Lowest recall change vs H001 | Promotion |
|---|---|---|---:|---:|---|---|
| H001, `run_20261104T185647Z_a16a00a9` | 52711 / 3 | Hunch | 0.72851 | 0.81885 | reference | Reference |
| H028, `run_20261105T180200Z_33916d10` | 52711 / 5 | Hunch | 0.73096 | 0.81774 | DF −0.06796 | Fail: below macro target and DF safeguard |
| H031, `run_20261105T195021Z_89314c90` | 52711 / 3 | Baseline | 0.72824 | 0.82107 | VASC −0.01562 | Fail: below macro target |
| H032, `run_20261105T213028Z_17e1c9bb` | 52711 / 4 | Baseline | 0.72909 | 0.82773 | AKIEC −0.06102 | Fail: below macro target and AKIEC safeguard |
| H033, `run_20261105T223655Z_7a28a9d5` | 52716 / 3 | Baseline | 0.71494 | 0.80941 | BCC −0.03913 | Fail: below macro target |

All five run reports reference the same development identity fingerprint. H001, H028, H031 and H032 use seed 52711; H033 changes seed and therefore fold assignments. Interpret H033 as descriptive seed sensitivity, not a paired comparison. H033's decision rationale incorrectly described four epochs; the actual decision left `epochs` null, and the controller used its three-epoch default. This was identified and documented in [the H033 result report](h033_seed_confirmation.md).

The two most informative follow-ups were H032's four-epoch schedule check and H033's new-seed reference run. H032 improved accuracy by 0.00888 and macro-F1 by 0.00058 versus H001, while trading away melanoma recall (−0.04217), AKIEC recall (−0.06102), and DF recall (−0.04854) to increase BKL recall (+0.05662). H033's Hunch arm changed macro-F1 by −0.00050 and accuracy by +0.00067 versus its matched baseline, providing no meaningful teacher lift. H028's Hunch arm similarly failed to improve meaningfully versus its image-only control; details are in [the H028 tradeoff audit](run15_melanoma_error_tradeoff.md).

The H032/H033 error review found no consistent image-quality or apparent label pattern supporting relabeling. The 25 AKIEC cases that H032 changed from correct under H001 to incorrect under H032 were predicted as MEL (7), BCC (3), or BKL (15), with varied image appearances. No data or labels were edited. See the [development-only contact sheet](h032_akiec_regressions.jpg) and the combined [saved-error audit](h001_h028_h031_error_audit.md).

One untested next-campaign hypothesis is a single-change augmentation ablation: reduce or disable random erasing while holding the data, folds, seed, and training settings fixed. The active trainer imports the existing transform from `tinyderm/augment.py`, which currently applies random erasing with probability 0.25 and area 2%–15%, plus random resized crops at scale 0.80–1.0. Since errors concentrate in visually overlapping AKIEC/BKL and MEL/NV classes but do not share a clear label or image-quality issue, retaining more lesion detail may help. This is a hypothesis only; no ablation was run, and it requires a new authorized campaign plus a versioned transform change. Reject it if it fails the unchanged promotion rule or causes a class recall regression beyond 0.05.

Controller status at this update: paused without error, 18/18 full runs and 25/25 decisions used; approximately 3.97 hours remain on the elapsed wall-clock budget. The campaign cannot launch another run under its current caps. The regression suite passed 20/20 with `TEMP` and `TMP` pointed to the project-local `.test_tmp` directory. The consumed final holdout remains excluded. Any further run requires a newly authorized campaign budget and a valid decision whose declared settings match its actual configuration.

