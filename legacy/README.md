# `legacy/` — historical scripts, not for new work

This folder holds the original monolithic entry-point scripts that lived at
the repository root before the codebase was reorganised into the modular
`models/`, `data/`, `losses/`, `evaluation/`, `training/`, and `utils/`
packages. The scripts here are preserved for **reference and one-off reruns
of completed jobs** — they are not maintained, not covered by tests, and
should not receive new feature work.

If you reach into this folder during day-to-day development, you are almost
certainly looking at the wrong file. Use the active equivalent under
`training/` instead.

## Migration map

| Legacy script (in this folder)         | Active replacement                  |
|----------------------------------------|-------------------------------------|
| `predict_scratch_segment.py`           | `training/train_scratch.py`         |
| `predict_scratch_segment_h5.py`        | `training/train_scratch_h5.py`      |
| `predict_TSO_segment.py`               | `training/train_tso.py`             |
| `predict_TSO_segment_patch.py`         | `training/train_tso_patch.py`       |
| `predict_TSO_segment_patch_h5.py`      | `training/train_tso_patch_h5.py`    |
| `predict_TSO_segment_patch_dlrtc.py`   | `training/train_tso_dlrtc.py`       |
| `predict_TSO_segment_patch_ray.py`     | *(no Ray-specific replacement; use the non-Ray patch trainer or contact the maintainers if Ray-distributed inference is still required)* |
| `pretrain.py`                          | `training/pretrain.py`              |
| `convert_parquet_to_h5.py`             | `training/convert_h5.py`            |

Each file in this folder begins with a banner restating the above mapping.

## Why this folder still exists at all

Two practical reasons:

1. **Reproducibility of historical runs.** Several published-paper and
   internal-report results were generated with these exact scripts. Keeping
   the byte-equivalent files (plus the legacy `Helpers/DL_models.py` and
   `Helpers/DL_helpers.py` they depend on) makes it possible to re-execute
   a run from a year ago without git-archaeology.
2. **Soft-deprecation buffer.** The active replacements have diverged
   slightly from the originals (different pip-preamble pins, occasional
   bug fixes), so a quiet deletion would break any stale Domino job or
   notebook that still calls into the legacy path. The deprecation banner
   gives those callers a visible warning before they fail.

## When this folder can be removed entirely

When (a) no internal Domino job, notebook, or shell script references any
file in this folder, and (b) the corresponding modules in `Helpers/` no
longer have any imports from outside `legacy/`. At that point the entire
`legacy/` folder, plus `Helpers/DL_models.py`, `Helpers/DL_helpers.py`,
and the three remaining files in `Helpers/net/` (`pretrainer.py`,
`dataparallel_pretrainer.py`, `embed.py`), become deletable in one
commit.
