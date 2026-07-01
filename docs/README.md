# Documentation index

Project documentation for the ResMamba wearable time-series codebase, which supports two
research papers: **Deep Scratch** (scratch detection) and **Deep TSO** (sleep-window
detection). See the repo root `CLAUDE.md` / `AGENTS.md` for the full codebase map.

## General

| Doc | What it covers |
|---|---|
| [`project_overview.md`](project_overview.md) | Consolidated architecture, models, data pipeline, design |
| [`algorithms.md`](algorithms.md) | Post-processing algorithms (smoothing, single-TSO enforcement) |

## Deep TSO (paper 2)

| Doc | What it covers |
|---|---|
| [`deep_tso/README.md`](deep_tso/README.md) | **Front door** — what Deep TSO is, where things live, quick start |
| [`deep_tso/experiment_plan.md`](deep_tso/experiment_plan.md) | Cross-cohort design spec + per-experiment Domino runbook |
| [`deep_tso/innovation_report.html`](deep_tso/innovation_report.html) | Rendered innovation summary |

Related, outside `docs/`: the paper source (`papers/JNJ_deepTSO/`), configs
(`experiments/configs/deep_tso/`), and run scripts (`experiments/deep_tso/`, incl.
`tables/` — one script per paper table).

## Deployment

| Doc | What it covers |
|---|---|
| [`deployment/README.md`](deployment/README.md) | Multi-GPU, distributed training, Domino setup |
| [`deployment/distributed_training.md`](deployment/distributed_training.md) | Distributed training guide |
| [`deployment/domino_ray.md`](deployment/domino_ray.md) | Domino + Ray deployment guide |

## Archive

[`archive/`](archive/) holds superseded or historical docs (the DLRTC guides, the older
single-TSO/prediction-smoothing notes, and the codebase-overview snapshot). Kept for
reference; not maintained.

`superpowers/` holds dated planning/spec documents from earlier development iterations.
