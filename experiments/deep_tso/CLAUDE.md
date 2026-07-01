# Deep TSO — Claude context

Agent onboarding for the **Deep TSO** project. The full guide lives in the sibling
`AGENTS.md` (single source of truth) — it is imported below. Human guide:
`docs/deep_tso/README.md`; design spec / Domino runbook: `docs/deep_tso/experiment_plan.md`.

**Three things to internalize first (details in AGENTS.md):**
1. **Cross-cohort design** — every experiment trains/validates on **UKB** (`predictTSO`) and
   tests on **all of noprod** (`inTSO` anchor). Single split, not k-fold. Uncertainty =
   spread **across noprod test subjects** (`gt_*_subj_std`), never per-fold std.
2. **Runs on Domino only** — no local `mamba_ssm` / GPU / data. Locally: edit + `py_compile`
   / `bash -n`, don't train.
3. **`f1_tso` is fidelity to the noisy labeler, never accuracy.** Headline = `gt_*` metrics
   vs `inTSO`; van-Hees is a reference row, not a beaten baseline.

@AGENTS.md
