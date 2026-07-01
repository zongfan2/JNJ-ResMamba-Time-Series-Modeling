# Experiment configs

YAML configs are split by paper / purpose:

```
experiments/configs/
├── deep_tso/        # Deep TSO paper (IMWUT paper 2) — cross-dataset train UKB / test noprod
│   ├── deep_tso_e1_*.yaml          # E1 architecture ablation (Table 4)
│   ├── deep_tso_phase1_*.yaml      # E2/E3/E4 arms (Tables 1,2,3,5,6): CE, GCE, SupCon,
│   │                               #   consistency, structural priors, interval head
│   ├── deep_tso_phase2_*.yaml      # Phase 2 annotator-consensus weighting (supplementary)
│   ├── deep_tso_ukb2noprod.yaml    # SUPERSEDED reference template (see file header)
│   └── tso_*.yaml                  # older aggregated-feature TSO configs
│
├── deep_scratch/    # Deep Scratch paper (IMWUT paper 1)
│   ├── scratch_*.yaml              # main scratch training configs
│   ├── smoke_pretrained_mbav1.yaml
│   ├── ablation/                   # architecture / baseline / pretrain-freeze ablations
│   └── deployment/                 # classical-baseline deployment configs
│
└── pretrain/        # shared upstream self-supervised pretraining on UKB (DINO / MAE)
    └── pretrain_ukb_*.yaml
```

Run scripts reference these paths directly:
- Deep TSO tables: `experiments/deep_tso/tables/` (one script per paper table) and
  `experiments/deep_tso/run_deep_tso_{ablation,e1,smoke}.sh`.
- Deep Scratch: `experiments/deep_scratch/run_scratch_mbav1.sh`, `run_ablation.sh`,
  `run_deploy_classical.sh`, `run_smoke_pretrained.sh`.
- Pretraining: `experiments/pretrain/run_pretrain_ukb_{dino,mae}.sh`.

Design rationale for the Deep TSO experiments: `docs/deep_tso/experiment_plan.md`.
