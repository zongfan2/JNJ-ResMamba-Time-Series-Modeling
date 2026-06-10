# Deep TSO Noisy-Label Domino Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the next Deep TSO research version from `/Users/zongfan/Downloads/deep_tso_innovation_report.html`: robust noisy-label training, cross-night subject supervision, consensus label weighting, and Domino-ready evaluation.

**Architecture:** Keep `MBA4TSO_Patch` as the per-minute encoder/head, but make the H5 data contract richer: each segment exposes subject ID, optional annotator label tracks, and confidence weights. Training composes a robust base loss, structural priors, optional cross-night SupCon, and label-free validation metrics without requiring local training.

**Tech Stack:** PyTorch, H5Py, NumPy, scikit-learn metrics, existing Mamba/ResNet modules, YAML experiment configs, Domino jobs.

---

## Revision Note (2026-06-09, post code-review)

This plan was reviewed against the live codebase before execution. The following corrections were folded in; read them before implementing because they change several task bodies:

1. **GCE loss routing (correctness/ablation).** `run_model_tso_h5` only routes to `measure_loss_tso_structural` when a structural weight is non-zero (`training/train_tso_patch_h5.py:404`). GCE is wired only into that function, so `base_loss="gce"` with zero structural weights would *silently fall back to plain CE* (violates "errors should never pass silently"), and bundling `w_trans`/`w_dur` into the GCE arm would confound the CE-vs-GCE comparison. **Fix:** the routing condition now also triggers on `base_loss != "ce"` and `supervision_weight is not None` (Task 4), and the Phase-1 configs are a clean one-variable-at-a-time ladder: CE → GCE → GCE+SupCon, all with structural weights `0` (Task 5).
2. **Cross-night SupCon needs same-subject nights in a batch (methodology).** Batching is a plain shuffle (`batch_generator_h5:146`); with random batches most have 0–1 nights per subject, so `SupConLossV2` has no positives and is a near-no-op. **Fix:** Task 4 adds a subject-grouped batch generator used only when `w_supcon > 0`, plus a zero-positive guard.
3. **Consensus path destroys the padding mask (correctness).** `pad_Y` encodes padding as `-100` (`data/padding.py:604`) and the base loss ignores `-100` — so the plain GCE path is safe. But Task 6's `torch.where(nonwear_mask, pad_Y, consensus_labels)` overwrites `-100` with `0/2`, after which `masked_fill(pad_Y == -100, ...)` is a no-op, training on padded minutes. **Fix:** Task 6 preserves `-100` and computes the weight mask from the original padding mask.
4. **Edit-anchor corrections.** Task 3's `data/padding.py` edits were re-verified against the live function (`Y_batch`, `num_minutes_max`, `segments_batch`, the per-minute `np.any` aggregation, and the 4-value return) and are correct as written — no change needed there. Two real anchor fixes were added elsewhere: Task 1 now threads `skip_connect`/`skip_cross_attention` through `models/setup.py` (the existing `case` block dropped them), and Task 3 fixes a pre-existing channel-count bug in `convert_h5.py:390` (`5` → `num_channels`).

5. **Data source + interpreter (added post-implementation).** The supervised TSO H5 is built from the **labelled GENEActive production** parquet (`/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_train_van_new_enh_lth-rth/raw/`, which carries `predictTSO`/`non-wear`) via `experiments/domino/build_deep_tso_h5.sh` → `training/convert_h5.py`. This is NOT UKB — UKB is the *unlabelled pretraining* set (`run_preprocess_ukb.sh`). `test-tools/check_parquet_columns.py` guards against `convert_h5`'s silent all-zero-label fallback. All Domino scripts invoke `python3.11`. See `experiments/domino/README.md`.

6. **Config-driven `input_h5`/`output` (added post-implementation).** `--input_h5` and `--output` were `required=True`, which argparse enforces against the *command line* — so config values injected via `set_defaults` did not satisfy it. Both are now `default=None` with a post-parse `parser.error(...)` check, so `data.input_h5` / `training.output` (and the already-optional `split_file` / `output_root`) can be set entirely in the YAML; CLI flags still override.

7. **Inline dependency install restored in `train_tso_patch_h5.py` (per user request, reverses Task 1 Step 5).** Because the separate `deep_tso_setup.sh` caused install trouble on Domino, the top-level `shell_script` + `subprocess.run(...)` block (matching `training/train_tso_patch.py`: `cd munge/predictive_modeling; sudo python3.11 -m pip install -r requirements-tso.txt; -e .; optuna/.../mamba-ssm`) is back at module top. Trade-off: importing the module (incl. the H5-contract / SupCon-batching pytest modules) now triggers the install — expected on Domino, matches the other trainers.

**Validation scope (decided):** Phase 1 gates model selection on a *label-free* signal (cross-night interval consistency + duration/fragmentation priors), NOT on provable TSO accuracy. The report's **primary** validation — the fixed-scratch-model downstream proxy — and the small expert/PSG gold set remain in **Post-Phase Runway** by design. Task 7 and the summary template state this explicitly so Phase-1 results are read as a robustness/stability gate, not as proof of TSO improvement.

---

## Findings From The Report And Repo

The report proposes four practical changes:

- Use Generalized Cross Entropy (GCE) instead of plain CE for TSO labels that come from noisy traditional algorithms.
- Add cross-night supervised contrastive learning using subject ID as the positive key.
- Treat Sadeh, Cole-Kripke, and van Hees style outputs as multiple annotators, converting agreement into per-minute confidence.
- Validate with downstream scratch proxy, cross-night interval consistency, and a small expert/PSG gold set when available.

The repo already has useful pieces:

- `losses/structural_priors.py` already contains transition-count, duration-prior, ELR, boundary reweighting, and circadian-bias code.
- `training/train_tso_patch_h5.py` already integrates those structural losses.
- `evaluation/postprocessing.py` already has single-TSO post-processing.
- `experiments/configs/tso_structural_priors.yaml` already sketches an ablation matrix.

Important gaps discovered during analysis:

- `models/setup.py` creates `MBA4TSO_Patch`, but the modular `models/resmamba.py` does not define that class. The class still exists in `Helpers/DL_models.py`, so the modular path can fail before training.
- `training/train_tso_patch_h5.py` and `training/convert_h5.py` run package installation in top-level code. That must move into Domino job scripts so imports and tests are deterministic.
- Current H5 files store `X`, `Y`, `seq_lengths`, and `segment_names`; they do not store subject IDs or annotator label tracks.
- The TSO training script is mostly CLI/hard-coded parameters, while the project now prefers YAML configs.

## File Structure

- Modify `models/resmamba.py`: add/port `MBA4TSO_Patch`, import `PatchEmbedding`, add optional projection head and embedding return.
- Modify `models/setup.py`: import and instantiate the modular `MBA4TSO_Patch`.
- Modify `models/__init__.py`: export `MBA4TSO_Patch`.
- Create `losses/noisy_labels.py`: sequence-aware GCE and consensus-weight helpers.
- Modify `losses/__init__.py`: export noisy-label helpers.
- Modify `losses/structural_priors.py`: allow `base_loss="ce"|"gce"` and optional supervision weights.
- Modify `training/convert_h5.py`: remove top-level install side effects; store subject IDs and optional annotator tracks.
- Modify `data/padding.py`: return optional minute-level consensus weights from H5 batches.
- Modify `training/train_tso_patch_h5.py`: add YAML loading, output root configuration, robust loss flags, SupCon hooks, subject IDs, label-free validation metrics, and cleaner checkpoint metadata.
- Create `evaluation/tso_validation.py`: interval extraction, fragmentation, duration, and cross-night consistency metrics.
- Create `experiments/configs/deep_tso_phase1_baseline.yaml`.
- Create `experiments/configs/deep_tso_phase1_gce.yaml`.
- Create `experiments/configs/deep_tso_phase1_gce_supcon.yaml`.
- Create `experiments/configs/deep_tso_phase2_consensus.yaml`.
- Create `experiments/domino/deep_tso_setup.sh`.
- Create `experiments/domino/run_deep_tso_smoke.sh`.
- Create `experiments/domino/run_deep_tso_ablation.sh`.
- Create `tests/test_deep_tso_noisy_labels.py`.
- Create `tests/test_deep_tso_h5_contract.py`.
- Create `tests/test_mba4tso_patch_factory.py`.

## Milestones

1. Stabilize the code path so Domino can import and instantiate the TSO model.
2. Add robust GCE and keep CE as the default baseline.
3. Extend the H5 contract for subject IDs and consensus labels.
4. Add cross-night SupCon without changing inference outputs.
5. Add label-free validation metrics and checkpoint selection.
6. Add Domino setup/run scripts and experiment configs.
7. Run Domino smoke and ablation jobs, then summarize results.

---

### Task 1: Stabilize Model Factory And Domino Setup Boundary

**Files:**
- Modify: `models/resmamba.py`
- Modify: `models/setup.py`
- Modify: `models/__init__.py`
- Modify: `training/train_tso_patch_h5.py`
- Modify: `training/convert_h5.py`
- Create: `experiments/domino/deep_tso_setup.sh`
- Test: `tests/test_mba4tso_patch_factory.py`

- [ ] **Step 1: Write the failing factory test**

Create `tests/test_mba4tso_patch_factory.py`:

```python
import torch


def test_setup_model_creates_mba4tso_patch_and_forward_returns_logits():
    from models.setup import setup_model

    params = {
        "batch_size": 2,
        "num_filters": 16,
        "dropout": 0.1,
        "droppath": 0.1,
        "kernel_f": 3,
        "kernel_MBA": 3,
        "num_feature_layers": 1,
        "blocks_MBA": 1,
        "featurelayer": "ResNet",
        "patch_size": 60,
        "patch_channels": 6,
        "norm1": "BN",
        "norm2": "GN",
        "output_channels": 1,
        "skip_connect": True,
        "skip_cross_attention": False,
    }
    model = setup_model("mba4tso_patch", None, 8, params, pretraining=False, num_classes=1)
    x = torch.randn(2, 8, 60, 6)
    lengths = torch.tensor([8, 6])
    logits = model(x, lengths)
    assert logits.shape == (2, 8, 1)
```

- [ ] **Step 2: Run the failing test on Domino**

Run in a Domino workspace or job, not locally:

```bash
python -m pytest tests/test_mba4tso_patch_factory.py -q
```

Expected before implementation: `NameError: name 'MBA4TSO_Patch' is not defined` or import failure from the modular model path.

- [ ] **Step 3: Port `MBA4TSO_Patch` into `models/resmamba.py`**

Add this import near the existing `models/resmamba.py` imports:

```python
from .specialized import PatchEmbedding
```

Add this class above `MBA4TSO` in `models/resmamba.py`:

```python
class MBA4TSO_Patch(nn.Module):
    """Mamba TSO model for raw per-minute patches."""

    def __init__(
        self,
        patch_size=1200,
        patch_channels=5,
        num_filters=64,
        num_feature_layers=3,
        num_encoder_layers=3,
        drop_path_rate=0.3,
        kernel_size_feature=3,
        kernel_size_mba=7,
        dropout_rate=0.2,
        add_positional_encoding=True,
        max_seq_len=1440,
        featurelayer="ResNet",
        skip_connect=True,
        skip_cross_attention=False,
        norm1="BN",
        norm2="IN",
        output_channels=3,
        projection_dim=128,
    ):
        super().__init__()
        self.num_feature_layers = num_feature_layers
        self.add_positional_encoding = add_positional_encoding
        self.output_channels = output_channels
        self.skip_connect = skip_connect
        self.skip_cross_attention = skip_cross_attention

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            in_channels=patch_channels,
            num_filters=num_filters,
        )

        self.feature_extractor = None
        if num_feature_layers > 0:
            self.feature_extractor = FeatureExtractor(
                tsm_horizon=64,
                in_channels=num_filters,
                pos_embed_dim=16,
                num_filters=num_filters,
                kernel_size=kernel_size_feature,
                num_feature_layers=num_feature_layers,
                tsm=False,
                featurelayer=featurelayer,
                norm=norm1,
            )

        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)

        if self.skip_connect and self.skip_cross_attention:
            self.encoder = nn.ModuleList([
                AttModule_mamba_cross(
                    2 ** i,
                    num_filters,
                    num_filters,
                    1,
                    1,
                    "sliding_att",
                    "encoder",
                    1,
                    drop_path_rate=drop_path_rate,
                    kernel_size=kernel_size_mba,
                    dropout_rate=dropout_rate,
                )
                for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([
                AttModule_mamba(
                    2 ** i,
                    num_filters,
                    num_filters,
                    "sliding_att",
                    "encoder",
                    1,
                    drop_path_rate=drop_path_rate,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size_mba,
                    norm=norm2,
                )
                for i in range(num_encoder_layers)
            ])

        self.output_projection = nn.Conv1d(num_filters, output_channels, kernel_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters, projection_dim),
        )

    def forward(self, x, x_lengths, return_embedding=False):
        batch_size, seq_len, _, _ = x.size()
        if isinstance(x_lengths, torch.Tensor):
            lengths = x_lengths.clone().detach().to(x.device)
        else:
            lengths = torch.tensor(x_lengths, device=x.device)
        mask = create_mask(lengths, seq_len, batch_size, x.device).bool()

        x = self.patch_embedding(x).permute(0, 2, 1)

        feature_maps = []
        if self.feature_extractor is not None:
            if self.skip_connect:
                x, feature_maps = self.feature_extractor(x, mask, return_intermediates=True)
            else:
                x = self.feature_extractor(x, mask)

            if x.size(2) != mask.size(1):
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=x.size(2),
                    mode="nearest",
                ).squeeze(1).bool()

        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        if self.skip_connect and self.num_feature_layers > 0:
            for i, layer in enumerate(self.encoder):
                if self.skip_cross_attention:
                    encoder_states = None
                    if i < len(feature_maps):
                        encoder_states = feature_maps[len(feature_maps) - 1 - i]
                    x = layer(x, encoder_states, mask)
                else:
                    x = layer(x, x, mask)
                    if i < len(feature_maps):
                        skip = feature_maps[len(feature_maps) - 1 - i]
                        if x.size(2) != skip.size(2):
                            skip = F.interpolate(skip, size=x.size(2), mode="linear", align_corners=False)
                        x = x + skip
        else:
            for layer in self.encoder:
                x = layer(x, x, mask)

        output = self.output_projection(x)
        if output.size(2) != seq_len:
            output = F.interpolate(output, size=seq_len, mode="linear", align_corners=False)
        output = output.permute(0, 2, 1)

        # Only compute the pooled night embedding when a caller (cross-night
        # SupCon, Task 4) actually needs it — otherwise it is wasted compute.
        if return_embedding:
            pooled = masked_avg_pool(x.permute(0, 2, 1), mask.float())
            embedding = self.projection_head(pooled)
            return output, embedding
        return output
```

- [ ] **Step 4: Update model imports and exports**

Change the `models/setup.py` import line from:

```python
from .resmamba import MBA_tsm, MBA_tsm_with_padding, MBA_patch, MBA4TSO, MBA_v1, MBA_v1_ForPretraining, latent_mixup, masked_avg_pool, create_mask
```

to:

```python
from .resmamba import MBA_tsm, MBA_tsm_with_padding, MBA_patch, MBA4TSO, MBA4TSO_Patch, MBA_v1, MBA_v1_ForPretraining, latent_mixup, masked_avg_pool, create_mask
```

In `models/__init__.py`, add `MBA4TSO_Patch` to the `from .resmamba import (...)` block and to `__all__`.

Also thread the skip-connection knobs through the factory so configs/tests can set them (the existing `case "mba4tso_patch":` block at `models/setup.py:281-310` silently drops them). Add these two reads after the `output_channels = best_params.get(...)` line:

```python
            skip_connect = best_params.get("skip_connect", True)
            skip_cross_attention = best_params.get("skip_cross_attention", False)
```

And add these two keyword arguments to the `model = MBA4TSO_Patch(...)` call (after `output_channels=output_channels`):

```python
                skip_connect=skip_connect,
                skip_cross_attention=skip_cross_attention,
```

- [ ] **Step 5: Remove top-level package installation side effects**

In both `training/train_tso_patch_h5.py` and `training/convert_h5.py`, delete the top-level block that imports `subprocess`, defines `shell_script`, and calls `subprocess.run(...)`.

Add this comment after the module docstring in each file:

```python
# Domino dependency installation is handled by experiments/domino/deep_tso_setup.sh.
# Keep training modules importable without mutating the environment.
```

- [ ] **Step 6: Create Domino dependency setup script**

Create `experiments/domino/deep_tso_setup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirement-ml.txt
python3.11 -m pip install -r requirements-tso.txt
python3.11 -m pip install -e .
python3.11 -m pip install optuna==4.3.0 seaborn ray TensorboardX torcheval ruptures "mamba-ssm[causal-conv1d]==2.2.2"
```

- [ ] **Step 7: Run the factory test on Domino**

Run:

```bash
bash experiments/domino/deep_tso_setup.sh
python -m pytest tests/test_mba4tso_patch_factory.py -q
```

Expected after implementation: `1 passed`.

- [ ] **Step 8: Commit**

```bash
git add models/resmamba.py models/setup.py models/__init__.py training/train_tso_patch_h5.py training/convert_h5.py experiments/domino/deep_tso_setup.sh tests/test_mba4tso_patch_factory.py
git commit -m "fix: stabilize modular MBA4TSO patch path"
```

---

### Task 2: Add Sequence-Aware GCE And Weighted Supervision

**Files:**
- Create: `losses/noisy_labels.py`
- Modify: `losses/__init__.py`
- Modify: `losses/structural_priors.py`
- Test: `tests/test_deep_tso_noisy_labels.py`

- [ ] **Step 1: Write failing tests for robust sequence loss**

Create `tests/test_deep_tso_noisy_labels.py`:

```python
import torch


def test_gce_binary_ignores_padding_and_returns_scalar():
    from losses.noisy_labels import generalized_cross_entropy_loss

    logits = torch.tensor([[[2.0], [-2.0], [0.0]]])
    labels = torch.tensor([[2, 0, -100]])
    loss = generalized_cross_entropy_loss(logits, labels, q=0.7)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gce_three_class_accepts_per_position_weight():
    from losses.noisy_labels import generalized_cross_entropy_loss

    logits = torch.tensor([[[0.0, 0.0, 4.0], [3.0, 0.0, 0.0]]])
    labels = torch.tensor([[2, 0]])
    weight = torch.tensor([[1.0, 0.25]])
    weighted = generalized_cross_entropy_loss(logits, labels, q=0.7, weight=weight)
    unweighted = generalized_cross_entropy_loss(logits, labels, q=0.7)
    assert torch.isfinite(weighted)
    assert weighted <= unweighted * 4


def test_consensus_confidence_from_annotator_votes():
    from losses.noisy_labels import consensus_from_annotators

    votes = torch.tensor([[[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0]]])
    label, weight = consensus_from_annotators(votes, positive_class=2)
    assert label.tolist() == [[2, 2, 0, 0]]
    # float32 confidences, so compare with tolerance (2/3 is not float32-exact).
    assert torch.allclose(weight, torch.tensor([[1.0, 2 / 3, 2 / 3, 1.0]]), atol=1e-6)
```

- [ ] **Step 2: Run failing tests on Domino**

Run:

```bash
python -m pytest tests/test_deep_tso_noisy_labels.py -q
```

Expected before implementation: `ModuleNotFoundError: No module named 'losses.noisy_labels'`.

- [ ] **Step 3: Add `losses/noisy_labels.py`**

Create `losses/noisy_labels.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def generalized_cross_entropy_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    q: float = 0.7,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """GCE for TSO sequence logits with padding and optional confidence weights."""
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1]; got {q}")

    valid = labels != ignore_index
    num_classes = outputs.shape[-1]

    if num_classes == 1:
        targets = (labels == 2).float()
        prob_pos = torch.sigmoid(outputs[..., 0]).clamp(min=1e-7, max=1.0)
        prob_true = torch.where(targets > 0.5, prob_pos, 1.0 - prob_pos)
    else:
        safe_labels = labels.masked_fill(~valid, 0)
        probs = torch.softmax(outputs, dim=-1).clamp(min=1e-7, max=1.0)
        prob_true = probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    loss = (1.0 - prob_true.pow(q)) / q
    loss = loss.masked_fill(~valid, 0.0)

    if weight is not None:
        w = weight.to(outputs.device).float().masked_fill(~valid, 0.0)
        return (loss * w).sum() / w.sum().clamp(min=1.0)

    return loss.sum() / valid.float().sum().clamp(min=1.0)


def consensus_from_annotators(
    annotator_tso: torch.Tensor,
    *,
    positive_class: int = 2,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert binary annotator TSO votes [B,T,A] to class labels and confidence."""
    if annotator_tso.dim() != 3:
        raise ValueError(f"annotator_tso must have shape [B,T,A]; got {tuple(annotator_tso.shape)}")
    vote_fraction = annotator_tso.float().mean(dim=-1)
    is_tso = vote_fraction >= threshold
    labels = torch.where(
        is_tso,
        torch.full_like(vote_fraction, positive_class, dtype=torch.long),
        torch.zeros_like(vote_fraction, dtype=torch.long),
    )
    confidence = torch.maximum(vote_fraction, 1.0 - vote_fraction)
    return labels, confidence
```

- [ ] **Step 4: Export noisy-label helpers**

In `losses/__init__.py`, add:

```python
from .noisy_labels import generalized_cross_entropy_loss, consensus_from_annotators
```

Add these names to `__all__`:

```python
'generalized_cross_entropy_loss',
'consensus_from_annotators',
```

- [ ] **Step 5: Wire GCE into structural loss**

In `losses/structural_priors.py`, update `measure_loss_tso_structural` signature:

```python
def measure_loss_tso_structural(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    x_lengths: torch.Tensor,
    *,
    patch_duration_hours: float,
    boundary_weight: Optional[torch.Tensor] = None,
    supervision_weight: Optional[torch.Tensor] = None,
    base_loss: str = "ce",
    gce_q: float = 0.7,
    w_trans: float = 0.0,
    w_dur: float = 0.0,
    w_elr: float = 0.0,
    trans_budget: float = 2.0,
    dur_min: float = 3.0,
    dur_max: float = 11.0,
    elr_target: Optional[torch.Tensor] = None,
) -> dict:
```

Replace the CE block at the start of the function with:

```python
    if boundary_weight is not None and supervision_weight is not None:
        weight = boundary_weight.to(outputs.device).float() * supervision_weight.to(outputs.device).float()
    elif boundary_weight is not None:
        weight = boundary_weight.to(outputs.device).float()
    elif supervision_weight is not None:
        weight = supervision_weight.to(outputs.device).float()
    else:
        weight = None

    if base_loss == "gce":
        from .noisy_labels import generalized_cross_entropy_loss
        ce = generalized_cross_entropy_loss(outputs, labels, q=gce_q, weight=weight)
    elif base_loss == "ce":
        if weight is not None:
            ce = boundary_reweighted_ce_loss(outputs, labels, weight)
        else:
            from .standard import measure_loss_tso
            ce = measure_loss_tso(outputs, labels, x_lengths)
    else:
        raise ValueError(f"base_loss must be 'ce' or 'gce'; got {base_loss}")
```

- [ ] **Step 6: Run loss tests on Domino**

Run:

```bash
python -m pytest tests/test_deep_tso_noisy_labels.py -q
```

Expected after implementation: `3 passed`.

- [ ] **Step 7: Commit**

```bash
git add losses/noisy_labels.py losses/__init__.py losses/structural_priors.py tests/test_deep_tso_noisy_labels.py
git commit -m "feat: add robust GCE supervision for TSO"
```

---

### Task 3: Extend H5 Contract For Subjects And Consensus Labels

**Files:**
- Modify: `training/convert_h5.py`
- Modify: `training/train_tso_patch_h5.py`
- Modify: `data/padding.py`
- Test: `tests/test_deep_tso_h5_contract.py`

- [ ] **Step 1: Write H5 contract tests**

Create `tests/test_deep_tso_h5_contract.py`:

```python
import h5py
import numpy as np


def test_h5_dataset_reads_subject_ids_and_falls_back_to_segment_prefix(tmp_path):
    from training.train_tso_patch_h5 import H5Dataset

    path = tmp_path / "mini.h5"
    with h5py.File(path, "w") as h5f:
        h5f.attrs["num_segments"] = 2
        h5f.attrs["max_seq_length"] = 120
        h5f.attrs["samples_per_second"] = 20
        h5f.attrs["max_len"] = 120
        h5f.attrs["num_channels"] = 6
        h5f.create_dataset("X", data=np.zeros((2, 120, 6), dtype=np.float32))
        h5f.create_dataset("Y", data=np.zeros((2, 120, 2), dtype=np.int8))
        h5f.create_dataset("seq_lengths", data=np.array([120, 120], dtype=np.int32))
        h5f.create_dataset("segment_names", data=np.array(["S1_0_day1", "S2_0_day1"], dtype=h5py.string_dtype()))

    ds = H5Dataset(str(path))
    assert ds[0]["subject"] == "S1"
    assert ds[1]["subject"] == "S2"
    ds.close()


def test_h5_dataset_reads_annotator_tracks_when_present(tmp_path):
    from training.train_tso_patch_h5 import H5Dataset

    path = tmp_path / "mini_annotators.h5"
    with h5py.File(path, "w") as h5f:
        h5f.attrs["num_segments"] = 1
        h5f.attrs["max_seq_length"] = 120
        h5f.attrs["samples_per_second"] = 20
        h5f.attrs["max_len"] = 120
        h5f.attrs["num_channels"] = 6
        h5f.attrs["annotator_names"] = np.array(["sadeh", "cole_kripke"], dtype=h5py.string_dtype())
        h5f.create_dataset("X", data=np.zeros((1, 120, 6), dtype=np.float32))
        h5f.create_dataset("Y", data=np.zeros((1, 120, 2), dtype=np.int8))
        h5f.create_dataset("Y_annotators", data=np.ones((1, 120, 2), dtype=np.int8))
        h5f.create_dataset("subject_ids", data=np.array(["S1"], dtype=h5py.string_dtype()))
        h5f.create_dataset("seq_lengths", data=np.array([120], dtype=np.int32))
        h5f.create_dataset("segment_names", data=np.array(["S1_0_day1"], dtype=h5py.string_dtype()))

    ds = H5Dataset(str(path))
    sample = ds[0]
    assert sample["Y_annotators"].shape == (120, 2)
    assert sample["subject"] == "S1"
    ds.close()
```

- [ ] **Step 2: Run failing H5 tests on Domino**

Run:

```bash
python -m pytest tests/test_deep_tso_h5_contract.py -q
```

Expected before implementation: missing `subject` and `Y_annotators` keys.

- [ ] **Step 3: Store subject IDs in H5 conversion**

In `training/convert_h5.py`, add `subject` to the dictionary returned by `load_and_preprocess_segment`:

```python
            'subject': current_subject,
```

In `convert_parquet_to_h5`, after creating `ds_segments`, create:

```python
        ds_subjects = h5f.create_dataset(
            "subject_ids",
            shape=(num_segments,),
            dtype=dt,
        )
```

After `ds_segments[idx] = segment_data['segment']`, add:

```python
            ds_subjects[idx] = segment_data["subject"]
```

If failed files require trimming, add (next to the existing `ds_segments.resize((actual_segments,))`):

```python
            ds_subjects.resize((actual_segments,))
```

While editing the trimming block, fix a latent channel-count bug: the existing line hardcodes 5 channels, which silently drops the `time_cos` channel from every segment whenever any file fails to load on a 6-channel (`use_sincos=True`) build. Change:

```python
            ds_X.resize((actual_segments, max_len, 5))
```

to:

```python
            ds_X.resize((actual_segments, max_len, num_channels))
```

- [ ] **Step 4: Add optional annotator column arguments to H5 conversion**

In `training/convert_h5.py`, change `load_and_preprocess_segment` signature to:

```python
def load_and_preprocess_segment(
    file,
    scaler,
    max_samples,
    samples_per_second=20,
    use_sincos=True,
    annotator_columns=None,
):
```

Inside `load_and_preprocess_segment`, before `return result`, add:

```python
        annotator_columns = annotator_columns or []
        annotator_tracks = []
        for col in annotator_columns:
            if col not in df.columns:
                raise ValueError(f"Annotator column {col!r} not found in {file}")
            annotator_tracks.append(df[col].values.astype(np.int8))
        if annotator_tracks:
            result["annotators"] = np.stack(annotator_tracks, axis=1)
```

Change `convert_parquet_to_h5` signature to include:

```python
                          annotator_columns=None,
```

Before processing files, add:

```python
    annotator_columns = annotator_columns or []
```

Inside the H5 file setup, after `ds_subjects` is created in Step 3 (this matters: the block below uses `dt = h5py.string_dtype(...)`, which is only defined just before `ds_segments`/`ds_subjects` — inserting it right after `ds_Y` would reference `dt` before it exists and raise `NameError`), add:

```python
        ds_Y_annotators = None
        if annotator_columns:
            h5f.attrs["annotator_names"] = np.array(annotator_columns, dtype=dt)
            ds_Y_annotators = h5f.create_dataset(
                "Y_annotators",
                shape=(num_segments, max_len, len(annotator_columns)),
                dtype=np.int8,
                chunks=(1, min(1200, max_len), len(annotator_columns)),
                compression="gzip",
                compression_opts=4,
            )
```

Change the loader call:

```python
            segment_data = load_and_preprocess_segment(
                file,
                scaler,
                max_samples,
                samples_per_second,
                use_sincos=use_sincos,
                annotator_columns=annotator_columns,
            )
```

After writing `ds_Y[idx]`, add:

```python
            if ds_Y_annotators is not None:
                annotators = segment_data["annotators"]
                if seg_len < max_len:
                    padding = np.zeros((max_len - seg_len, len(annotator_columns)), dtype=np.int8)
                    annotators = np.vstack([annotators, padding])
                ds_Y_annotators[idx] = annotators
```

If trimming failed files, add:

```python
            if ds_Y_annotators is not None:
                ds_Y_annotators.resize((actual_segments, max_len, len(annotator_columns)))
```

In the CLI parser, add:

```python
    parser.add_argument(
        "--annotator_columns",
        type=str,
        default="",
        help="Comma-separated binary TSO label columns to store as Y_annotators.",
    )
```

Pass this to conversion:

```python
        annotator_columns=[c.strip() for c in args.annotator_columns.split(",") if c.strip()],
```

- [ ] **Step 5: Read subject IDs and annotators in `H5Dataset`**

In `training/train_tso_patch_h5.py`, add this helper above `H5Dataset`:

```python
def _decode_h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _subject_from_segment_name(segment_name):
    text = _decode_h5_string(segment_name)
    return text.split("_")[0] if "_" in text else text
```

In `H5Dataset.__init__`, add:

```python
        self.subject_ids = self.h5f["subject_ids"][:] if "subject_ids" in self.h5f else None
        self.Y_annotators = self.h5f["Y_annotators"] if "Y_annotators" in self.h5f else None
```

In `H5Dataset.__getitem__`, build and return subject and annotators:

```python
        segment_name = self.segment_names[actual_idx]
        subject = (
            _decode_h5_string(self.subject_ids[actual_idx])
            if self.subject_ids is not None
            else _subject_from_segment_name(segment_name)
        )
        sample = {
            "X": self.X[actual_idx],
            "Y": self.Y[actual_idx],
            "seq_length": self.seq_lengths[actual_idx],
            "segment": segment_name,
            "segment_id": int(actual_idx),
            "subject": subject,
        }
        if self.Y_annotators is not None:
            sample["Y_annotators"] = self.Y_annotators[actual_idx]
        return sample
```

- [ ] **Step 6: Add optional consensus batch return**

In `data/padding.py`, update `add_padding_tso_patch_h5` to collect annotators:

```python
    annotator_batch = []
    has_annotators = False
```

Inside the sample loop, after `Y_batch.append(sample['Y'])`, add:

```python
        if "Y_annotators" in sample:
            has_annotators = True
            annotator_batch.append(sample["Y_annotators"])
```

After `pad_Y` initialization, add:

```python
    pad_Y_annotators = None
    if has_annotators:
        annotator_batch = np.stack(annotator_batch)
        num_annotators = annotator_batch.shape[-1]
        pad_Y_annotators = np.zeros((batch_size, num_minutes_max, num_annotators), dtype=np.int8)
```

Inside the minute loop, after setting `pad_Y[i, m]`, add:

```python
                if pad_Y_annotators is not None:
                    annotator_minute = annotator_batch[i, m * patch_size:(m + 1) * patch_size, :]
                    pad_Y_annotators[i, m, :] = np.any(annotator_minute, axis=0).astype(np.int8)
```

Before returning, convert annotators:

```python
    if pad_Y_annotators is not None:
        pad_Y_annotators = torch.from_numpy(pad_Y_annotators).to(device)
```

Return five values:

```python
    return pad_X, pad_Y, x_lens, segments_batch, pad_Y_annotators
```

Update the caller in `run_model_tso_h5` (`training/train_tso_patch_h5.py`) immediately so the script stays runnable. Change:

```python
        pad_X, pad_Y, x_lens, batch_samples = add_padding_tso_patch_h5(
```

to:

```python
        pad_X, pad_Y, x_lens, batch_samples, pad_Y_annotators = add_padding_tso_patch_h5(
```

`add_padding_tso_patch_h5` has a SECOND live consumer: `training/train_tso_dlrtc.py` (canonical, imports it `from data`) unpacks 4 values at three call sites (~lines 659, 806, 1013). Changing the arity to 5 breaks them with `ValueError: too many values to unpack`. The DLRTC trainer does not use consensus, so update each of those three sites to discard the new value:

```python
        pad_X, pad_Y, x_lens, batch_samples, _ = add_padding_tso_patch_h5(
```

(The legacy `legacy/predict_TSO_segment_patch_*.py` and `NS_production/` consumers import their own unchanged 4-value copy from `Helpers/DL_helpers.py`, so they are unaffected.) Add `training/train_tso_dlrtc.py` to this task's commit.

- [ ] **Step 7: Run H5 contract tests on Domino**

Run:

```bash
python -m pytest tests/test_deep_tso_h5_contract.py -q
```

Expected after implementation: `2 passed`.

- [ ] **Step 8: Commit**

```bash
git add training/convert_h5.py training/train_tso_patch_h5.py training/train_tso_dlrtc.py data/padding.py tests/test_deep_tso_h5_contract.py
git commit -m "feat: extend TSO H5 contract for subjects and consensus labels"
```

---

### Task 4: Add Cross-Night SupCon Training Hook

**Files:**
- Modify: `training/train_tso_patch_h5.py`
- Modify: `models/resmamba.py`
- Modify: `data/padding.py` (propagate `subject_index` into `batch_samples`)
- Test: `tests/test_mba4tso_patch_factory.py`
- Test: `tests/test_deep_tso_supcon_batching.py`

- [ ] **Step 1: Extend factory test for embedding return**

Append to `tests/test_mba4tso_patch_factory.py`:

```python
def test_mba4tso_patch_can_return_projection_embedding():
    from models.setup import setup_model

    params = {
        "batch_size": 2,
        "num_filters": 16,
        "dropout": 0.1,
        "droppath": 0.1,
        "kernel_f": 3,
        "kernel_MBA": 3,
        "num_feature_layers": 1,
        "blocks_MBA": 1,
        "featurelayer": "ResNet",
        "patch_size": 60,
        "patch_channels": 6,
        "projection_dim": 12,
        "norm1": "BN",
        "norm2": "GN",
        "output_channels": 1,
        "skip_connect": True,
        "skip_cross_attention": False,
    }
    model = setup_model("mba4tso_patch", None, 8, params, pretraining=False, num_classes=1)
    logits, embedding = model(torch.randn(2, 8, 60, 6), torch.tensor([8, 8]), return_embedding=True)
    assert logits.shape == (2, 8, 1)
    assert embedding.shape == (2, 12)
```

- [ ] **Step 2: Pass projection dimension through setup**

In `models/setup.py`, inside the `case "mba4tso_patch":` block, add:

```python
            projection_dim = best_params.get("projection_dim", 128)
```

Pass it to `MBA4TSO_Patch`:

```python
                projection_dim=projection_dim,
```

- [ ] **Step 3: Add subject integer encoding to `H5Dataset`**

In `H5Dataset.__init__`, after loading `segment_names`, add:

```python
        subjects = []
        for idx in self.indices:
            if self.subject_ids is not None:
                subjects.append(_decode_h5_string(self.subject_ids[idx]))
            else:
                subjects.append(_subject_from_segment_name(self.segment_names[idx]))
        self.subject_to_index = {subject: i for i, subject in enumerate(sorted(set(subjects)))}
        # Per-position subject string (aligned with __getitem__ ordering) so the
        # subject-grouped batch generator can group nights without loading X.
        self.subjects = subjects
```

In `H5Dataset.__getitem__`, add to `sample`:

```python
            "subject_index": self.subject_to_index[subject],
```

- [ ] **Step 4: Add SupCon CLI flags**

In `training/train_tso_patch_h5.py`, add parser arguments (the `--base_loss`/`--gce_q` flags live here, not in Task 5, so Step 6 below can reference `args.base_loss`/`args.gce_q` within this same task):

```python
parser.add_argument("--base_loss", type=str, default="ce", choices=["ce", "gce"],
                    help="Base supervised loss for TSO.")
parser.add_argument("--gce_q", type=float, default=0.7,
                    help="GCE q parameter when --base_loss=gce.")
parser.add_argument("--w_supcon", type=float, default=0.0,
                    help="Weight for cross-night supervised contrastive loss.")
parser.add_argument("--supcon_temperature", type=float, default=0.07,
                    help="Temperature for SupConLossV2.")
parser.add_argument("--projection_dim", type=int, default=128,
                    help="Projection dimension for the TSO night embedding head.")
```

Add to `best_params`:

```python
    "projection_dim": args.projection_dim,
```

- [ ] **Step 5: Add subject-grouped batching and compute SupCon loss in `run_model_tso_h5`**

**(5a) Write the failing batch-generator test.** Create `tests/test_deep_tso_supcon_batching.py`:

```python
from collections import Counter


def test_subject_grouped_batches_keep_same_subject_nights_together():
    from training.train_tso_patch_h5 import subject_grouped_batch_generator

    class _Stub:
        def __init__(self, subjects):
            self.subjects = subjects

        def __len__(self):
            return len(self.subjects)

    subjects = ["S1", "S1", "S1", "S2", "S2", "S3", "S3", "S3", "S3"]
    ds = _Stub(subjects)

    seen = []
    positive_batches = 0
    total_batches = 0
    for batch in subject_grouped_batch_generator(ds, batch_size=4, nights_per_group=4):
        positions = batch.tolist()
        seen.extend(positions)
        counts = Counter(subjects[p] for p in positions)
        total_batches += 1
        if any(c >= 2 for c in counts.values()):
            positive_batches += 1

    assert sorted(seen) == list(range(len(subjects)))  # every night emitted exactly once
    assert positive_batches == total_batches            # every batch has SupCon positives
```

Run it (expected fail: `ImportError: cannot import name 'subject_grouped_batch_generator'`):

```bash
python -m pytest tests/test_deep_tso_supcon_batching.py -q
```

**(5b) Add the generator** next to `batch_generator_h5` in `training/train_tso_patch_h5.py`:

```python
def subject_grouped_batch_generator(dataset, batch_size, nights_per_group=4):
    """Yield batches that keep same-subject nights together so the cross-night
    SupCon objective has positive pairs. Each night is emitted once per epoch.

    Plain random batching almost never co-locates multiple nights of one
    subject, so SupCon would otherwise see zero positives and contribute nothing.
    """
    by_subject = defaultdict(list)
    for pos in range(len(dataset)):
        by_subject[dataset.subjects[pos]].append(pos)

    groups = []
    for positions in by_subject.values():
        positions = list(positions)
        np.random.shuffle(positions)
        for start in range(0, len(positions), nights_per_group):
            groups.append(positions[start:start + nights_per_group])
    np.random.shuffle(groups)

    batch = []
    for group in groups:
        if batch and len(batch) + len(group) > batch_size:
            yield np.array(batch, dtype=np.int64)
            batch = []
        batch.extend(group)
    if batch:
        yield np.array(batch, dtype=np.int64)
```

Add `from collections import defaultdict` to the module imports if it is not already present.

**(5c) Propagate subject info into the batch.** `add_padding_tso_patch_h5` builds its own `segments_batch` dicts (only `segment`/`segment_id` today), so the training loop cannot read `subject_index` without this. In `data/padding.py`, replace the `segments_batch.append({...})` block inside the sample-collection loop with:

```python
        segments_batch.append({
            'segment': sample['segment'],
            'segment_id': sample.get('segment_id', int(idx)),
            'subject': sample.get('subject'),
            'subject_index': sample.get('subject_index'),
        })
```

**(5d) Import `SupConLossV2`** in `training/train_tso_patch_h5.py` (extend the existing `from losses import (...)` block):

```python
from losses import (
    measure_loss_tso,
    measure_loss_tso_with_continuity,
    measure_loss_tso_structural,
    SupConLossV2,
    ELRMemory,
    CircadianPriorBias,
    hour_from_time_channels,
    compute_boundary_weights,
)
```

**(5e) Add parameters to `run_model_tso_h5`.** Insert these immediately before the existing `patch_duration_hours=None):` line (do NOT add `supervision_weight` here — Task 6 computes it per-batch as a local):

```python
                    base_loss="ce", gce_q=0.7,
                    w_supcon=0.0, supcon_temperature=0.07,
```

**(5f) Use the subject-grouped generator only when SupCon is active.** Replace the loop header:

```python
    for batch_indices in batch_generator_h5(dataset, batch_size=batch_size, shuffle=train_mode):
```

with:

```python
    if w_supcon > 0 and train_mode:
        batch_iter = subject_grouped_batch_generator(dataset, batch_size)
    else:
        batch_iter = batch_generator_h5(dataset, batch_size=batch_size, shuffle=train_mode)
    for batch_indices in batch_iter:
```

**(5g) Replace the forward call** `outputs = model(pad_X, x_lens)` with:

```python
        if w_supcon > 0 and train_mode:
            outputs, embedding = model(pad_X, x_lens, return_embedding=True)
        else:
            outputs = model(pad_X, x_lens)
            embedding = None
```

**(5h) Fix loss routing so GCE and consensus weighting actually take effect.** First, initialize the consensus weight as a local right after the boundary-weight precompute block (Task 6 will replace this single line with the real consensus computation):

```python
        supervision_weight = None
```

Then replace the `use_structural` condition:

```python
        use_structural = (
            w_trans > 0 or w_dur > 0 or w_elr > 0 or boundary_weight is not None
        )
```

with:

```python
        use_structural = (
            base_loss != "ce"
            or supervision_weight is not None
            or w_trans > 0 or w_dur > 0 or w_elr > 0
            or boundary_weight is not None
        )
```

and thread the robust-loss/consensus args into the structural call by replacing:

```python
            loss_dict = measure_loss_tso_structural(
                outputs, pad_Y, x_lens,
                patch_duration_hours=patch_duration_hours,
                boundary_weight=boundary_weight,
                w_trans=w_trans, w_dur=w_dur, w_elr=w_elr,
                trans_budget=trans_budget, dur_min=dur_min, dur_max=dur_max,
                elr_target=elr_target,
            )
```

with:

```python
            loss_dict = measure_loss_tso_structural(
                outputs, pad_Y, x_lens,
                patch_duration_hours=patch_duration_hours,
                boundary_weight=boundary_weight,
                supervision_weight=supervision_weight,
                base_loss=base_loss, gce_q=gce_q,
                w_trans=w_trans, w_dur=w_dur, w_elr=w_elr,
                trans_budget=trans_budget, dur_min=dur_min, dur_max=dur_max,
                elr_target=elr_target,
            )
```

**(5i) Add the SupCon term with a zero-positive guard**, after `total_loss` is computed and before the backward pass:

```python
        if w_supcon > 0 and train_mode and embedding is not None:
            subject_indices = torch.tensor(
                [s["subject_index"] for s in batch_samples],
                dtype=torch.long,
                device=device,
            )
            # SupCon needs at least one subject with >= 2 nights in the batch;
            # otherwise it has no positive pairs and is undefined/unstable.
            _, subject_counts = torch.unique(subject_indices, return_counts=True)
            if (subject_counts >= 2).any():
                supcon_loss = SupConLossV2(temperature=supcon_temperature)(embedding, subject_indices)
                total_loss = total_loss + w_supcon * supcon_loss
```

Run the generator test (expected pass):

```bash
python -m pytest tests/test_deep_tso_supcon_batching.py -q
```

- [ ] **Step 6: Pass SupCon flags in training only**

In the train call to `run_model_tso_h5`, pass:

```python
                base_loss=args.base_loss,
                gce_q=args.gce_q,
                w_supcon=args.w_supcon,
                supcon_temperature=args.supcon_temperature,
```

In validation and test calls, pass:

```python
                base_loss=args.base_loss,
                gce_q=args.gce_q,
                w_supcon=0.0,
                supcon_temperature=args.supcon_temperature,
```

- [ ] **Step 7: Run tests on Domino**

Run:

```bash
python -m pytest tests/test_mba4tso_patch_factory.py tests/test_deep_tso_supcon_batching.py tests/test_deep_tso_h5_contract.py -q
```

Expected after implementation: all tests pass.

- [ ] **Step 8: Commit**

(Task 4 does NOT modify `models/resmamba.py` — `MBA4TSO_Patch` already gained `projection_dim`/`return_embedding` in Task 1. The edited files are `models/setup.py` (projection_dim wiring), `training/train_tso_patch_h5.py` (CLI flags, generator, SupCon, loss routing), `data/padding.py` (subject_index propagation), and the two test files.)

```bash
git add models/setup.py training/train_tso_patch_h5.py data/padding.py tests/test_mba4tso_patch_factory.py tests/test_deep_tso_supcon_batching.py
git commit -m "feat: add cross-night SupCon hook for Deep TSO"
```

---

### Task 5: Add YAML Config Loading And Robust Loss Flags

**Files:**
- Modify: `training/train_tso_patch_h5.py`
- Create: `experiments/configs/deep_tso_phase1_baseline.yaml`
- Create: `experiments/configs/deep_tso_phase1_gce.yaml`
- Create: `experiments/configs/deep_tso_phase1_gce_supcon.yaml`
- Create: `experiments/configs/deep_tso_phase2_consensus.yaml`

- [ ] **Step 1: Add YAML parser support**

At the top of `training/train_tso_patch_h5.py`, add:

```python
import yaml
```

Add parser arg before `args = parser.parse_args()`:

```python
parser.add_argument("--config", type=str, default="", help="Path to YAML config file.")
parser.add_argument("--output_root", type=str, default="/mnt/data/GENEActive-featurized/results/DL",
                    help="Root folder for Domino training outputs.")
```

(`--base_loss` and `--gce_q` were already added in Task 4 Step 4.)

Replace `args = parser.parse_args()` with:

```python
def _flatten_config(config):
    flat = {}
    for section in ("data", "model", "training", "loss", "evaluation"):
        values = config.get(section, {})
        if isinstance(values, dict):
            flat.update(values)
    loss_components = config.get("loss", {}).get("components", {})
    if isinstance(loss_components, dict):
        flat.update(loss_components)
    return flat


def _apply_config_defaults(parser, argv):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    known, _ = pre_parser.parse_known_args(argv)
    if not known.config:
        return
    with open(known.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    flat = _flatten_config(config)
    defaults = {}
    mapping = {
        "input_h5": "input_h5",
        "split_file": "split_file",
        "output": "output",
        "output_root": "output_root",
        "architecture": "model",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr": "finetune_lr",
        "base_loss": "base_loss",
        "gce_q": "gce_q",
        "w_supcon": "w_supcon",
        "supcon_temperature": "supcon_temperature",
        "use_consensus_weight": "use_consensus_weight",
        "projection_dim": "projection_dim",
        "w_trans": "w_trans",
        "w_dur": "w_dur",
        "w_elr": "w_elr",
        "boundary_tau_steps": "boundary_tau_steps",
        "enforce_single_tso": "enforce_single_tso",
    }
    for key, arg_name in mapping.items():
        if key in flat:
            defaults[arg_name] = flat[key]
    parser.set_defaults(**defaults)


_apply_config_defaults(parser, sys.argv[1:])
args = parser.parse_args()
```

- [ ] **Step 2: Use configurable output root**

Replace:

```python
results_folder = f"/mnt/data/GENEActive-featurized/results/DL/{args.output}"
```

with:

```python
results_folder = os.path.join(args.output_root, args.output)
```

- [ ] **Step 3: Use YAML batch size**

Add parser arg:

```python
parser.add_argument("--batch_size", type=int, default=24, help="Batch size.")
```

Change `best_params['batch_size']` to:

```python
    'batch_size': args.batch_size,
```

- [ ] **Step 4: Add phase 1 baseline config**

Create `experiments/configs/deep_tso_phase1_baseline.yaml`:

```yaml
experiment:
  name: deep_tso_phase1_baseline
data:
  input_h5: /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5
  split_file: /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_split.npz
model:
  architecture: mba4tso_patch
  projection_dim: 128
training:
  output: deep_tso_phase1_baseline
  output_root: /mnt/data/GENEActive-featurized/results/DL
  batch_size: 24
  epochs: 60
  lr: 0.001
loss:
  base_loss: ce
  components:
    w_trans: 0.0
    w_dur: 0.0
    w_elr: 0.0
    w_supcon: 0.0
evaluation:
  enforce_single_tso: true
```

- [ ] **Step 5: Add phase 1 GCE config**

Create `experiments/configs/deep_tso_phase1_gce.yaml`:

```yaml
experiment:
  name: deep_tso_phase1_gce
data:
  input_h5: /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5
  split_file: /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_split.npz
model:
  architecture: mba4tso_patch
  projection_dim: 128
training:
  output: deep_tso_phase1_gce
  output_root: /mnt/data/GENEActive-featurized/results/DL
  batch_size: 24
  epochs: 60
  lr: 0.001
loss:
  base_loss: gce
  gce_q: 0.7
  # Structural priors stay OFF so this arm isolates GCE vs the CE baseline.
  components:
    w_trans: 0.0
    w_dur: 0.0
    w_elr: 0.0
    w_supcon: 0.0
evaluation:
  enforce_single_tso: true
```

- [ ] **Step 6: Add GCE + SupCon config**

Create `experiments/configs/deep_tso_phase1_gce_supcon.yaml`:

```yaml
experiment:
  name: deep_tso_phase1_gce_supcon
data:
  input_h5: /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5
  split_file: /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_split.npz
model:
  architecture: mba4tso_patch
  projection_dim: 128
training:
  output: deep_tso_phase1_gce_supcon
  output_root: /mnt/data/GENEActive-featurized/results/DL
  batch_size: 24
  epochs: 60
  lr: 0.001
loss:
  base_loss: gce
  gce_q: 0.7
  # Structural priors OFF: this arm isolates the SupCon term on top of GCE.
  components:
    w_trans: 0.0
    w_dur: 0.0
    w_elr: 0.0
    w_supcon: 0.1
  supcon_temperature: 0.07
evaluation:
  enforce_single_tso: true
```

- [ ] **Step 7: Add phase 2 consensus config**

Create `experiments/configs/deep_tso_phase2_consensus.yaml`:

```yaml
experiment:
  name: deep_tso_phase2_consensus
data:
  input_h5: /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_consensus.h5
  split_file: /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_consensus_split.npz
model:
  architecture: mba4tso_patch
  projection_dim: 128
training:
  output: deep_tso_phase2_consensus
  output_root: /mnt/data/GENEActive-featurized/results/DL
  batch_size: 24
  epochs: 60
  lr: 0.001
loss:
  base_loss: gce
  gce_q: 0.7
  use_consensus_weight: true
  # Structural priors OFF: this arm isolates annotator-consensus weighting
  # on top of GCE + SupCon.
  components:
    w_trans: 0.0
    w_dur: 0.0
    w_elr: 0.0
    w_supcon: 0.1
  supcon_temperature: 0.07
evaluation:
  enforce_single_tso: true
```

- [ ] **Step 8: Commit**

```bash
git add training/train_tso_patch_h5.py experiments/configs/deep_tso_phase1_baseline.yaml experiments/configs/deep_tso_phase1_gce.yaml experiments/configs/deep_tso_phase1_gce_supcon.yaml experiments/configs/deep_tso_phase2_consensus.yaml
git commit -m "feat: add Deep TSO YAML experiment configs"
```

---

### Task 6: Use Consensus Weights During Training

**Files:**
- Modify: `training/train_tso_patch_h5.py`
- Modify: `data/padding.py`
- Test: `tests/test_deep_tso_noisy_labels.py`

- [ ] **Step 1: Add parser flag**

In `training/train_tso_patch_h5.py`, add:

```python
parser.add_argument("--use_consensus_weight", action="store_true",
                    help="Use Y_annotators agreement as per-minute supervision confidence.")
```

- [ ] **Step 2: Confirm annotators are unpacked from padding**

This was already done in Task 3 Step 6 — the call in `run_model_tso_h5` reads:

```python
        pad_X, pad_Y, x_lens, batch_samples, pad_Y_annotators = add_padding_tso_patch_h5(
```

No change needed here; just verify it is present before continuing.

- [ ] **Step 3: Build consensus labels and weights (preserving the padding mask)**

Add `use_consensus_weight=False` to the `run_model_tso_h5` parameter list, and pass it from the train/val/test calls.

Import:

```python
from losses import consensus_from_annotators
```

Task 4 Step 5h added a placeholder `supervision_weight = None` right after the boundary-weight precompute. **Replace that single line** with the consensus block below. The ordering matters: `consensus_from_annotators` emits only `{0, 2}` (never `-100`), so it must not overwrite padded or non-wear minutes, and the weight mask must be taken from the *original* padding mask (the earlier plan computed it from the already-overwritten `pad_Y`, which silently re-enabled training on padding):

```python
        supervision_weight = None
        if use_consensus_weight:
            if pad_Y_annotators is None:
                raise ValueError("--use_consensus_weight requires Y_annotators in the H5 file")
            # Capture masks BEFORE relabeling — consensus_labels has no -100.
            padding_mask = pad_Y == -100
            nonwear_mask = pad_Y == 1
            consensus_labels, supervision_weight = consensus_from_annotators(
                pad_Y_annotators, positive_class=2
            )
            keep_original = padding_mask | nonwear_mask
            pad_Y = torch.where(keep_original, pad_Y, consensus_labels)
            # Zero supervision on padded minutes using the ORIGINAL mask so GCE
            # never trains on padding (pad_Y no longer contains -100 after the
            # torch.where above).
            supervision_weight = supervision_weight.masked_fill(padding_mask, 0.0)
```

- [ ] **Step 4: Verify supervision weights reach the structural loss**

No new edit is required: Task 4 Step 5h already threads `supervision_weight=supervision_weight, base_loss=base_loss, gce_q=gce_q` into the `measure_loss_tso_structural(...)` call, and the routing condition already includes `supervision_weight is not None`. The local `supervision_weight` populated in Step 3 therefore flows through automatically. Confirm both are present.

- [ ] **Step 5: Commit**

```bash
git add training/train_tso_patch_h5.py data/padding.py tests/test_deep_tso_noisy_labels.py
git commit -m "feat: use annotator consensus confidence for TSO training"
```

---

### Task 7: Add Label-Free TSO Validation Metrics

**Files:**
- Create: `evaluation/tso_validation.py`
- Modify: `evaluation/__init__.py`
- Modify: `training/train_tso_patch_h5.py`
- Test: `tests/test_deep_tso_validation.py`

- [ ] **Step 1: Write validation tests**

Create `tests/test_deep_tso_validation.py`:

```python
import numpy as np


def test_extract_tso_interval_binary_sequence():
    from evaluation.tso_validation import extract_tso_interval

    result = extract_tso_interval(np.array([0, 0, 1, 1, 1, 0]), timestep_minutes=1.0)
    assert result["onset_minute"] == 2
    assert result["offset_minute"] == 5
    assert result["duration_hours"] == 3 / 60
    assert result["segment_count"] == 1


def test_cross_night_consistency_groups_by_subject():
    from evaluation.tso_validation import cross_night_consistency

    intervals = [
        {"subject": "S1", "onset_minute": 100, "offset_minute": 500, "duration_hours": 6.67, "segment_count": 1},
        {"subject": "S1", "onset_minute": 110, "offset_minute": 505, "duration_hours": 6.58, "segment_count": 1},
        {"subject": "S2", "onset_minute": 200, "offset_minute": 600, "duration_hours": 6.67, "segment_count": 1},
    ]
    metrics = cross_night_consistency(intervals)
    assert metrics["subjects_with_multiple_nights"] == 1
    assert metrics["mean_onset_std_minutes"] == 5.0
```

- [ ] **Step 2: Add validation module**

Create `evaluation/tso_validation.py`:

```python
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    start = None
    for i, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def extract_tso_interval(pred_classes: np.ndarray, *, timestep_minutes: float = 1.0) -> dict:
    tso_class = 1 if np.max(pred_classes) <= 1 else 2
    segments = _segments(pred_classes == tso_class)
    if not segments:
        return {
            "onset_minute": np.nan,
            "offset_minute": np.nan,
            "duration_hours": 0.0,
            "segment_count": 0,
        }
    longest = max(segments, key=lambda pair: pair[1] - pair[0])
    duration_steps = longest[1] - longest[0]
    return {
        "onset_minute": float(longest[0] * timestep_minutes),
        "offset_minute": float(longest[1] * timestep_minutes),
        "duration_hours": float(duration_steps * timestep_minutes / 60.0),
        "segment_count": len(segments),
    }


def cross_night_consistency(intervals: list[dict]) -> dict:
    by_subject = defaultdict(list)
    for item in intervals:
        by_subject[item["subject"]].append(item)

    onset_stds = []
    offset_stds = []
    duration_stds = []
    for subject_intervals in by_subject.values():
        if len(subject_intervals) < 2:
            continue
        onset_stds.append(float(np.nanstd([x["onset_minute"] for x in subject_intervals])))
        offset_stds.append(float(np.nanstd([x["offset_minute"] for x in subject_intervals])))
        duration_stds.append(float(np.nanstd([x["duration_hours"] for x in subject_intervals])))

    return {
        "subjects_with_multiple_nights": len(onset_stds),
        "mean_onset_std_minutes": float(np.nanmean(onset_stds)) if onset_stds else np.nan,
        "mean_offset_std_minutes": float(np.nanmean(offset_stds)) if offset_stds else np.nan,
        "mean_duration_std_hours": float(np.nanmean(duration_stds)) if duration_stds else np.nan,
    }
```

- [ ] **Step 3: Export validation helpers**

In `evaluation/__init__.py`, add:

```python
from .tso_validation import extract_tso_interval, cross_night_consistency
```

Add both names to `__all__` if the file defines `__all__`.

- [ ] **Step 4: Record interval metrics in TSO evaluation**

In `training/train_tso_patch_h5.py`, import:

```python
from evaluation.tso_validation import extract_tso_interval, cross_night_consistency
```

Inside `run_model_tso_h5`, initialize:

```python
    interval_records = []
```

Inside the per-sample collection loop, after `valid_labels`, add:

```python
            if num_out_channels == 1:
                pred_seq = (valid_preds[:, 0] > 0.5).astype(int)
            else:
                pred_seq = np.argmax(valid_preds, axis=1)
            interval = extract_tso_interval(pred_seq, timestep_minutes=1.0)
            interval["segment"] = _decode_h5_string(batch_samples[i]["segment"])
            interval["subject"] = batch_samples[i]["subject"]
            interval_records.append(interval)
```

Before returning, add:

```python
    if interval_records:
        consistency = cross_night_consistency(interval_records)
        metrics.update({
            "mean_pred_tso_duration_hours": float(np.nanmean([x["duration_hours"] for x in interval_records])),
            "mean_pred_tso_segment_count": float(np.nanmean([x["segment_count"] for x in interval_records])),
            **consistency,
        })
    predictions["interval_records"] = interval_records
```

- [ ] **Step 5: Add selection score**

> **Scope caveat (label clearly):** This score still leans on `val_metrics["loss"]`, which is computed against the *noisy* traditional-algorithm labels. It is therefore a **robustness/structural-plausibility gate**, NOT proof of TSO accuracy. The report's primary validation — the fixed-scratch-model downstream proxy — and the small expert/PSG gold set are deferred to Post-Phase Runway. Treat the `mean_pred_tso_*` and cross-night-consistency terms as the genuinely label-free signal; the penalty weights below are deliberate placeholders to be revisited once the gold set exists.

After `val_metrics` is computed, add:

```python
                # NOTE: selection_score is a label-free-leaning robustness gate,
                # not a TSO-accuracy metric — val loss is still vs. noisy labels.
                selection_score = (
                    val_metrics["loss"]
                    + 0.05 * val_metrics.get("mean_pred_tso_segment_count", 0.0)
                    + 0.05 * abs(val_metrics.get("mean_pred_tso_duration_hours", 7.0) - 7.0)
                )
                val_metrics["selection_score"] = selection_score
```

Use `selection_score` for early stopping:

```python
                early_stopping(selection_score, model)
```

Add `val_selection_score` to the initial `history` dict:

```python
        'val_selection_score': [],
```

After appending validation metrics, add:

```python
                history["val_selection_score"].append(selection_score)
```

Save `val_selection_score` in the checkpoint.

- [ ] **Step 6: Run validation tests on Domino**

Run:

```bash
python -m pytest tests/test_deep_tso_validation.py -q
```

Expected after implementation: `2 passed`.

- [ ] **Step 7: Commit**

```bash
git add evaluation/tso_validation.py evaluation/__init__.py training/train_tso_patch_h5.py tests/test_deep_tso_validation.py
git commit -m "feat: add label-free TSO validation metrics"
```

---

### Task 8: Add Domino Smoke And Ablation Scripts

**Files:**
- Create: `experiments/domino/run_deep_tso_smoke.sh`
- Create: `experiments/domino/run_deep_tso_ablation.sh`
- Modify: `docs/deployment.md`

- [ ] **Step 1: Create smoke script**

Create `experiments/domino/run_deep_tso_smoke.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_H5:?Set INPUT_H5 to the Domino H5 path}"
: "${OUTPUT_ROOT:=/mnt/data/GENEActive-featurized/results/DL}"

python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso_phase1_gce_supcon.yaml \
  --input_h5 "${INPUT_H5}" \
  --output "deep_tso_smoke_${DOMINO_RUN_ID:-manual}" \
  --output_root "${OUTPUT_ROOT}" \
  --epochs 2 \
  --batch_size 4 \
  --val_size 0.05 \
  --num_gpu 0
```

- [ ] **Step 2: Create ablation script**

Create `experiments/domino/run_deep_tso_ablation.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_H5:?Set INPUT_H5 to the Domino H5 path}"
: "${SPLIT_FILE:=}"
: "${OUTPUT_ROOT:=/mnt/data/GENEActive-featurized/results/DL}"

configs=(
  experiments/configs/deep_tso_phase1_baseline.yaml
  experiments/configs/deep_tso_phase1_gce.yaml
  experiments/configs/deep_tso_phase1_gce_supcon.yaml
)

for config in "${configs[@]}"; do
  name="$(basename "${config}" .yaml)"
  cmd=(
    python3.11 training/train_tso_patch_h5.py
    --config "${config}"
    --input_h5 "${INPUT_H5}"
    --output "${name}_${DOMINO_RUN_ID:-manual}"
    --output_root "${OUTPUT_ROOT}"
    --num_gpu 0
  )
  if [[ -n "${SPLIT_FILE}" ]]; then
    cmd+=(--split_file "${SPLIT_FILE}")
  fi
  "${cmd[@]}"
done
```

- [ ] **Step 3: Document Domino-only execution**

Append to `docs/deployment.md`:

```markdown
## Deep TSO Noisy-Label Domino Jobs

Full Deep TSO experiments are intended to run on Domino because the local machine does not have the required training data and GPU/runtime stack.

Smoke test:

```bash
export INPUT_H5=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5
export OUTPUT_ROOT=/mnt/data/GENEActive-featurized/results/DL
bash experiments/domino/deep_tso_setup.sh
bash experiments/domino/run_deep_tso_smoke.sh
```

Ablation:

```bash
export INPUT_H5=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5
export SPLIT_FILE=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_split.npz
export OUTPUT_ROOT=/mnt/data/GENEActive-featurized/results/DL
bash experiments/domino/deep_tso_setup.sh
bash experiments/domino/run_deep_tso_ablation.sh
```

Use the smoke job after every code change. Use the ablation job after the smoke job passes.
```

- [ ] **Step 4: Make scripts executable**

Run:

```bash
chmod +x experiments/domino/deep_tso_setup.sh experiments/domino/run_deep_tso_smoke.sh experiments/domino/run_deep_tso_ablation.sh
```

- [ ] **Step 5: Commit**

```bash
git add experiments/domino/deep_tso_setup.sh experiments/domino/run_deep_tso_smoke.sh experiments/domino/run_deep_tso_ablation.sh docs/deployment.md
git commit -m "chore: add Domino Deep TSO run scripts"
```

---

### Task 9: Run Domino Verification Matrix

**Files:**
- No source edits unless a failure is found.
- Read: Domino job logs.
- Read: `experiments/logs/` or `/mnt/data/GENEActive-featurized/results/DL/*`.

- [ ] **Step 1: Run unit tests on Domino**

Run:

```bash
bash experiments/domino/deep_tso_setup.sh
python -m pytest \
  tests/test_mba4tso_patch_factory.py \
  tests/test_deep_tso_noisy_labels.py \
  tests/test_deep_tso_h5_contract.py \
  tests/test_deep_tso_supcon_batching.py \
  tests/test_deep_tso_validation.py \
  -q
```

(`test_deep_tso_noisy_labels.py` and `test_deep_tso_validation.py` already pass locally; the other three need the Domino `mamba_ssm` stack.)

Expected: all tests pass.

- [ ] **Step 2: Run smoke training**

Run:

```bash
export INPUT_H5=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5
export OUTPUT_ROOT=/mnt/data/GENEActive-featurized/results/DL
bash experiments/domino/run_deep_tso_smoke.sh
```

Expected:

```text
Epoch 1/2
Epoch 2/2
Training complete!
```

Expected artifacts:

```text
/mnt/data/GENEActive-featurized/results/DL/deep_tso_smoke_${DOMINO_RUN_ID:-manual}/training/model_weights/best_model_iter_0.pt
/mnt/data/GENEActive-featurized/results/DL/deep_tso_smoke_${DOMINO_RUN_ID:-manual}/training/predictions/results_iter_0.joblib
```

- [ ] **Step 3: Run phase 1 ablation**

Run:

```bash
export INPUT_H5=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5
export SPLIT_FILE=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_split.npz
export OUTPUT_ROOT=/mnt/data/GENEActive-featurized/results/DL
bash experiments/domino/run_deep_tso_ablation.sh
```

Expected: baseline, GCE, and GCE+SupCon jobs each finish and save `results_iter_0.joblib`.

- [ ] **Step 4: Decide phase 2 consensus run**

Run this only after an H5 file with `Y_annotators` is created:

```bash
python training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso_phase2_consensus.yaml \
  --input_h5 /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_consensus.h5 \
  --split_file /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_consensus_split.npz \
  --use_consensus_weight \
  --num_gpu 0
```

Expected: no `ValueError` about missing `Y_annotators`.

- [ ] **Step 5: Summarize results**

Run this from the repo root after the ablation jobs complete:

```bash
python - <<'PY'
from pathlib import Path
import os
import subprocess

import joblib

output_root = Path(os.environ.get("OUTPUT_ROOT", "/mnt/data/GENEActive-featurized/results/DL"))
run_id = os.environ.get("DOMINO_RUN_ID", "manual")
input_h5 = os.environ.get("INPUT_H5", "/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5")
split_file = os.environ.get("SPLIT_FILE", "/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_split.npz")
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

runs = [
    ("baseline", "experiments/configs/deep_tso_phase1_baseline.yaml", f"deep_tso_phase1_baseline_{run_id}"),
    ("gce", "experiments/configs/deep_tso_phase1_gce.yaml", f"deep_tso_phase1_gce_{run_id}"),
    ("gce_supcon", "experiments/configs/deep_tso_phase1_gce_supcon.yaml", f"deep_tso_phase1_gce_supcon_{run_id}"),
]

rows = []
for label, config, run_name in runs:
    result_path = output_root / run_name / "training" / "predictions" / "results_iter_0.joblib"
    data = joblib.load(result_path)
    history = data["history"]
    scores = history.get("val_selection_score", history["val_loss"])
    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    test_metrics = data["test_metrics"]
    rows.append([
        label,
        config,
        str(best_idx + 1),
        f"{scores[best_idx]:.6f}",
        f"{history['val_f1_tso'][best_idx]:.6f}",
        f"{test_metrics.get('mean_pred_tso_segment_count', float('nan')):.6f}",
        f"{test_metrics.get('mean_pred_tso_duration_hours', float('nan')):.6f}",
        str(result_path),
    ])

lines = [
    "# Deep TSO Phase 1 Domino Summary",
    "",
    "## Runs",
    "",
    "| Run | Config | Best Epoch | Selection Score | Val F1 TSO | Mean Segment Count | Mean Duration Hours | Result Path |",
    "|---|---|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    lines.append("| " + " | ".join(row) + " |")
lines.extend([
    "",
    "## Decision",
    "",
    "Proceed to phase 2 consensus if GCE or GCE+SupCon improves selection score while keeping mean duration in the 3-11 hour band and mean segment count near 1.",
    "",
    "**Caveat:** selection score still includes noisy-label val loss, so these are robustness/structural signals, not provable TSO-accuracy gains. The downstream scratch proxy + gold set (Post-Phase Runway) remain required before claiming TSO improvement.",
    "",
    "## Notes",
    "",
    f"- Input H5: `{input_h5}`",
    f"- Split file: `{split_file}`",
    f"- Commit hash: `{commit_hash}`",
])

out = Path("experiments/logs/deep_tso_phase1_summary.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY
```

- [ ] **Step 6: Commit results summary**

```bash
git add experiments/logs/deep_tso_phase1_summary.md
git commit -m "docs: summarize Deep TSO phase 1 Domino results"
```

---

## Post-Phase Runway

After the Phase 1/2 Domino matrix identifies a stable training recipe:

- Add the downstream scratch proxy as a separate plan because it touches the Deep Scratch inference pipeline and requires a fixed scratch checkpoint. **This is the report's *primary* validation signal (§6) — until it lands, Phase 1/2 results are robustness/consistency evidence only, not proof that Deep TSO beats the traditional labeler.** It also unlocks the small expert/PSG gold set used for final model selection.
- Add a direct interval/CRF head as a separate plan because it replaces the output contract, checkpoint compatibility, and post-processing assumptions.
- Add TS-TCC or TF-C pretraining as a separate plan because it is a separate training stage with different data sampling.

This keeps the first implementation focused on the report's low-risk "NOW" and "NEXT" items while preserving a clear path to the higher-risk architecture changes.

## Self-Review

- Spec coverage: The plan covers robust GCE, multi-annotator consensus, cross-night SupCon, single-interval structural metrics, validation changes, and Domino-only execution.
- Placeholder scan: The implementation steps use concrete files, commands, and code snippets. Domino paths are explicit defaults and can be overridden through environment variables.
- Type consistency: `Y_annotators` is `[segment, sample, annotator]` in H5 and `[batch, minute, annotator]` after padding. `subject` is a string, and `subject_index` is the integer label for SupCon.
- Execution risk: The first task fixes model import/runtime side effects before training changes, so failures are isolated early.

### Post-review verification (2026-06-09)

Every "current state" claim was checked against the live code, and the fixes were verified against real signatures:

- **GCE routing.** `run_model_tso_h5` selects the loss at `training/train_tso_patch_h5.py:403-405` (`use_structural` gate → `measure_loss_tso_structural`, else continuity, else plain CE). The revised gate now triggers on `base_loss != "ce"` / `supervision_weight is not None`, so `base_loss="gce"` can no longer silently fall back to CE, and the Phase-1 configs isolate one variable each (CE → GCE → +SupCon → +consensus, all structural weights `0`).
- **SupCon positives.** Batching is plain shuffle (`batch_generator_h5:131-150`); the new `subject_grouped_batch_generator` keeps same-subject nights intact per batch, gated on `w_supcon > 0 and train_mode`, with a `>= 2 nights` guard before invoking `SupConLossV2`. `subject_index` now flows through `add_padding_tso_patch_h5`'s `segments_batch` dicts (previously only `segment`/`segment_id`, which would have `KeyError`-ed).
- **Padding integrity.** `pad_Y` is `-100` for padded minutes (`data/padding.py:604`) and `measure_loss_tso` ignores `-100` (`losses/standard.py:344,365`), so the plain GCE path is safe. The consensus path now captures `padding_mask`/`nonwear_mask` before `torch.where` and masks the weight from the original `padding_mask`, fixing the earlier no-op that trained on padding.
- **Signatures confirmed:** `PatchEmbedding` is in `models/specialized.py:26` (the plan's import is correct); `FeatureExtractor.forward(..., return_intermediates=True)`, `create_mask`, `masked_avg_pool`, `PositionalEncoding`, `AttModule_mamba(_cross)`, and `SupConLossV2.forward(features, labels)` all match the code the plan generates. `losses/__init__.py` already exports every name the training script imports.
- **Anchor fixes:** `models/setup.py:281-310` now also receives `skip_connect`/`skip_cross_attention`; `convert_h5.py:390` channel-count bug (`5` → `num_channels`) is corrected; `data/padding.py` edits were re-verified to match real variable names.
- **Validation scope (explicit):** Phase-1 selection is labeled throughout as a robustness/consistency gate (noisy val loss still dominates `selection_score`); the primary downstream scratch proxy and gold set remain in Post-Phase Runway by design.
- **Known cross-task ordering:** Task 4 Step 6 wires `args.base_loss`/`args.gce_q`/`args.w_supcon` into the train/val/test calls; those flags are now all defined within Task 4 Step 4, so the script is import-clean after each task. The unit-test gates (factory / H5 contract / noisy-labels / supcon batching / validation) never execute `main()`, so they pass per-task; full end-to-end execution is validated by the Task 9 Domino smoke run.
