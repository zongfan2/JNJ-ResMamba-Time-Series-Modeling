"""Tests for the structured single-interval output (C4): boundary helper,
pointer-style interval loss, and the decode that guarantees offset >= onset.

These exercise the pure-torch pieces of the interval head; the full model
forward requires mamba_ssm and is verified on the training server.
"""
import torch
import torch.nn as nn

from losses.structural_priors import (
    tso_interval_bounds,
    interval_boundary_loss,
    decode_interval,
)


def test_interval_bounds_longest_run_and_padding():
    labels = torch.tensor([
        [0, 0, 2, 2, 2, 2, 0, 0, 2, 0],          # longest TSO run = minutes 2..5
        [0, 0, 0, 0, 0, 0, 0, 0, -100, -100],    # no TSO
    ])
    x_lengths = torch.tensor([10, 8])
    onset, offset, has = tso_interval_bounds(labels, x_lengths, positive_class=2)
    assert onset.tolist() == [2, -1]
    assert offset.tolist() == [5, -1]
    assert has.tolist() == [True, False]


def test_interval_bounds_picks_longest_of_multiple_runs():
    labels = torch.tensor([[2, 2, 0, 2, 2, 2, 0, 0, 0, 0]])  # runs [0..1] (len2) and [3..5] (len3)
    onset, offset, has = tso_interval_bounds(labels, torch.tensor([10]), positive_class=2)
    assert (onset.item(), offset.item()) == (3, 5)
    assert bool(has.item())


def test_interval_boundary_loss_finite_and_differentiable():
    labels = torch.tensor([
        [0, 0, 2, 2, 2, 2, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -100, -100],
    ])
    x_lengths = torch.tensor([10, 8])
    B, T, C = 2, 10, 8
    torch.manual_seed(0)
    hidden = torch.randn(B, C, T, requires_grad=True)
    onset_logits = nn.Conv1d(C, 1, 1)(hidden).squeeze(1)
    offset_logits = nn.Conv1d(C, 1, 1)(hidden).squeeze(1)
    minute_mask = torch.tensor([[1] * 10, [1] * 8 + [0] * 2]).bool()
    onset_logits = onset_logits.masked_fill(~minute_mask, -1e4)
    offset_logits = offset_logits.masked_fill(~minute_mask, -1e4)

    loss = interval_boundary_loss(onset_logits, offset_logits, labels, x_lengths)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(hidden.grad).all()


def test_interval_boundary_loss_zero_when_no_tso():
    labels = torch.zeros(2, 10, dtype=torch.long)  # no positive class anywhere
    logits = torch.randn(2, 10)
    loss = interval_boundary_loss(logits, logits, labels, torch.tensor([10, 10]))
    assert loss.item() == 0.0


def test_decode_interval_enforces_offset_after_onset():
    onset_logits = torch.tensor([[0., 0, 0, 5, 0, 0, 0, 0, 0, 0]])   # onset -> 3
    offset_logits = torch.tensor([[9., 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # raw peak at 0 (< onset)
    onset, offset = decode_interval(onset_logits, offset_logits)
    assert onset.item() == 3
    assert offset.item() >= onset.item()
