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
