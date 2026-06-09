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
