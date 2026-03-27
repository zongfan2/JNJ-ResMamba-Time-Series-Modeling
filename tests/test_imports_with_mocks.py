#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import test with mocking of heavy dependencies.

This test can check module imports by mocking torch, mamba_ssm, transformers, etc.
allowing us to verify the module structure without requiring CUDA or large packages.
"""

import sys
import os
from unittest import mock
import importlib

PROJECT_ROOT = '/sessions/bold-dreamy-dijkstra/mnt/JNJ'
sys.path.insert(0, PROJECT_ROOT)

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def setup_mocks():
    """Set up mocks for heavy dependencies."""

    # Mock torch and related modules
    sys.modules['torch'] = mock.MagicMock()
    sys.modules['torch.nn'] = mock.MagicMock()
    sys.modules['torch.nn.functional'] = mock.MagicMock()
    sys.modules['torch.nn.utils'] = mock.MagicMock()
    sys.modules['torch.nn.utils.weight_norm'] = mock.MagicMock()
    sys.modules['torch.nn.parallel'] = mock.MagicMock()
    sys.modules['torch.optim'] = mock.MagicMock()
    sys.modules['torch.optim.lr_scheduler'] = mock.MagicMock()
    sys.modules['torch.utils'] = mock.MagicMock()
    sys.modules['torch.utils.data'] = mock.MagicMock()
    sys.modules['torch.nn.utils.rnn'] = mock.MagicMock()

    # Mock mamba_ssm
    sys.modules['mamba_ssm'] = mock.MagicMock()
    sys.modules['causal_conv1d'] = mock.MagicMock()

    # Mock transformers
    sys.modules['transformers'] = mock.MagicMock()

    # Mock other heavy dependencies
    sys.modules['matplotlib'] = mock.MagicMock()
    sys.modules['matplotlib.pyplot'] = mock.MagicMock()
    sys.modules['matplotlib.patches'] = mock.MagicMock()
    sys.modules['seaborn'] = mock.MagicMock()
    sys.modules['datasets'] = mock.MagicMock()
    sys.modules['PIL'] = mock.MagicMock()
    sys.modules['PIL.Image'] = mock.MagicMock()
    sys.modules['linear_attention_transformer'] = mock.MagicMock()
    sys.modules['einops'] = mock.MagicMock()
    sys.modules['torchvision'] = mock.MagicMock()
    sys.modules['torchvision.ops'] = mock.MagicMock()


def test_module_import(module_name: str) -> tuple:
    """Test module import."""
    try:
        importlib.import_module(module_name)
        return True, f"✓ {module_name}"
    except Exception as e:
        return False, f"✗ {module_name}: {type(e).__name__}: {str(e)[:100]}"


def main():
    """Run tests with mocks."""

    print(f"\n{'='*70}")
    print("MODULE IMPORT TEST (WITH MOCKS)")
    print(f"{'='*70}\n")

    # Set up mocks for heavy dependencies
    setup_mocks()

    modules_to_test = [
        # Models
        'models',
        'models.components',
        'models.attention',
        'models.mamba_blocks',
        'models.resmamba',
        'models.encoder_decoder',
        'models.pretraining',
        'models.baselines',
        'models.resnet',
        'models.unet',
        'models.specialized',
        'models.conv1d',
        'models.setup',
        # Data
        'data',
        'data.loading',
        'data.preprocessing',
        'data.padding',
        'data.augmentation',
        'data.batching',
        # Losses
        'losses',
        'losses.standard',
        'losses.dlrtc',
        # Evaluation
        'evaluation',
        'evaluation.metrics',
        'evaluation.postprocessing',
        # Utils
        'utils',
        'utils.common',
    ]

    print(f"{YELLOW}Testing module imports with mocks...{RESET}\n")

    results = []
    for module_name in modules_to_test:
        success, message = test_module_import(module_name)
        results.append((success, message))
        color = GREEN if success else RED
        print(f"{color}{message}{RESET}")

    # Test key classes
    print(f"\n{YELLOW}Testing key class imports...{RESET}\n")

    class_imports = [
        ('from models import MBA_tsm_with_padding', 'models.MBA_tsm_with_padding'),
        ('from models import setup_model', 'models.setup_model'),
        ('from losses import FocalLoss', 'losses.FocalLoss'),
        ('from losses import SupConLoss', 'losses.SupConLoss'),
        ('from data import load_data', 'data.load_data'),
        ('from evaluation import calculate_metrics_nn', 'evaluation.calculate_metrics_nn'),
    ]

    class_results = []
    for import_stmt, check_name in class_imports:
        try:
            namespace = {}
            exec(import_stmt, namespace)
            class_results.append((True, f"✓ {import_stmt}"))
            print(f"{GREEN}✓ {import_stmt}{RESET}")
        except Exception as e:
            class_results.append((False, f"✗ {import_stmt}: {str(e)[:80]}"))
            print(f"{RED}✗ {import_stmt}: {type(e).__name__}{RESET}")

    # Summary
    passed = sum(1 for success, _ in results if success)
    total = len(results)
    class_passed = sum(1 for success, _ in class_results if success)
    class_total = len(class_results)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nModule imports:     {passed}/{total} passed")
    print(f"Class imports:      {class_passed}/{class_total} passed")

    failures = [msg for success, msg in results if not success] + \
               [msg for success, msg in class_results if not success]

    if failures:
        print(f"\n{RED}Failures:{RESET}")
        for failure in failures:
            print(f"  {RED}{failure}{RESET}")
    else:
        print(f"\n{GREEN}✓ All imports successful!{RESET}")

    print(f"\n{'='*70}\n")

    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
