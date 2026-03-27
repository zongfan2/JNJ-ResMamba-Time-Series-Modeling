#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify all module imports work correctly.

This script tests:
1. All submodules can be imported
2. Key classes are importable from their respective modules
3. Reports any import errors and their root causes
"""

import sys
import importlib
import traceback
from typing import Tuple, List

# Add project root to sys.path
PROJECT_ROOT = '/sessions/bold-dreamy-dijkstra/mnt/JNJ'
sys.path.insert(0, PROJECT_ROOT)

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def test_module_import(module_name: str) -> Tuple[bool, str]:
    """
    Test if a module can be imported.

    Args:
        module_name: Full module name (e.g., 'models.components')

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        importlib.import_module(module_name)
        return True, f"✓ {module_name}"
    except ImportError as e:
        return False, f"✗ {module_name}: ImportError: {str(e)}"
    except Exception as e:
        return False, f"✗ {module_name}: {type(e).__name__}: {str(e)}"


def test_class_import(import_statement: str) -> Tuple[bool, str]:
    """
    Test if a specific class can be imported.

    Args:
        import_statement: Python import statement (e.g., 'from models import MBA_tsm_with_padding')

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        exec(import_statement, {})
        return True, f"✓ {import_statement}"
    except ImportError as e:
        return False, f"✗ {import_statement}: ImportError: {str(e)}"
    except Exception as e:
        return False, f"✗ {import_statement}: {type(e).__name__}: {str(e)}"


def main():
    """Run all import tests."""

    print(f"\n{'='*70}")
    print("MODULE IMPORT TEST SUITE")
    print(f"{'='*70}\n")

    # ============================================================================
    # Test all submodules
    # ============================================================================

    modules_to_test = [
        # Models submodules
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
        # Data submodules
        'data',
        'data.loading',
        'data.preprocessing',
        'data.padding',
        'data.augmentation',
        'data.batching',
        # Losses submodules
        'losses',
        'losses.standard',
        'losses.dlrtc',
        # Evaluation submodules
        'evaluation',
        'evaluation.metrics',
        'evaluation.postprocessing',
        # Utils
        'utils',
        'utils.common',
    ]

    print(f"{YELLOW}Testing module imports...{RESET}\n")

    module_results = []
    for module_name in modules_to_test:
        success, message = test_module_import(module_name)
        module_results.append((success, message))
        color = GREEN if success else RED
        print(f"{color}{message}{RESET}")

    # ============================================================================
    # Test key class imports
    # ============================================================================

    class_imports = [
        # Models
        'from models import MBA_tsm_with_padding',
        'from models import MBA_tsm',
        'from models import setup_model',
        'from models import EfficientUNet',
        'from models import PatchTST',
        'from models import FeatureExtractor',
        'from models import AttModule_mamba',
        # Data
        'from data import load_data',
        'from data import load_data_from_h5',
        'from data import add_padding',
        'from data import batch_generator',
        'from data import augment_iteration',
        # Losses
        'from losses import FocalLoss',
        'from losses import SupConLoss',
        'from losses import measure_loss',
        'from losses import measure_loss_multitask_with_padding',
        # Evaluation
        'from evaluation import metrics',
        # Utils
        'from utils import common',
    ]

    print(f"\n{YELLOW}Testing class/function imports...{RESET}\n")

    class_results = []
    for import_stmt in class_imports:
        success, message = test_class_import(import_stmt)
        class_results.append((success, message))
        color = GREEN if success else RED
        print(f"{color}{message}{RESET}")

    # ============================================================================
    # Summary
    # ============================================================================

    module_passed = sum(1 for success, _ in module_results if success)
    module_total = len(module_results)

    class_passed = sum(1 for success, _ in class_results if success)
    class_total = len(class_results)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nModule imports:     {module_passed}/{module_total} passed")
    print(f"Class imports:      {class_passed}/{class_total} passed")
    print(f"Overall:            {module_passed + class_passed}/{module_total + class_total} passed")

    # ============================================================================
    # Report failures in detail
    # ============================================================================

    failures = [(msg, 'module') for success, msg in module_results if not success] + \
               [(msg, 'class') for success, msg in class_results if not success]

    if failures:
        print(f"\n{RED}{'='*70}")
        print("FAILURES")
        print(f"{'='*70}{RESET}")

        for message, import_type in failures:
            print(f"\n{RED}{message}{RESET}")
    else:
        print(f"\n{GREEN}All imports successful!{RESET}")

    print(f"\n{'='*70}\n")

    # Return non-zero exit code if there are failures
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
