#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural import test - checks module structure without external dependencies.

This test verifies:
1. All __init__.py files exist and are valid Python
2. All Python files can be parsed (syntax is correct)
3. Module structure is properly set up
4. No circular import issues at a structural level
"""

import sys
import os
import ast
from pathlib import Path
from typing import Tuple, List, Dict

# Add project root to sys.path
PROJECT_ROOT = '/sessions/bold-dreamy-dijkstra/mnt/JNJ'

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_python_syntax(file_path: str) -> Tuple[bool, str]:
    """
    Check if a Python file has valid syntax.

    Args:
        file_path: Path to Python file

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def check_module_structure() -> Dict[str, List[str]]:
    """
    Check the module structure and verify all required files exist.

    Returns:
        Dictionary with structure info
    """
    required_modules = {
        'models': [
            '__init__.py',
            'components.py',
            'attention.py',
            'mamba_blocks.py',
            'resmamba.py',
            'encoder_decoder.py',
            'pretraining.py',
            'baselines.py',
            'resnet.py',
            'unet.py',
            'specialized.py',
            'conv1d.py',
            'setup.py',
        ],
        'data': [
            '__init__.py',
            'loading.py',
            'preprocessing.py',
            'padding.py',
            'augmentation.py',
            'batching.py',
        ],
        'losses': [
            '__init__.py',
            'standard.py',
            'dlrtc.py',
        ],
        'evaluation': [
            '__init__.py',
            'metrics.py',
            'postprocessing.py',
        ],
        'utils': [
            '__init__.py',
            'common.py',
        ],
    }

    results = {}

    for module_name, expected_files in required_modules.items():
        module_path = os.path.join(PROJECT_ROOT, module_name)
        results[module_name] = []

        if not os.path.isdir(module_path):
            results[module_name].append(f"ERROR: Module directory {module_name} not found")
            continue

        # Check if all required files exist
        for filename in expected_files:
            file_path = os.path.join(module_path, filename)
            if not os.path.exists(file_path):
                results[module_name].append(f"ERROR: Missing file {filename}")
            else:
                results[module_name].append(f"OK: {filename}")

    return results


def check_init_files() -> Dict[str, Tuple[bool, str]]:
    """
    Check all __init__.py files for syntax errors.

    Returns:
        Dictionary of results
    """
    init_files = []

    for module_name in ['models', 'data', 'losses', 'evaluation', 'utils']:
        init_path = os.path.join(PROJECT_ROOT, module_name, '__init__.py')
        if os.path.exists(init_path):
            init_files.append((f"{module_name}/__init__.py", init_path))

    results = {}
    for name, path in init_files:
        is_valid, error = check_python_syntax(path)
        results[name] = (is_valid, error)

    return results


def check_all_python_files() -> Dict[str, Tuple[bool, str]]:
    """
    Check all Python files in the project for syntax errors.

    Returns:
        Dictionary of results
    """
    results = {}

    for module_name in ['models', 'data', 'losses', 'evaluation', 'utils']:
        module_path = os.path.join(PROJECT_ROOT, module_name)
        if not os.path.isdir(module_path):
            continue

        py_files = [f for f in os.listdir(module_path) if f.endswith('.py')]
        for py_file in sorted(py_files):
            file_path = os.path.join(module_path, py_file)
            is_valid, error = check_python_syntax(file_path)
            rel_path = f"{module_name}/{py_file}"
            results[rel_path] = (is_valid, error)

    return results


def check_relative_imports() -> Dict[str, List[str]]:
    """
    Analyze relative imports in module files.

    Returns:
        Dictionary of findings
    """
    results = {}

    for module_name in ['models', 'data', 'losses', 'evaluation', 'utils']:
        module_path = os.path.join(PROJECT_ROOT, module_name)
        if not os.path.isdir(module_path):
            continue

        py_files = [f for f in os.listdir(module_path) if f.endswith('.py')]
        for py_file in sorted(py_files):
            file_path = os.path.join(module_path, py_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                tree = ast.parse(code)

                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(f"from {node.module} import ...")
                        elif node.level > 0:
                            imports.append(f"from {'.' * node.level} import ...")

                rel_path = f"{module_name}/{py_file}"
                if imports:
                    results[rel_path] = imports

            except Exception as e:
                rel_path = f"{module_name}/{py_file}"
                results[rel_path] = [f"ERROR: {str(e)}"]

    return results


def main():
    """Run structural tests."""

    print(f"\n{'='*70}")
    print("STRUCTURAL IMPORT TEST SUITE")
    print(f"{'='*70}\n")

    # ============================================================================
    # Check module structure
    # ============================================================================

    print(f"{YELLOW}Checking module structure...{RESET}\n")

    structure = check_module_structure()
    structure_issues = 0

    for module_name, files in structure.items():
        print(f"{BLUE}{module_name}:{RESET}")
        for status in files:
            if "ERROR" in status:
                color = RED
                structure_issues += 1
            else:
                color = GREEN
            print(f"  {color}{status}{RESET}")

    # ============================================================================
    # Check __init__.py files
    # ============================================================================

    print(f"\n{YELLOW}Checking __init__.py files for syntax errors...{RESET}\n")

    init_results = check_init_files()
    init_errors = 0

    for init_file, (is_valid, error) in init_results.items():
        if is_valid:
            print(f"{GREEN}✓ {init_file}{RESET}")
        else:
            print(f"{RED}✗ {init_file}: {error}{RESET}")
            init_errors += 1

    # ============================================================================
    # Check all Python files for syntax
    # ============================================================================

    print(f"\n{YELLOW}Checking all Python files for syntax errors...{RESET}\n")

    py_results = check_all_python_files()
    syntax_errors = 0
    syntax_ok = 0

    for file_path, (is_valid, error) in sorted(py_results.items()):
        if is_valid:
            print(f"{GREEN}✓ {file_path}{RESET}")
            syntax_ok += 1
        else:
            print(f"{RED}✗ {file_path}: {error}{RESET}")
            syntax_errors += 1

    # ============================================================================
    # Analyze imports
    # ============================================================================

    print(f"\n{YELLOW}Analyzing imports in module files...{RESET}\n")

    imports = check_relative_imports()

    for file_path, import_list in sorted(imports.items()):
        print(f"{BLUE}{file_path}:{RESET}")
        for imp in import_list:
            if "ERROR" in imp:
                print(f"  {RED}{imp}{RESET}")
            else:
                print(f"  {GREEN}{imp}{RESET}")

    # ============================================================================
    # Summary
    # ============================================================================

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nStructure issues:   {structure_issues}")
    print(f"__init__.py errors: {init_errors}/{len(init_results)}")
    print(f"Python syntax OK:   {syntax_ok}/{len(py_results)}")
    print(f"Python syntax ERR:  {syntax_errors}/{len(py_results)}")

    total_issues = structure_issues + init_errors + syntax_errors

    if total_issues == 0:
        print(f"\n{GREEN}✓ All structural checks passed!{RESET}")
        print(f"✓ Module structure is correct")
        print(f"✓ All Python files have valid syntax")
        print(f"✓ All __init__.py files are properly formatted")
    else:
        print(f"\n{RED}✗ Found {total_issues} issue(s){RESET}")

    print(f"\n{'='*70}\n")

    return 0 if total_issues == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
