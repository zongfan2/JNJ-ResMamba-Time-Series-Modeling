#!/bin/bash
# -*- coding: utf-8 -*-
#
# Pipeline Equivalence Test Runner
#
# This script runs the comprehensive pipeline equivalence test suite that verifies
# the OLD pipeline (Helpers/DL_models.py and Helpers/DL_helpers.py) produces
# IDENTICAL output to the NEW modular pipeline (models/, data/, losses/, etc.)
#
# Usage:
#   ./run_equivalence_test.sh          # Run with pytest if available, else unittest
#   ./run_equivalence_test.sh -v       # Verbose output
#   ./run_equivalence_test.sh -m        # Run with unittest instead of pytest
#

set -e  # Exit on error

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Pipeline Equivalence Test Suite"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Test file: tests/test_pipeline_equivalence.py"
echo ""

# Try pytest first if available, fall back to unittest
if command -v pytest &> /dev/null && [[ "$1" != "-m" ]]; then
    echo "Running with pytest..."
    python -m pytest tests/test_pipeline_equivalence.py -v --tb=short --color=yes
    exit_code=$?
else
    echo "Running with unittest..."
    python -m unittest tests.test_pipeline_equivalence -v
    exit_code=$?
fi

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "All tests completed successfully!"
else
    echo "Some tests failed or encountered issues."
fi
echo "=========================================="

exit $exit_code
