#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirement-ml.txt
python -m pip install -r requirements-tso.txt
python -m pip install -e .
python -m pip install optuna==4.3.0 seaborn ray TensorboardX torcheval ruptures "mamba-ssm[causal-conv1d]==2.2.2"
