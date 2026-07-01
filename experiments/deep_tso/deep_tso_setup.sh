#!/usr/bin/env bash
set -euo pipefail

python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirement-ml.txt
python3.11 -m pip install -r requirement-tso.txt
python3.11 -m pip install -e .
python3.11 -m pip install optuna==4.3.0 seaborn ray TensorboardX torcheval ruptures "mamba-ssm[causal-conv1d]==2.2.2"
