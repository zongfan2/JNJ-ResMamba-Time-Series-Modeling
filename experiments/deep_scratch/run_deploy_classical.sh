#!/usr/bin/env bash
# run_deploy_classical.sh
# Deployment retrain launcher for the classical scratch-detection baselines.
#
# Trains one (or all) classical baseline(s) on the FULL NOPROD cohort — no
# held-out fold — and writes a production-ready joblib bundle that drops
# straight into NS_production/model/.  This is the artifact your manager
# wants for deployment.
#
# Performance is NOT measured for the all-data retrain.  Report the LOSO
# CV numbers from the matching ablation config (Table 8 in the paper) as
# the expected production performance.
#
# Usage (from Domino):
#   bash <path>/experiments/deep_scratch/run_deploy_classical.sh <arch> [extra args]
#
# Architectures:
#   mahadevan2021  — RandomForest (50 trees) + 26-of-36 RFECV features
#   ji2023         — LightGBM + 36 hand-crafted features (DL feats off)
#   mdpi2024_fe    — sklearn GBM + tsfresh comprehensive features
#   mdpi2024_cnn   — ConvNormPool 1D CNN on raw 3-s windows
#   all            — train all four sequentially (longest first)
#
# Extra CLI args after the arch are forwarded to train_classical.py, e.g.:
#   bash run_deploy_classical.sh mdpi2024_cnn --num_gpu 0
#
# Notes:
#   - mdpi2024_cnn is the slowest; mdpi2024_fe is second-slowest (tsfresh).
#   - All four configs set ``training.train_all: true``; the runner picks
#     up the flag from YAML, but you can also force it on the CLI with
#     ``--train_all`` (overrides any config that forgets it).
#   - Bundle filenames are <arch>_weights.joblib — matches the
#     NS_production/Helpers/helpers.py::get_scratch_model lookup so no
#     renaming is needed when copying to NS_production/model/.

set -euo pipefail

ROOT_DIR="/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling"
DEPLOY_DIR="${ROOT_DIR}/experiments/configs/deployment"
RUNNER="${ROOT_DIR}/training/train_classical.py"

VALID_ARCHS=(mahadevan2021 ji2023 mdpi2024_fe mdpi2024_cnn)

if [[ $# -lt 1 ]]; then
    echo "Error: missing arch argument." >&2
    echo "Usage: $0 <arch> [extra args]" >&2
    echo "Architectures: ${VALID_ARCHS[*]} | all" >&2
    exit 2
fi

ARCH="$1"
shift

run_one() {
    local arch="$1"; shift
    local cfg="${DEPLOY_DIR}/deploy_${arch}_all.yaml"
    if [[ ! -f "${cfg}" ]]; then
        echo "Error: config not found: ${cfg}" >&2
        exit 2
    fi
    echo "=================================================="
    echo " Deployment retrain — ${arch} (all NOPROD)"
    echo "   config: ${cfg}"
    echo "=================================================="
    python3.11 "${RUNNER}" --config "${cfg}" "$@"
}

if [[ "${ARCH}" == "all" ]]; then
    # Train fastest → slowest so a partial failure still leaves the cheap
    # ones done.  Mahadevan + Ji are minutes; mdpi_fe (tsfresh) is tens of
    # minutes; mdpi_cnn (200 epochs) is hours.
    for arch in mahadevan2021 ji2023 mdpi2024_fe mdpi2024_cnn; do
        run_one "${arch}" "$@"
    done
else
    matched=0
    for a in "${VALID_ARCHS[@]}"; do
        if [[ "$a" == "${ARCH}" ]]; then matched=1; break; fi
    done
    if [[ ${matched} -eq 0 ]]; then
        echo "Error: unknown arch '${ARCH}'." >&2
        echo "Architectures: ${VALID_ARCHS[*]} | all" >&2
        exit 2
    fi
    run_one "${ARCH}" "$@"
fi

echo
echo "=================================================="
echo " Done.  To deploy, copy the bundle(s) into NS_production/model/:"
echo
if [[ "${ARCH}" == "all" ]]; then
    for arch in "${VALID_ARCHS[@]}"; do
        echo "   cp /mnt/data/GENEActive-featurized/results/DL/<dataset>/deploy-${arch}-all-NOPROD/training/model_weights/${arch}_weights.joblib  NS_production/model/"
    done
else
    echo "   cp /mnt/data/GENEActive-featurized/results/DL/<dataset>/deploy-${ARCH}-all-NOPROD/training/model_weights/${ARCH}_weights.joblib  NS_production/model/"
fi
echo
echo " Then run the production pipeline with --scratch_model <arch>:"
echo "   python3.11 NS_production/NS-pipeline.py ... --scratch --scratch_model <arch>"
echo "=================================================="
