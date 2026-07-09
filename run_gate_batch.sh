#!/bin/bash
# ==============================================================================
# Batch gate-vs-full comparison across multiple MSI datasets.
#
# For every dataset under DATA_ROOT (each a subfolder with images/ + masks/):
#   - auto-detect channel count C
#   - train gate (learned hard top-k), full input (M_{k=B}), and optionally a
#     random-k control and/or a Concrete-Autoencoder selection baseline
#     (--concrete), for each seed and each k
#   - emit per-dataset comparison + selected-band charts and a cross-dataset
#     summary (does the gate help more on wider candidate ranges?)
#
# Resilient to crashes: each run is an isolated subprocess; finished runs are
# skipped on re-run (train.py done-flag), progress tracked in batch_manifest.json.
#
# Usage:
#   bash run_gate_batch.sh                       # defaults below
#   bash run_gate_batch.sh --dry_run             # list planned runs only
#   bash run_gate_batch.sh --plots_only          # rebuild charts from manifest
#   bash run_gate_batch.sh --seeds 42,123 --ks 2,3,4 --random_k
#   bash run_gate_batch.sh --concrete            # add the Concrete-AE baseline arm
# ==============================================================================

set -e

# ===== user config =====
DATA_ROOT="/root/autodl-fs/15"
OUT_ROOT="outputs/gate_batch"
SEEDS="42,123,456"
KS="3,4,5"
EPOCHS=150
EXTRA=""        # set to "" to disable the random-k control
# =======================

python scripts/run_gate_batch.py \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --seeds "$SEEDS" \
    --ks "$KS" \
    --epochs "$EPOCHS" \
    $EXTRA \
    "$@"
