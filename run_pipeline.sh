#!/usr/bin/env bash
set -euo pipefail

# One-shot production runner for the retained model families.

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${STAGE:-full}"
TRAIN_PATH="${TRAIN_PATH:-}"
TEST_PATH="${TEST_PATH:-}"

export TRAIN_PATH
export TEST_PATH

XGB_ARGS=${XGB_ARGS:-""}
BERT_ARGS=${BERT_ARGS:-""}
FULL_ARGS=${FULL_ARGS:-"--mode full-only --task-modes binary_derived"}
PREF_ARGS=${PREF_ARGS:-"--stage ${STAGE} --task-mode binary_derived --methods dpo,orpo,kto --modes lora,qlora"}

echo "[1/4] XGBoost baseline"
$PYTHON_BIN run_xgboost_baseline.py --stage "$STAGE" $XGB_ARGS

echo "[2/4] Encoder baseline"
$PYTHON_BIN run_encoder_baseline.py --stage "$STAGE" $BERT_ARGS

echo "[3/4] SFT pipeline"
$PYTHON_BIN run_sft_pipeline.py $FULL_ARGS

echo "[4/4] Preference pipeline (DPO/ORPO/KTO)"
$PYTHON_BIN run_prefopt_pipeline.py $PREF_ARGS

echo "Completed all pipelines."
