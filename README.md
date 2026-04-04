# Preference Optimization Experiment Framework

Production-oriented experiment runner for:

- preference optimization (`DPO`, `ORPO`, `KTO`) as the main pipeline
- reference baselines (`SFT`, `BERT`, `XGBoost`) for side-by-side comparison

## Repository Layout

- `run_prefopt_pipeline.py`: central preference-optimization runner
- `run_sft_pipeline.py`: SFT LoRA/QLoRA sweep
- `run_encoder_baseline.py`: encoder baseline runner
- `run_xgboost_baseline.py`: classical baseline runner
- `run_pipeline.sh`: one-shot orchestration across all families

## Operational Quick Start

Run the central production path:

```bash
python run_prefopt_pipeline.py --stage full --task-mode binary_derived --methods dpo,orpo,kto --modes lora,qlora
```

Run all families in one command:

```bash
./run_pipeline.sh
```

One-shot runtime overrides:

- `PYTHON_BIN` (default: `python`)
- `STAGE` (default: `full`)
- `TRAIN_PATH` (optional; shared data path override for all runners)
- `TEST_PATH` (optional; shared data path override for all runners)
- `PREF_ARGS`
- `FULL_ARGS`
- `BERT_ARGS`
- `XGB_ARGS`

Example:

```bash
STAGE=smoke PREF_ARGS="--stage smoke --task-mode binary_derived --methods orpo --modes qlora" ./run_pipeline.sh
```

Example with shared dataset-path override:

```bash
TRAIN_PATH=/data/train.csv TEST_PATH=/data/test.csv ./run_pipeline.sh
```

## Input Contract

Current scripts expect CSV input with these required columns:

- `value` (text payload)
- `GAD_1`, `GAD_2`, `PHQ_1`, `PHQ_2` (integer labels, `0..3`)

Derived binary targets used internally:

- `anxiety = (GAD_1 > 0) or (GAD_2 > 0)`
- `depression = (PHQ_1 > 0) or (PHQ_2 > 0)`

Minimal row:

```csv
value,GAD_1,GAD_2,PHQ_1,PHQ_2
"I feel overwhelmed and cannot sleep lately.",2,1,1,0
```

Path configuration options:

- CLI flags per runner: `--train-path`, `--test-path`
- Environment variables: `TRAIN_PATH`, `TEST_PATH`
- Defaults fallback to the current local dataset paths used in this environment

## Output Contract

Each runner writes both timestamped artifacts and rolling `latest` artifacts.

- XGBoost: `xgb_results/xgb_results_<timestamp>.csv|jsonl`, `xgb_results/xgb_results_latest.csv|jsonl`
- BERT: `bert_results/bert_results_<timestamp>.csv|jsonl`, `bert_results/bert_results_latest.csv|jsonl`
- SFT: `sweep_results/{smoke,full,all}_results_<timestamp>.csv|jsonl`, `sweep_results/{smoke,full,all}_results_latest.csv|jsonl`
- Preference: `pref_results/pref_results_<timestamp>.csv|jsonl`, `pref_results/pref_results_latest.csv|jsonl`

Use `csv` for reporting and `jsonl` for append-friendly run logs.

## Production Notes

- Generated outputs are intentionally not tracked by git.
- Repository is intentionally minimal to keep PRs focused on runnable code.
- Historical outputs and legacy files are archived in `archive/cleanup_YYYYMMDD_HHMMSS/`.

## Extensibility Path

To repurpose this framework for non-mental-health domains:

- parameterize dataset path and column mapping
- externalize label-mapping logic
- externalize prompt/template logic
- keep orchestration and logging flow unchanged

## Citation

If needed, citation for the originating study:

```bibtex
@misc{arcan2026baselinespreferencescomparativestudy,
  title={From Baselines to Preferences: A Comparative Study of LoRA/QLoRA and Preference Optimization for Mental Health Text Classification},
  author={Mihael Arcan},
  year={2026},
  eprint={2604.00773},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.00773}
}
```
