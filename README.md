# MH-Bench: Joint Mental Health Classification Benchmark

## Goal
Build a fair, reproducible benchmark for classifying joint anxiety/depression states, and compare classic ML, encoder models, SFT adapters, and preference-optimization methods under matched evaluation settings.

## Methods compared
- **XGBoost** (TF-IDF baseline)
- **BERT-family encoders**: `bert-base-uncased`, `distilbert-base-uncased`, `roberta-base` (vanilla + temperature-scaled)
- **Qwen SFT adapters**: LoRA / QLoRA, generative + discriminative settings, optimizer sweeps
- **Preference optimization**: DPO / ORPO / KTO (LoRA + QLoRA; vanilla + rebalanced)

## Dataset/task description (1-2 lines)
The primary task is a **4-class joint label** setup (`0_0`, `1_0`, `0_1`, `1_1`) derived from anxiety/depression signals, evaluated with macro/weighted F1 and macro precision/recall.  
Experiments use chunked text windows and compare methods across shared window settings for fair cross-family reporting.

## Best headline results (3-6 bullets)
- **Best BERT result**: `bert-base-uncased (Optimized/Temp)` reached **F1_macro = 0.3518** (window 384).
- **Best XGBoost result**: **F1_macro = 0.3375** (window 512).
- **Best SFT result**: **QLoRA Vanilla = 0.3399** (window 512).
- **Best preference (vanilla)**: **ORPO = 0.3181**.
- **Best preference (rebalanced)**: **ORPO = 0.3798** (strongest preference run).

## Key findings (5 bullets)
- `bert-base-uncased` is the strongest encoder baseline overall in this setup.
- QLoRA generally matches or slightly exceeds LoRA at peak SFT performance.
- Preference methods are highly sensitive to setup: ORPO benefited most from rebalancing.
- KTO remained weak in this benchmark configuration, both vanilla and rebalanced.
- Window size matters: best runs repeatedly cluster around mid/large windows (notably 384/512), reinforcing the need for explicit window ablations.
