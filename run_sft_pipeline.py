import os
import gc
import json
import time
import math
import random
import inspect
import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Adafactor,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

DEFAULT_TRAIN_PATH = "/mnt/3d56a808-6937-48b6-ba29-c2b68a3e2d94/datasets/train_dataset.csvV4_per_user"
DEFAULT_TEST_PATH = "/mnt/3d56a808-6937-48b6-ba29-c2b68a3e2d94/datasets/test_dataset.csvV4_per_user"


# ----------------------------
# Config
# ----------------------------

@dataclass
class RunConfig:
    run_name: str
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"   # safer starting point on 16 GB
    mode: Literal["full", "lora", "qlora"] = "lora"
    objective: Literal["generative", "discriminative"] = "generative"
    optimizer_name: Literal["adamw", "adafactor", "adam8bit"] = "adamw"
    output_schema: Literal["label_only", "label_rationale", "label_confidence"] = "label_only"
    task_mode: Literal["binary_derived", "ordinal_raw"] = "binary_derived"

    max_length: int = 768
    max_new_tokens_eval: int = 64
    gen_eval_batch_size: int = 8

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    use_class_weights: bool = False
    class_weight_clip: float = 8.0
    use_temperature_scaling: bool = False
    minority_threshold_grid: str = ""
    minority_class_name: Literal["0_0", "1_0", "0_1", "1_1"] = "0_1"
    minority_min_recall: float = 0.0
    minority_min_recall_grid: str = ""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    seed: int = 42
    bf16: bool = True
    smoke_fraction: float = 0.10
    eval_fraction: float = 0.10
    window_size: int = 512
    window_overlap: int = 128

    output_dir_root: str = "./sweep_results"


# ----------------------------
# Repro / memory helpers
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def frac_tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def make_training_arguments(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters)
    # Compat across HF versions.
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**filtered)


def make_trainer_kwargs(**kwargs):
    sig = inspect.signature(Trainer.__init__)
    params = set(sig.parameters)
    # Transformers >=5 uses `processing_class` instead of `tokenizer`.
    if "tokenizer" in kwargs and "tokenizer" not in params and "processing_class" in params:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    return {k: v for k, v in kwargs.items() if k in params}


JOINT_CLASS_LABELS = [0, 1, 2, 3]  # 0_0, 1_0, 0_1, 1_1
JOINT_CLASS_NAMES = {0: "0_0", 1: "1_0", 2: "0_1", 3: "1_1"}


def compute_joint_class_metrics(y_true, y_pred) -> Dict[str, Any]:
    if len(y_true) == 0:
        out: Dict[str, Any] = {}
        for label in JOINT_CLASS_LABELS:
            name = JOINT_CLASS_NAMES[label]
            out[f"class_{name}_f1"] = np.nan
            out[f"class_{name}_recall"] = np.nan
            out[f"class_{name}_support"] = 0
        return out

    _, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=JOINT_CLASS_LABELS,
        zero_division=0,
    )
    out: Dict[str, Any] = {}
    for idx, label in enumerate(JOINT_CLASS_LABELS):
        name = JOINT_CLASS_NAMES[label]
        out[f"class_{name}_f1"] = float(f1[idx])
        out[f"class_{name}_recall"] = float(recall[idx])
        out[f"class_{name}_support"] = int(support[idx])
    return out


def parse_float_grid(spec: str) -> List[float]:
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    vals: List[float] = []
    for tok in spec.split(","):
        t = tok.strip()
        if not t:
            continue
        v = float(t)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Threshold value out of [0,1]: {v}")
        vals.append(v)
    return sorted(set(vals))


def parse_chunk_grid(spec: str) -> List[Tuple[int, int]]:
    if spec is None or not spec.strip():
        return [(94, 24), (192, 48), (384, 96), (512, 128)]
    out: List[Tuple[int, int]] = []
    for token in spec.split(","):
        t = token.strip()
        if not t:
            continue
        if ":" not in t:
            raise ValueError(f"Invalid chunk token '{t}'. Expected format window:overlap")
        ws_s, ov_s = t.split(":", 1)
        ws = int(ws_s.strip())
        ov = int(ov_s.strip())
        if ws <= 0 or ov < 0 or ov >= ws:
            raise ValueError(f"Invalid chunk pair ({ws}, {ov}). Need ws>0 and 0<=ov<ws.")
        out.append((ws, ov))
    if not out:
        raise ValueError("Chunk grid is empty.")
    return out


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(np.sum(ex, axis=1, keepdims=True), 1e-12, None)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    if len(labels) == 0:
        return float("nan")
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(labels))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(accuracies[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += (float(np.sum(mask)) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def multiclass_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 4) -> float:
    if len(labels) == 0:
        return float("nan")
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calibration_metrics_from_probs(probs: np.ndarray, labels: np.ndarray, num_classes: int = 4) -> Dict[str, float]:
    return {
        "ece": expected_calibration_error(probs, labels, n_bins=15),
        "brier": multiclass_brier_score(probs, labels, num_classes=num_classes),
    }


def fit_temperature_grid(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.concatenate([np.arange(0.5, 1.05, 0.05), np.arange(1.1, 3.05, 0.1)])
    best_t = 1.0
    best_nll = float("inf")
    for t in candidates:
        probs = softmax_np(logits / t)
        p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
        nll = float(-np.mean(np.log(p)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def apply_minority_threshold(
    probs: np.ndarray,
    base_preds: np.ndarray,
    minority_idx: int,
    threshold: float,
) -> np.ndarray:
    preds = base_preds.copy()
    preds[probs[:, minority_idx] >= threshold] = minority_idx
    return preds


def evaluate_joint_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    p_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    r_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    r_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    out = {
        "accuracy": acc,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "precision_macro": p_m,
        "precision_weighted": p_w,
        "recall_macro": r_m,
        "recall_weighted": r_w,
        "joint_exact_match": acc,
        "joint_accuracy": acc,
        "joint_f1_macro": f1_m,
        "joint_f1_weighted": f1_w,
        "joint_precision_macro": p_m,
        "joint_precision_weighted": p_w,
        "joint_recall_macro": r_m,
        "joint_recall_weighted": r_w,
    }
    out.update(compute_joint_class_metrics(y_true, y_pred))
    return out


def evaluate_secondary_binary_from_joint(
    y_true_joint: np.ndarray,
    y_pred_joint: np.ndarray,
    joint_probs: Optional[np.ndarray] = None,
    include_calibration: bool = True,
) -> Dict[str, Any]:
    y_true_joint = np.asarray(y_true_joint, dtype=int)
    y_pred_joint = np.asarray(y_pred_joint, dtype=int)
    y_true_anx = (y_true_joint % 2).astype(int)
    y_true_dep = (y_true_joint // 2).astype(int)
    y_pred_anx = (y_pred_joint % 2).astype(int)
    y_pred_dep = (y_pred_joint // 2).astype(int)

    out: Dict[str, Any] = {}
    targets = [
        ("anxiety", y_true_anx, y_pred_anx),
        ("depression", y_true_dep, y_pred_dep),
    ]
    for name, y_true, y_pred in targets:
        out[f"{name}_accuracy"] = float(accuracy_score(y_true, y_pred))
        out[f"{name}_f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        out[f"{name}_f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        out[f"{name}_precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        out[f"{name}_precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        out[f"{name}_recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        out[f"{name}_recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    if include_calibration and joint_probs is not None:
        joint_probs = np.asarray(joint_probs, dtype=np.float64)
        anx_pos = np.clip(joint_probs[:, 1] + joint_probs[:, 3], 1e-12, 1.0 - 1e-12)
        dep_pos = np.clip(joint_probs[:, 2] + joint_probs[:, 3], 1e-12, 1.0 - 1e-12)
        anx_probs = np.stack([1.0 - anx_pos, anx_pos], axis=1)
        dep_probs = np.stack([1.0 - dep_pos, dep_pos], axis=1)
        out["anxiety_ece"] = expected_calibration_error(anx_probs, y_true_anx, n_bins=15)
        out["anxiety_brier"] = multiclass_brier_score(anx_probs, y_true_anx, num_classes=2)
        out["depression_ece"] = expected_calibration_error(dep_probs, y_true_dep, n_bins=15)
        out["depression_brier"] = multiclass_brier_score(dep_probs, y_true_dep, num_classes=2)
    else:
        out["anxiety_ece"] = np.nan
        out["anxiety_brier"] = np.nan
        out["depression_ece"] = np.nan
        out["depression_brier"] = np.nan
    return out


def compute_class_weights(train_labels: np.ndarray, n_classes: int = 4, clip: float = 8.0) -> torch.Tensor:
    counts = np.bincount(train_labels, minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    n = float(np.sum(counts))
    weights = n / (n_classes * counts)
    weights = weights / np.mean(weights)
    if clip is not None and clip > 0:
        weights = np.clip(weights, 0.0, clip)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedDiscriminativeTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs["logits"]
        if self.class_weights is None:
            loss_fct = torch.nn.CrossEntropyLoss()
        else:
            cw = self.class_weights.to(logits.device, dtype=logits.dtype)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ----------------------------
# Data
# ----------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # remove accidental index columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = {"value", "GAD_1", "GAD_2", "PHQ_1", "PHQ_2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["value"] = df["value"].fillna("").astype(str)

    # binary derived labels
    df["anxiety"] = ((df["GAD_1"] > 0) | (df["GAD_2"] > 0)).astype(int)
    df["depression"] = ((df["PHQ_1"] > 0) | (df["PHQ_2"] > 0)).astype(int)

    # composite for stratification
    df["label_combo"] = df["anxiety"].astype(str) + "_" + df["depression"].astype(str)

    # discriminative target
    df["joint_label"] = (
        (df["anxiety"] == 1) & (df["depression"] == 0)
    ).astype(int) + 2 * (
        (df["anxiety"] == 0) & (df["depression"] == 1)
    ).astype(int) + 3 * (
        (df["anxiety"] == 1) & (df["depression"] == 1)
    ).astype(int)

    return df


def load_data(train_path: str, test_path: str, val_size=0.2, seed=42):

    # -----------------------
    # 1. Load raw data
    # -----------------------
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # -----------------------
    # 2. Preprocess BOTH
    # -----------------------
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    # -----------------------
    # 3. Train → train/val split
    # -----------------------
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["label_combo"],
        random_state=seed
    )

    # -----------------------
    # 4. Convert to HF datasets
    # -----------------------
    ds = DatasetDict({
        "train": Dataset.from_pandas(train_split.reset_index(drop=True), preserve_index=False),
        "validation": Dataset.from_pandas(val_split.reset_index(drop=True), preserve_index=False),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True), preserve_index=False)
    })

    # -----------------------
    # 5. Debug distribution (VERY useful)
    # -----------------------
    print("\n=== DATA DISTRIBUTION ===")
    print("Train:\n", train_split["label_combo"].value_counts())
    print("Validation:\n", val_split["label_combo"].value_counts())
    print("Test:\n", test_df["label_combo"].value_counts())

    return ds


def make_splits(df: pd.DataFrame, seed: int = 42, val_size: float = 0.2) -> DatasetDict:
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["label_combo"],
    )
    return DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True), preserve_index=False),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True), preserve_index=False),
    })


def subsample_dataset(ds: DatasetDict, frac: float, seed: int) -> DatasetDict:
    if frac >= 1.0:
        return ds
    out = {}
    rng = random.Random(seed)
    for split, split_ds in ds.items():
        n = len(split_ds)
        k = max(1, int(n * frac))
        idx = list(range(n))
        rng.shuffle(idx)
        idx = idx[:k]
        out[split] = split_ds.select(idx)
    return DatasetDict(out)


def subsample_split(split_ds: Dataset, frac: float, seed: int) -> Dataset:
    if frac >= 1.0:
        return split_ds
    n = len(split_ds)
    if n == 0:
        return split_ds
    k = max(1, int(n * frac))
    k = min(k, n)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    return split_ds.select(idx[:k])


def chunk_text_words(text: str, window_size: int, overlap: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return [""]
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if overlap < 0:
        raise ValueError(f"window_overlap must be >= 0, got {overlap}")
    if overlap >= window_size:
        raise ValueError(
            f"window_overlap ({overlap}) must be smaller than window_size ({window_size})"
        )
    if len(words) <= window_size:
        return [" ".join(words)]

    step = window_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window_words = words[start:start + window_size]
        if not window_words:
            break
        chunks.append(" ".join(window_words))
        if start + window_size >= len(words):
            break
    return chunks


def expand_dataset_with_chunks(ds: DatasetDict, cfg: RunConfig) -> DatasetDict:
    out = {}
    for split in ["train", "validation", "test"]:
        rows: List[Dict[str, Any]] = []
        for idx, ex in enumerate(ds[split]):
            chunks = chunk_text_words(
                ex.get("value", ""),
                window_size=cfg.window_size,
                overlap=cfg.window_overlap,
            )
            for chunk_id, chunk_text in enumerate(chunks):
                row = dict(ex)
                row["value"] = chunk_text
                row["__example_id"] = idx
                row["__chunk_id"] = chunk_id
                rows.append(row)
        out[split] = Dataset.from_list(rows)
    return DatasetDict(out)


# ----------------------------
# Prompt / targets
# ----------------------------

def build_generative_target(ex: Dict[str, Any], task_mode: str, output_schema: str) -> str:
    if task_mode == "binary_derived":
        base = {
            "anxiety": int(ex["anxiety"]),
            "depression": int(ex["depression"]),
        }
    else:
        base = {
            "GAD_1": int(ex["GAD_1"]),
            "GAD_2": int(ex["GAD_2"]),
            "PHQ_1": int(ex["PHQ_1"]),
            "PHQ_2": int(ex["PHQ_2"]),
        }

    if output_schema == "label_only":
        target = base
    elif output_schema == "label_rationale":
        target = {
            **base,
            "rationale": "Brief evidence from the transcript."
        }
    elif output_schema == "label_confidence":
        target = base.copy()
        for k in list(base.keys()):
            target[f"{k}_confidence"] = 1.0
    else:
        raise ValueError(output_schema)

    return json.dumps(target, separators=(",", ":"))


def build_messages(ex: Dict[str, Any], task_mode: str, output_schema: str):
    if task_mode == "binary_derived":
        label_part = 'Return JSON with keys "anxiety" and "depression".'
    else:
        label_part = 'Return JSON with keys "GAD_1","GAD_2","PHQ_1","PHQ_2".'

    if output_schema == "label_only":
        extra = "Return labels only. Do not add explanations."
    elif output_schema == "label_rationale":
        extra = 'Also include a short "rationale" field.'
    else:
        extra = "Also include confidence values between 0 and 1 for each label."

    system = (
        "You are a mental health transcript classifier. "
        "Use only the transcript. "
        f"{label_part} {extra}"
    )
    user = f"Transcript:\n{ex['value']}"
    assistant = build_generative_target(ex, task_mode, output_schema)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ----------------------------
# Tokenization
# ----------------------------

def tokenize_generative(ex, tokenizer, cfg: RunConfig):
    messages = build_messages(ex, cfg.task_mode, cfg.output_schema)
    prompt_messages = messages[:-1]
    answer_text = messages[-1]["content"]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(
        prompt_text,
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]
    answer_ids = tokenizer(
        answer_text,
        truncation=True,
        max_length=cfg.max_length,
        add_special_tokens=False,
    )["input_ids"]

    # Ensure at least one supervised token survives truncation.
    if not answer_ids:
        fallback_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        answer_ids = [fallback_token]

    if len(answer_ids) > cfg.max_length:
        answer_ids = answer_ids[:cfg.max_length]
    max_prompt_tokens = cfg.max_length - len(answer_ids)
    prompt_ids = prompt_ids[:max(0, max_prompt_tokens)]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def tokenize_discriminative(ex, tokenizer, cfg: RunConfig):
    if cfg.task_mode != "binary_derived":
        raise ValueError("Discriminative objective currently supports only task_mode=binary_derived")
    toks = tokenizer(
        ex["value"],
        truncation=True,
        max_length=cfg.max_length,
    )
    toks["labels"] = int(ex["joint_label"])
    return toks


# ----------------------------
# Model factory
# ----------------------------

def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model_and_tokenizer(cfg: RunConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation with batched inputs should use left padding.
    tokenizer.padding_side = "left" if cfg.objective == "generative" else "right"

    common_kwargs = {"trust_remote_code": True, "device_map": "auto"}

    if cfg.objective == "generative":
        if cfg.mode == "qlora":
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                quantization_config=make_bnb_config(),
                **common_kwargs,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
                **common_kwargs,
            )
    else:
        if cfg.mode == "qlora":
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model_name,
                num_labels=4,
                quantization_config=make_bnb_config(),
                **common_kwargs,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model_name,
                num_labels=4,
                torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
                **common_kwargs,
            )

    if cfg.mode in {"lora", "qlora"}:
        if cfg.objective == "generative":
            task_type = "CAUSAL_LM"
        else:
            task_type = "SEQ_CLS"

        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=task_type,
        )
        model = get_peft_model(model, peft_cfg)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# ----------------------------
# Optimizer
# ----------------------------

def make_optimizer(model, cfg: RunConfig):
    params = [p for p in model.parameters() if p.requires_grad]

    if cfg.optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    if cfg.optimizer_name == "adafactor":
        return Adafactor(
            params,
            lr=cfg.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=cfg.weight_decay,
        )

    if cfg.optimizer_name == "adam8bit":
        return bnb.optim.Adam8bit(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    raise ValueError(cfg.optimizer_name)


# ----------------------------
# Metrics
# ----------------------------

def parse_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end+1])
    except Exception:
        return None


@torch.inference_mode()
def eval_generative(model, tokenizer, ds: Dataset, cfg: RunConfig) -> Dict[str, float]:
    valid_json = 0
    raw_keys = ["GAD_1", "GAD_2", "PHQ_1", "PHQ_2"]
    n = len(ds)
    if n == 0:
        return {
            "valid_json_rate": 0.0,
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "joint_exact_match": 0.0,
        }

    def safe_int(value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    had_gc = bool(getattr(model, "is_gradient_checkpointing", False))
    original_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if had_gc:
        model.gradient_checkpointing_disable()
    if original_use_cache is not None:
        model.config.use_cache = True

    if cfg.task_mode == "binary_derived":
        y_true_joint = []
        y_pred_joint = []
        gold: Dict[str, List[int]] = {"anxiety": [], "depression": []}
        pred: Dict[str, List[int]] = {"anxiety": [], "depression": []}
    else:
        gold = {k: [] for k in raw_keys}
        pred = {k: [] for k in raw_keys}

    try:
        batch_size = max(1, int(cfg.gen_eval_batch_size))
        for start_idx in range(0, n, batch_size):
            batch = [ds[i] for i in range(start_idx, min(start_idx + batch_size, n))]
            seen = min(start_idx + len(batch), n)
            if seen % 25 == 0 or seen == n:
                print(f"[gen-eval:{cfg.run_name}] {seen}/{n}")

            prompt_texts = []
            for ex in batch:
                prompt_messages = build_messages(ex, cfg.task_mode, cfg.output_schema)[:-1]
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_texts.append(prompt_text)

            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens_eval,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            for i, ex in enumerate(batch):
                gen = tokenizer.decode(
                    outputs[i][int(input_lens[i]):],
                    skip_special_tokens=True,
                )
                parsed = parse_json(gen)

                if cfg.task_mode == "binary_derived":
                    gold_a = int(ex["anxiety"])
                    gold_d = int(ex["depression"])
                    gold["anxiety"].append(gold_a)
                    gold["depression"].append(gold_d)

                    if parsed is not None and "anxiety" in parsed and "depression" in parsed:
                        valid_json += 1
                        pred["anxiety"].append(1 if safe_int(parsed["anxiety"]) > 0 else 0)
                        pred["depression"].append(1 if safe_int(parsed["depression"]) > 0 else 0)
                    else:
                        pred["anxiety"].append(0)
                        pred["depression"].append(0)

                    y_true_joint.append(gold_a + 2 * gold_d)
                    y_pred_joint.append(pred["anxiety"][-1] + 2 * pred["depression"][-1])
                else:
                    for k in raw_keys:
                        gold[k].append(int(ex[k]))

                    if parsed is not None and all(k in parsed for k in raw_keys):
                        valid_json += 1
                        for k in raw_keys:
                            pred_val = safe_int(parsed[k])
                            pred[k].append(max(0, min(3, pred_val)))
                    else:
                        for k in raw_keys:
                            pred[k].append(0)
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        if had_gc:
            model.gradient_checkpointing_enable()

    if cfg.task_mode == "binary_derived":
        y_true_joint_np = np.asarray(y_true_joint, dtype=int)
        y_pred_joint_np = np.asarray(y_pred_joint, dtype=int)
        out = {
            "valid_json_rate": valid_json / max(1, len(ds)),
            **evaluate_joint_preds(y_true_joint_np, y_pred_joint_np),
            **evaluate_secondary_binary_from_joint(y_true_joint_np, y_pred_joint_np, None, False),
            "ece": np.nan,
            "brier": np.nan,
        }
        return out

    per_key_f1 = [
        f1_score(gold[k], pred[k], average="macro", zero_division=0)
        for k in raw_keys
    ]
    per_key_f1_weighted = [
        f1_score(gold[k], pred[k], average="weighted", zero_division=0)
        for k in raw_keys
    ]
    per_key_recall = [
        recall_score(gold[k], pred[k], average="macro", zero_division=0)
        for k in raw_keys
    ]
    per_key_accuracy = [
        accuracy_score(gold[k], pred[k])
        for k in raw_keys
    ]

    return {
        "valid_json_rate": valid_json / max(1, len(ds)),
        "accuracy": float(np.mean(per_key_accuracy)),
        "f1_macro": float(np.mean(per_key_f1)),
        "f1_weighted": float(np.mean(per_key_f1_weighted)),
        "raw_f1_macro_mean": float(np.mean(per_key_f1)),
        "raw_recall_macro_mean": float(np.mean(per_key_recall)),
        "raw_joint_exact_match": float(
            np.mean(
                [
                    all(gold[k][i] == pred[k][i] for k in raw_keys)
                    for i in range(len(ds))
                ]
            )
        ),
    }


@torch.inference_mode()
def eval_discriminative(trainer: Trainer, ds: Dataset) -> Dict[str, float]:
    pred_out = trainer.predict(ds)
    y_true = pred_out.label_ids
    y_pred = np.argmax(pred_out.predictions, axis=-1)
    return evaluate_joint_preds(y_true, y_pred)


# ----------------------------
# Trainer factory
# ----------------------------

def build_trainer(model, tokenizer, ds_tok: DatasetDict, cfg: RunConfig, class_weights: Optional[torch.Tensor] = None):
    run_dir = os.path.join(cfg.output_dir_root, cfg.run_name)

    args = make_training_arguments(
        output_dir=run_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=cfg.bf16,
        fp16=not cfg.bf16,
        max_grad_norm=cfg.max_grad_norm,
    )

    optimizer = make_optimizer(model, cfg)

    if cfg.objective == "generative":
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )
    else:
        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    trainer_kwargs = make_trainer_kwargs(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        compute_metrics=(
            compute_metrics_discriminative
            if cfg.objective == "discriminative"
            else None
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(optimizer, None),
    )
    if cfg.objective == "discriminative":
        return WeightedDiscriminativeTrainer(**trainer_kwargs, class_weights=class_weights)
    return Trainer(**trainer_kwargs)


# ----------------------------
# One run
# ----------------------------

def run_one(cfg: RunConfig, full_ds: DatasetDict) -> Dict[str, Any]:
    set_seed(cfg.seed)
    reset_cuda()

    run_start = time.time()

    ds = deepcopy(full_ds)
    ds["train"] = subsample_split(ds["train"], cfg.smoke_fraction, cfg.seed)
    ds["validation"] = subsample_split(ds["validation"], cfg.eval_fraction, cfg.seed + 1)
    ds["test"] = subsample_split(ds["test"], cfg.eval_fraction, cfg.seed + 2)
    ds = expand_dataset_with_chunks(ds, cfg)
    print(
        f"[split:{cfg.run_name}] train={len(ds['train'])} "
        f"val={len(ds['validation'])} test={len(ds['test'])} "
        f"(smoke_fraction={cfg.smoke_fraction}, eval_fraction={cfg.eval_fraction}, "
        f"window={cfg.window_size}, overlap={cfg.window_overlap})"
    )

    model, tokenizer = load_model_and_tokenizer(cfg)

    if cfg.objective == "generative":
        ds_tok = ds.map(
            lambda ex: tokenize_generative(ex, tokenizer, cfg),
            remove_columns=ds["train"].column_names,
        )
    else:
        ds_tok = ds.map(
            lambda ex: tokenize_discriminative(ex, tokenizer, cfg),
            remove_columns=ds["train"].column_names,
        )

    class_weights = None
    if cfg.objective == "discriminative" and cfg.use_class_weights:
        train_labels = np.array(ds_tok["train"]["labels"], dtype=int)
        class_weights = compute_class_weights(train_labels, n_classes=4, clip=cfg.class_weight_clip)
        print(f"[weights:{cfg.run_name}] class_weights={class_weights.tolist()}")

    trainer = build_trainer(model, tokenizer, ds_tok, cfg, class_weights=class_weights)
    train_out = trainer.train()

    if cfg.objective == "generative":
        val_metrics = eval_generative(model, tokenizer, ds["validation"], cfg)
        test_metrics = eval_generative(model, tokenizer, ds["test"], cfg)
    else:
        val_pred_out = trainer.predict(ds_tok["validation"])
        test_pred_out = trainer.predict(ds_tok["test"])
        val_logits = np.asarray(val_pred_out.predictions)
        val_labels = np.asarray(val_pred_out.label_ids)
        test_logits = np.asarray(test_pred_out.predictions)
        test_labels = np.asarray(test_pred_out.label_ids)

        val_preds = np.argmax(val_logits, axis=-1)
        test_preds = np.argmax(test_logits, axis=-1)
        val_metrics = evaluate_joint_preds(val_labels, val_preds)
        test_metrics = evaluate_joint_preds(test_labels, test_preds)
        val_probs = softmax_np(val_logits)
        test_probs = softmax_np(test_logits)
        val_metrics.update(calibration_metrics_from_probs(val_probs, val_labels, num_classes=4))
        test_metrics.update(calibration_metrics_from_probs(test_probs, test_labels, num_classes=4))
        val_metrics.update(evaluate_secondary_binary_from_joint(val_labels, val_preds, val_probs, True))
        test_metrics.update(evaluate_secondary_binary_from_joint(test_labels, test_preds, test_probs, True))

        selected_temperature = 1.0
        selected_minority_threshold = 0.0
        selected_minority_min_recall = float(cfg.minority_min_recall)
        post_val_metrics = dict(val_metrics)
        post_test_metrics = dict(test_metrics)

        threshold_grid = parse_float_grid(cfg.minority_threshold_grid)
        minority_idx = {v: k for k, v in JOINT_CLASS_NAMES.items()}.get(cfg.minority_class_name, 2)
        if cfg.use_temperature_scaling or threshold_grid:
            if cfg.use_temperature_scaling:
                selected_temperature = fit_temperature_grid(val_logits, val_labels)

            val_probs = softmax_np(val_logits / selected_temperature)
            test_probs = softmax_np(test_logits / selected_temperature)
            val_base_preds = np.argmax(val_probs, axis=-1)
            test_base_preds = np.argmax(test_probs, axis=-1)
            val_metrics.update(calibration_metrics_from_probs(val_probs, val_labels, num_classes=4))
            test_metrics.update(calibration_metrics_from_probs(test_probs, test_labels, num_classes=4))
            val_metrics.update(evaluate_secondary_binary_from_joint(val_labels, val_base_preds, val_probs, True))
            test_metrics.update(evaluate_secondary_binary_from_joint(test_labels, test_base_preds, test_probs, True))

            if threshold_grid:
                recall_targets = parse_float_grid(cfg.minority_min_recall_grid)
                if not recall_targets:
                    recall_targets = [float(cfg.minority_min_recall)]

                candidates = []
                for recall_target in recall_targets:
                    best_feasible = None
                    best_fallback = None
                    for t in threshold_grid:
                        val_preds_t = apply_minority_threshold(val_probs, val_base_preds, minority_idx, t)
                        m = evaluate_joint_preds(val_labels, val_preds_t)
                        score = float(m["joint_f1_macro"])
                        recall = float(m.get(f"class_{cfg.minority_class_name}_recall", 0.0))
                        rec = {
                            "target": float(recall_target),
                            "threshold": float(t),
                            "score": score,
                            "recall": recall,
                        }
                        if recall >= float(recall_target):
                            if (
                                best_feasible is None
                                or rec["score"] > best_feasible["score"]
                                or (rec["score"] == best_feasible["score"] and rec["recall"] > best_feasible["recall"])
                            ):
                                best_feasible = rec
                        if (
                            best_fallback is None
                            or rec["recall"] > best_fallback["recall"]
                            or (rec["recall"] == best_fallback["recall"] and rec["score"] > best_fallback["score"])
                        ):
                            best_fallback = rec
                    candidates.append(
                        {
                            **(best_feasible if best_feasible is not None else best_fallback),
                            "feasible": best_feasible is not None,
                        }
                    )

                feasible_candidates = [c for c in candidates if c["feasible"]]
                if feasible_candidates:
                    best = max(feasible_candidates, key=lambda c: (c["target"], c["score"], c["recall"]))
                else:
                    best = max(candidates, key=lambda c: (c["recall"], c["score"], -c["target"]))
                selected_minority_threshold = float(best["threshold"])
                selected_minority_min_recall = float(best["target"])

            val_post_preds = apply_minority_threshold(
                val_probs, val_base_preds, minority_idx, selected_minority_threshold
            )
            test_post_preds = apply_minority_threshold(
                test_probs, test_base_preds, minority_idx, selected_minority_threshold
            )
            post_val_metrics = evaluate_joint_preds(val_labels, val_post_preds)
            post_test_metrics = evaluate_joint_preds(test_labels, test_post_preds)
            post_val_metrics.update(evaluate_secondary_binary_from_joint(val_labels, val_post_preds, None, False))
            post_test_metrics.update(evaluate_secondary_binary_from_joint(test_labels, test_post_preds, None, False))

        val_metrics["selected_temperature"] = selected_temperature
        val_metrics["selected_minority_threshold"] = selected_minority_threshold
        val_metrics["selected_minority_min_recall"] = selected_minority_min_recall
        for k, v in post_val_metrics.items():
            val_metrics[f"post_{k}"] = v
        for k, v in post_test_metrics.items():
            test_metrics[f"post_{k}"] = v

    val_prefixed = {f"val_{k}": v for k, v in val_metrics.items()}
    test_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}

    elapsed = time.time() - run_start
    peak_mem = peak_gpu_mem_gb()

    result = {
        **asdict(cfg),
        "train_runtime_sec": elapsed,
        "train_loss": float(train_out.training_loss) if hasattr(train_out, "training_loss") else None,
        "peak_gpu_mem_gb": peak_mem,
        "val_acc": val_metrics.get("accuracy", val_metrics.get("joint_accuracy", val_metrics.get("joint_exact_match"))),
        "val_f1": val_metrics.get("f1_macro", val_metrics.get("joint_f1_macro", val_metrics.get("raw_f1_macro_mean"))),
        "val_f1_weighted": val_metrics.get("f1_weighted", val_metrics.get("joint_f1_weighted", np.nan)),
        "val_minority_f1": val_metrics.get("class_0_1_f1", np.nan),
        "val_minority_recall": val_metrics.get("class_0_1_recall", np.nan),
        "test_acc": test_metrics.get("accuracy", test_metrics.get("joint_accuracy", test_metrics.get("joint_exact_match"))),
        "test_f1": test_metrics.get("f1_macro", test_metrics.get("joint_f1_macro", test_metrics.get("raw_f1_macro_mean"))),
        "test_f1_weighted": test_metrics.get("f1_weighted", test_metrics.get("joint_f1_weighted", np.nan)),
        "val_precision_macro": val_metrics.get("precision_macro", val_metrics.get("joint_precision_macro", np.nan)),
        "val_precision_weighted": val_metrics.get("precision_weighted", val_metrics.get("joint_precision_weighted", np.nan)),
        "val_recall_macro": val_metrics.get("recall_macro", val_metrics.get("joint_recall_macro", np.nan)),
        "val_recall_weighted": val_metrics.get("recall_weighted", val_metrics.get("joint_recall_weighted", np.nan)),
        "test_precision_macro": test_metrics.get("precision_macro", test_metrics.get("joint_precision_macro", np.nan)),
        "test_precision_weighted": test_metrics.get("precision_weighted", test_metrics.get("joint_precision_weighted", np.nan)),
        "test_recall_macro": test_metrics.get("recall_macro", test_metrics.get("joint_recall_macro", np.nan)),
        "test_recall_weighted": test_metrics.get("recall_weighted", test_metrics.get("joint_recall_weighted", np.nan)),
        "val_ece": val_metrics.get("ece", np.nan),
        "val_brier": val_metrics.get("brier", np.nan),
        "test_ece": test_metrics.get("ece", np.nan),
        "test_brier": test_metrics.get("brier", np.nan),
        "test_minority_f1": test_metrics.get("class_0_1_f1", np.nan),
        "test_minority_recall": test_metrics.get("class_0_1_recall", np.nan),
        "selected_temperature": val_metrics.get("selected_temperature", np.nan),
        "selected_minority_threshold": val_metrics.get("selected_minority_threshold", np.nan),
        "selected_minority_min_recall": val_metrics.get("selected_minority_min_recall", np.nan),
        "minority_min_recall": cfg.minority_min_recall,
        "minority_min_recall_grid": cfg.minority_min_recall_grid,
        "val_post_acc": val_metrics.get("post_accuracy", val_metrics.get("post_joint_accuracy", np.nan)),
        "val_post_f1": val_metrics.get("post_f1_macro", val_metrics.get("post_joint_f1_macro", np.nan)),
        "val_post_f1_weighted": val_metrics.get("post_f1_weighted", val_metrics.get("post_joint_f1_weighted", np.nan)),
        "val_post_minority_f1": val_metrics.get("post_class_0_1_f1", np.nan),
        "val_post_minority_recall": val_metrics.get("post_class_0_1_recall", np.nan),
        "test_post_acc": test_metrics.get("post_accuracy", test_metrics.get("post_joint_accuracy", np.nan)),
        "test_post_f1": test_metrics.get("post_f1_macro", test_metrics.get("post_joint_f1_macro", np.nan)),
        "test_post_f1_weighted": test_metrics.get("post_f1_weighted", test_metrics.get("post_joint_f1_weighted", np.nan)),
        "val_post_precision_macro": val_metrics.get("post_precision_macro", val_metrics.get("post_joint_precision_macro", np.nan)),
        "val_post_precision_weighted": val_metrics.get("post_precision_weighted", val_metrics.get("post_joint_precision_weighted", np.nan)),
        "val_post_recall_macro": val_metrics.get("post_recall_macro", val_metrics.get("post_joint_recall_macro", np.nan)),
        "val_post_recall_weighted": val_metrics.get("post_recall_weighted", val_metrics.get("post_joint_recall_weighted", np.nan)),
        "test_post_precision_macro": test_metrics.get("post_precision_macro", test_metrics.get("post_joint_precision_macro", np.nan)),
        "test_post_precision_weighted": test_metrics.get("post_precision_weighted", test_metrics.get("post_joint_precision_weighted", np.nan)),
        "test_post_recall_macro": test_metrics.get("post_recall_macro", test_metrics.get("post_joint_recall_macro", np.nan)),
        "test_post_recall_weighted": test_metrics.get("post_recall_weighted", test_metrics.get("post_joint_recall_weighted", np.nan)),
        "test_post_minority_f1": test_metrics.get("post_class_0_1_f1", np.nan),
        "test_post_minority_recall": test_metrics.get("post_class_0_1_recall", np.nan),
        **val_prefixed,
        **test_prefixed,
    }

    # cleanup
    del trainer
    del model
    del tokenizer
    reset_cuda()

    return result


# ----------------------------
# Experiment design
# ----------------------------

def build_smoke_grid(
    smoke_fraction: float = 0.10,
    eval_fraction: Optional[float] = None,
    task_mode: str = "binary_derived",
    num_train_epochs: int = 2,
    use_class_weights: bool = False,
    class_weight_clip: float = 8.0,
    use_temperature_scaling: bool = False,
    minority_threshold_grid: str = "",
    minority_class_name: str = "0_1",
    minority_min_recall: float = 0.0,
    minority_min_recall_grid: str = "",
    chunk_grid: Optional[List[Tuple[int, int]]] = None,
) -> List[RunConfig]:
    grid = []
    effective_chunk_grid = list(chunk_grid) if chunk_grid else [(94, 24), (192, 48), (384, 96), (512, 128)]

    # Strongest first-pass comparisons only
    candidates = [
        # Generative
        dict(mode="lora",  objective="generative",    optimizer_name="adamw",   output_schema="label_only"),
        dict(mode="qlora", objective="generative",    optimizer_name="adamw",   output_schema="label_only"),
        dict(mode="lora",  objective="generative",    optimizer_name="adam8bit", output_schema="label_only"),

        # Discriminative
        dict(mode="lora",  objective="discriminative", optimizer_name="adamw",   output_schema="label_only"),
        dict(mode="qlora", objective="discriminative", optimizer_name="adamw",   output_schema="label_only"),
    ]

    effective_eval_fraction = eval_fraction if eval_fraction is not None else smoke_fraction
    i = 0
    for c in candidates:
        if task_mode == "ordinal_raw" and c["objective"] == "discriminative":
            continue
        for window_size, window_overlap in effective_chunk_grid:
            name_prefix = (
                f"smoke_{i}_sf{frac_tag(smoke_fraction)}_ef{frac_tag(effective_eval_fraction)}_ep{num_train_epochs}"
            )
            grid.append(
                RunConfig(
                    run_name=(
                        f"{name_prefix}_{c['objective']}_{c['mode']}_{c['optimizer_name']}"
                        f"_tm{task_mode}_ws{window_size}_ov{window_overlap}"
                    ),
                    model_name="Qwen/Qwen3-4B-Instruct-2507",
                    task_mode=task_mode,
                    smoke_fraction=smoke_fraction,
                    eval_fraction=effective_eval_fraction,
                    num_train_epochs=num_train_epochs,
                    max_length=512,
                    use_class_weights=use_class_weights,
                    class_weight_clip=class_weight_clip,
                    use_temperature_scaling=use_temperature_scaling,
                    minority_threshold_grid=minority_threshold_grid,
                    minority_class_name=minority_class_name,
                    minority_min_recall=minority_min_recall,
                    minority_min_recall_grid=minority_min_recall_grid,
                    window_size=window_size,
                    window_overlap=window_overlap,
                    **c,
                )
            )
            i += 1
    return grid


def build_full_grid(
    eval_fraction: float = 1.0,
    task_mode: str = "binary_derived",
    train_fraction: float = 1.0,
    num_train_epochs: int = 3,
    use_class_weights: bool = False,
    class_weight_clip: float = 8.0,
    use_temperature_scaling: bool = False,
    minority_threshold_grid: str = "",
    minority_class_name: str = "0_1",
    minority_min_recall: float = 0.0,
    minority_min_recall_grid: str = "",
    chunk_grid: Optional[List[Tuple[int, int]]] = None,
    include_rationale_sweep: bool = False,
) -> List[RunConfig]:
    grid = []
    idx = 0
    effective_chunk_grid = list(chunk_grid) if chunk_grid else [(94, 24), (192, 48), (384, 96), (512, 128)]

    for objective in ["generative", "discriminative"]:
        if task_mode == "ordinal_raw" and objective == "discriminative":
            continue
        for mode in ["lora", "qlora"]:
            for optimizer_name in ["adamw", "adafactor", "adam8bit"]:
                for output_schema in ["label_only", "label_confidence"]:
                    if objective == "discriminative" and output_schema != "label_only":
                        continue

                    for window_size, window_overlap in effective_chunk_grid:
                        name_prefix = (
                            f"full_{idx}_sf{frac_tag(train_fraction)}_ef{frac_tag(eval_fraction)}_ep{num_train_epochs}"
                        )
                        grid.append(
                            RunConfig(
                                run_name=(
                                    f"{name_prefix}_{objective}_{mode}_{optimizer_name}_{output_schema}"
                                    f"_tm{task_mode}_ws{window_size}_ov{window_overlap}"
                                ),
                                model_name="Qwen/Qwen3-4B-Instruct-2507",
                                mode=mode,
                                objective=objective,
                                optimizer_name=optimizer_name,
                                output_schema=output_schema,
                                task_mode=task_mode,
                                smoke_fraction=train_fraction,
                                eval_fraction=eval_fraction,
                                num_train_epochs=num_train_epochs,
                                max_length=768,
                                max_new_tokens_eval=32,
                                use_class_weights=use_class_weights,
                                class_weight_clip=class_weight_clip,
                                use_temperature_scaling=use_temperature_scaling,
                                minority_threshold_grid=minority_threshold_grid,
                                minority_class_name=minority_class_name,
                                minority_min_recall=minority_min_recall,
                                minority_min_recall_grid=minority_min_recall_grid,
                                window_size=window_size,
                                window_overlap=window_overlap,
                            )
                        )
                        idx += 1

    # Optional: add rationale-only objective as an explicit extension of the core matrix.
    if include_rationale_sweep:
        for window_size, window_overlap in effective_chunk_grid:
            name_prefix = f"full_{idx}_sf{frac_tag(train_fraction)}_ef{frac_tag(eval_fraction)}_ep{num_train_epochs}"
            grid.append(
                RunConfig(
                    run_name=(
                        f"{name_prefix}_generative_lora_adamw_label_rationale"
                        f"_tm{task_mode}_ws{window_size}_ov{window_overlap}"
                    ),
                    model_name="Qwen/Qwen3-4B-Instruct-2507",
                    mode="lora",
                    objective="generative",
                    optimizer_name="adamw",
                    output_schema="label_rationale",
                    task_mode=task_mode,
                    smoke_fraction=train_fraction,
                    eval_fraction=eval_fraction,
                    num_train_epochs=num_train_epochs,
                    max_length=768,
                    max_new_tokens_eval=32,
                    use_class_weights=use_class_weights,
                    class_weight_clip=class_weight_clip,
                    use_temperature_scaling=use_temperature_scaling,
                    minority_threshold_grid=minority_threshold_grid,
                    minority_class_name=minority_class_name,
                    minority_min_recall=minority_min_recall,
                    minority_min_recall_grid=minority_min_recall_grid,
                    window_size=window_size,
                    window_overlap=window_overlap,
                )
            )
            idx += 1

    return grid


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen full experiment sweep.")
    parser.add_argument(
        "--stage",
        choices=["smoke", "full", "both"],
        default="full",
        help="Which stage to run."
    )
    parser.add_argument(
        "--smoke-fraction",
        type=float,
        default=None,
        help="Train-data fraction override (applies to selected stage). Defaults: smoke=0.10, full=1.00."
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=None,
        help="Optional validation/test fraction override."
    )
    parser.add_argument(
        "--task-mode",
        choices=["binary_derived", "ordinal_raw", "both"],
        default="binary_derived",
        help="Evaluation/target mode. Use 'both' to run binary_derived and ordinal_raw."
    )
    parser.add_argument("--use-class-weights", action="store_true", help="Use class-weighted CE for discriminative runs.")
    parser.add_argument(
        "--vanilla",
        action="store_true",
        help="Force vanilla evaluation: disables class weights, temperature scaling, and minority-threshold tuning.",
    )
    parser.add_argument("--class-weight-clip", type=float, default=8.0, help="Clip for class weights.")
    parser.add_argument("--use-temperature-scaling", action="store_true", help="Fit temperature on validation logits (discriminative).")
    parser.add_argument(
        "--minority-threshold-grid",
        type=str,
        default="",
        help="Comma-separated threshold candidates in [0,1] for minority override, e.g. '0.15,0.2,0.25'.",
    )
    parser.add_argument(
        "--minority-class-name",
        choices=["0_0", "1_0", "0_1", "1_1"],
        default="0_1",
        help="Minority class for threshold override.",
    )
    parser.add_argument(
        "--minority-min-recall",
        type=float,
        default=0.0,
        help="Constraint for threshold tuning: choose best macro F1 among thresholds with minority recall >= this value.",
    )
    parser.add_argument(
        "--minority-min-recall-grid",
        type=str,
        default="",
        help="Comma-separated recall constraints to try, e.g. '0.2,0.3'. Chooses strictest feasible target automatically.",
    )
    parser.add_argument(
        "--run-name-contains",
        type=str,
        default="",
        help="If set, run only configs whose run_name contains this substring.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="Override epochs for both smoke/full stages (default: smoke=2, full=3).",
    )
    parser.add_argument(
        "--chunk-grid",
        type=str,
        default="94:24,192:48,384:96,512:128",
        help="Comma-separated window:overlap pairs applied uniformly to all configs.",
    )
    parser.add_argument(
        "--include-rationale-sweep",
        action="store_true",
        help="Include extra generative label_rationale runs (excluded by default for clean comparisons).",
    )
    parser.add_argument(
        "--train-path",
        default=os.getenv("TRAIN_PATH", DEFAULT_TRAIN_PATH),
        help="Path to training CSV. Can also be set via TRAIN_PATH env var.",
    )
    parser.add_argument(
        "--test-path",
        default=os.getenv("TEST_PATH", DEFAULT_TEST_PATH),
        help="Path to test CSV. Can also be set via TEST_PATH env var.",
    )
    return parser.parse_args()


# ----------------------------
# Evaluation
# ----------------------------


def compute_metrics_discriminative(eval_pred):
    logits, labels = eval_pred

    preds = np.argmax(logits, axis=-1)

    out = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
    }
    out.update(compute_joint_class_metrics(labels, preds))
    return out

def parse_output(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]
        data = json.loads(json_str)

        anxiety = int(data.get("anxiety", 0))
        depression = int(data.get("depression", 0))

        return anxiety + 2 * depression
    except:
        return 0  # fallback
    

def evaluate_generative(model, tokenizer, dataset, max_samples=50):
    

    model.eval()

    preds = []
    labels = []

    for ex in dataset.select(range(min(max_samples, len(dataset)))):

        prompt = f"""Classify mental health signals.

Return JSON:
{{"anxiety":0/1,"depression":0/1}}

Text:
{ex['value']}
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        pred = parse_output(decoded)
        label = int(ex["anxiety"] + 2 * ex["depression"])

        preds.append(pred)
        labels.append(label)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0)
    }

# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"train CSV not found: {args.train_path}")
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"test CSV not found: {args.test_path}")
    ds = load_data(args.train_path, args.test_path, val_size=0.2, seed=42)

    os.makedirs("./sweep_results", exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    smoke_csv_log = f"./sweep_results/smoke_results_{run_stamp}.csv"
    smoke_jsonl_log = f"./sweep_results/smoke_results_{run_stamp}.jsonl"
    full_csv_log = f"./sweep_results/full_results_{run_stamp}.csv"
    full_jsonl_log = f"./sweep_results/full_results_{run_stamp}.jsonl"
    all_csv_log = f"./sweep_results/all_runs_{run_stamp}.csv"
    all_jsonl_log = f"./sweep_results/all_runs_{run_stamp}.jsonl"
    latest_smoke_csv = "./sweep_results/smoke_results_latest.csv"
    latest_smoke_jsonl = "./sweep_results/smoke_results_latest.jsonl"
    latest_full_csv = "./sweep_results/full_results_latest.csv"
    latest_full_jsonl = "./sweep_results/full_results_latest.jsonl"
    latest_all_csv = "./sweep_results/all_runs_latest.csv"
    latest_all_jsonl = "./sweep_results/all_runs_latest.jsonl"

    # reset "latest" JSONL files for this run
    open(latest_smoke_jsonl, "w", encoding="utf-8").close()
    open(latest_full_jsonl, "w", encoding="utf-8").close()
    open(latest_all_jsonl, "w", encoding="utf-8").close()

    run_smoke = args.stage in {"smoke", "both"}
    run_full = args.stage in {"full", "both"}
    smoke_fraction = args.smoke_fraction if args.smoke_fraction is not None else 0.10
    smoke_eval_fraction = args.eval_fraction if args.eval_fraction is not None else smoke_fraction
    full_train_fraction = args.smoke_fraction if args.smoke_fraction is not None else 1.0
    full_eval_fraction = args.eval_fraction if args.eval_fraction is not None else 1.0
    chunk_grid = parse_chunk_grid(args.chunk_grid)

    task_modes = ["binary_derived", "ordinal_raw"] if args.task_mode == "both" else [args.task_mode]
    effective_use_class_weights = bool(args.use_class_weights)
    effective_use_temperature_scaling = bool(args.use_temperature_scaling)
    effective_minority_threshold_grid = args.minority_threshold_grid
    effective_minority_min_recall = float(args.minority_min_recall)
    effective_minority_min_recall_grid = args.minority_min_recall_grid
    if args.vanilla:
        effective_use_class_weights = False
        effective_use_temperature_scaling = False
        effective_minority_threshold_grid = ""
        effective_minority_min_recall = 0.0
        effective_minority_min_recall_grid = ""
        print("[vanilla] Enabled: disabling class weights, temperature scaling, and minority-threshold tuning.")

    print(
        f"Stage={args.stage} "
        f"(smoke_fraction={smoke_fraction}, "
        f"smoke_eval_fraction={smoke_eval_fraction}, "
        f"full_train_fraction={full_train_fraction}, "
        f"full_eval_fraction={full_eval_fraction}, "
        f"task_modes={task_modes}, "
        f"chunk_grid={chunk_grid}, "
        f"include_rationale_sweep={args.include_rationale_sweep})"
    )

    smoke_results = []
    all_results = []
    if run_smoke:
        # 1) smoke test
        smoke_cfgs = []
        for task_mode in task_modes:
            smoke_cfgs.extend(
                build_smoke_grid(
                    smoke_fraction=smoke_fraction,
                    eval_fraction=smoke_eval_fraction,
                    task_mode=task_mode,
                    num_train_epochs=args.num_train_epochs if args.num_train_epochs is not None else 2,
                    use_class_weights=effective_use_class_weights,
                    class_weight_clip=args.class_weight_clip,
                    use_temperature_scaling=effective_use_temperature_scaling,
                    minority_threshold_grid=effective_minority_threshold_grid,
                    minority_class_name=args.minority_class_name,
                    minority_min_recall=effective_minority_min_recall,
                    minority_min_recall_grid=effective_minority_min_recall_grid,
                    chunk_grid=chunk_grid,
                )
            )
        if args.run_name_contains:
            smoke_cfgs = [c for c in smoke_cfgs if args.run_name_contains in c.run_name]
            print(f"[filter] smoke configs after run_name_contains='{args.run_name_contains}': {len(smoke_cfgs)}")
        for cfg in smoke_cfgs:
            try:
                result = run_one(cfg, ds)
                row = {
                    **result,
                    "status": "ok",
                    "error": "",
                    "stage": "smoke",
                    "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                smoke_results.append(row)
                all_results.append(row)
                append_jsonl(smoke_jsonl_log, row)
                append_jsonl(all_jsonl_log, row)
                append_jsonl(latest_smoke_jsonl, row)
                append_jsonl(latest_all_jsonl, row)
                pd.DataFrame(smoke_results).to_csv(smoke_csv_log, index=False)
                pd.DataFrame(smoke_results).to_csv(latest_smoke_csv, index=False)
                pd.DataFrame(all_results).to_csv(all_csv_log, index=False)
                pd.DataFrame(all_results).to_csv(latest_all_csv, index=False)
                print(json.dumps(row, indent=2))
            except Exception as e:
                row = {
                    **asdict(cfg),
                    "status": "failed",
                    "error": str(e),
                    "stage": "smoke",
                    "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                smoke_results.append(row)
                all_results.append(row)
                append_jsonl(smoke_jsonl_log, row)
                append_jsonl(all_jsonl_log, row)
                append_jsonl(latest_smoke_jsonl, row)
                append_jsonl(latest_all_jsonl, row)
                pd.DataFrame(smoke_results).to_csv(smoke_csv_log, index=False)
                pd.DataFrame(smoke_results).to_csv(latest_smoke_csv, index=False)
                pd.DataFrame(all_results).to_csv(all_csv_log, index=False)
                pd.DataFrame(all_results).to_csv(latest_all_csv, index=False)
                print(f"FAILED: {cfg.run_name} -> {e}")
    else:
        print("Skipping smoke runs.")

    smoke_df = pd.DataFrame(smoke_results)
    smoke_df.to_csv("./sweep_results/smoke_results.csv", index=False)

    # 2) smoke feasibility report
    if "peak_gpu_mem_gb" not in smoke_df.columns:
        smoke_df["peak_gpu_mem_gb"] = np.nan
    if "status" not in smoke_df.columns:
        smoke_df["status"] = "ok"

    feasible = smoke_df[
        (smoke_df["peak_gpu_mem_gb"].fillna(np.inf) < 15.5) &
        (smoke_df["status"] != "failed")
    ].copy()

    print("\nFeasible smoke-test configs:")
    if feasible.empty:
        print("None (all smoke runs failed or exceeded memory threshold).")
    else:
        print(feasible[["run_name", "peak_gpu_mem_gb"]])

    # 3) full run (intentionally keep all configured comparisons)
    full_results = []
    if run_full:
        full_cfgs = []
        for task_mode in task_modes:
            full_cfgs.extend(
                build_full_grid(
                    eval_fraction=full_eval_fraction,
                    task_mode=task_mode,
                    train_fraction=full_train_fraction,
                    num_train_epochs=args.num_train_epochs if args.num_train_epochs is not None else 3,
                    use_class_weights=effective_use_class_weights,
                    class_weight_clip=args.class_weight_clip,
                    use_temperature_scaling=effective_use_temperature_scaling,
                    minority_threshold_grid=effective_minority_threshold_grid,
                    minority_class_name=args.minority_class_name,
                    minority_min_recall=effective_minority_min_recall,
                    minority_min_recall_grid=effective_minority_min_recall_grid,
                    chunk_grid=chunk_grid,
                    include_rationale_sweep=args.include_rationale_sweep,
                )
            )
        if args.run_name_contains:
            full_cfgs = [c for c in full_cfgs if args.run_name_contains in c.run_name]
            print(f"[filter] full configs after run_name_contains='{args.run_name_contains}': {len(full_cfgs)}")
        for cfg in full_cfgs:
            try:
                result = run_one(cfg, ds)
                row = {
                    **result,
                    "status": "ok",
                    "error": "",
                    "stage": "full",
                    "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                full_results.append(row)
                all_results.append(row)
                append_jsonl(full_jsonl_log, row)
                append_jsonl(all_jsonl_log, row)
                append_jsonl(latest_full_jsonl, row)
                append_jsonl(latest_all_jsonl, row)
                pd.DataFrame(full_results).to_csv(full_csv_log, index=False)
                pd.DataFrame(full_results).to_csv(latest_full_csv, index=False)
                pd.DataFrame(all_results).to_csv(all_csv_log, index=False)
                pd.DataFrame(all_results).to_csv(latest_all_csv, index=False)
                print(json.dumps(row, indent=2))
            except Exception as e:
                row = {
                    **asdict(cfg),
                    "status": "failed",
                    "error": str(e),
                    "stage": "full",
                    "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                full_results.append(row)
                all_results.append(row)
                append_jsonl(full_jsonl_log, row)
                append_jsonl(all_jsonl_log, row)
                append_jsonl(latest_full_jsonl, row)
                append_jsonl(latest_all_jsonl, row)
                pd.DataFrame(full_results).to_csv(full_csv_log, index=False)
                pd.DataFrame(full_results).to_csv(latest_full_csv, index=False)
                pd.DataFrame(all_results).to_csv(all_csv_log, index=False)
                pd.DataFrame(all_results).to_csv(latest_all_csv, index=False)
                print(f"FAILED: {cfg.run_name} -> {e}")
    else:
        print("Skipping full runs.")

    full_df = pd.DataFrame(full_results)
    full_df.to_csv("./sweep_results/full_results.csv", index=False)

    # rank by robust metric
    metric_col = "joint_f1_macro" if "joint_f1_macro" in full_df.columns else "joint_exact_match"
    if metric_col in full_df.columns:
        ranked = full_df.sort_values(metric_col, ascending=False)
        ranked.to_csv("./sweep_results/full_results_ranked.csv", index=False)
        print("\nTop runs:")
        print(ranked.head(10)[["run_name", metric_col, "peak_gpu_mem_gb", "train_runtime_sec"]])


if __name__ == "__main__":
    main()
