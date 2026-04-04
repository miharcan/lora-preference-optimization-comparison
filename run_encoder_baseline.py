import os
import gc
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

DEFAULT_TRAIN_PATH = "/mnt/3d56a808-6937-48b6-ba29-c2b68a3e2d94/datasets/train_dataset.csvV4_per_user"
DEFAULT_TEST_PATH = "/mnt/3d56a808-6937-48b6-ba29-c2b68a3e2d94/datasets/test_dataset.csvV4_per_user"


JOINT_CLASS_LABELS = [0, 1, 2, 3]
JOINT_CLASS_NAMES = {0: "0_0", 1: "1_0", 2: "0_1", 3: "1_1"}


@dataclass
class BertRunConfig:
    run_name: str
    model_name: str = "bert-base-uncased"
    model_label: str = "bert_base_uncased"
    mode: str = "bert"
    objective: str = "discriminative"
    optimizer_name: str = "adamw_torch"
    output_schema: str = "joint_label"
    task_mode: str = "binary_derived"
    max_length: int = 512
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    use_class_weights: bool = False
    class_weight_clip: float = 10.0
    use_temperature_scaling: bool = False
    minority_threshold_grid: str = ""
    minority_class_name: str = "0_1"
    minority_min_recall: float = 0.0
    minority_min_recall_grid: str = ""
    seed: int = 42
    bf16: bool = True
    smoke_fraction: float = 0.1
    eval_fraction: float = 0.1
    window_size: int = 384
    window_overlap: int = 96
    output_dir_root: str = "./bert_results"


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


def make_model_label(model_name: str) -> str:
    label = model_name.strip().lower().replace("/", "_").replace("-", "_").replace(".", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or "model"


def parse_float_grid(spec: str) -> List[float]:
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    out = []
    for token in spec.split(","):
        t = token.strip()
        if not t:
            continue
        v = float(t)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Threshold value out of [0,1]: {v}")
        out.append(v)
    return sorted(set(out))


def parse_chunk_grid(spec: str) -> List[tuple[int, int]]:
    if spec is None or not spec.strip():
        return [(96, 24), (192, 48), (384, 96), (512, 128), (640, 128)]
    out = []
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


def fit_temperature_grid(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    candidates = np.concatenate(
        [
            np.arange(0.5, 1.05, 0.05),
            np.arange(1.1, 3.05, 0.1),
        ]
    )
    best_t = 1.0
    best_nll = float("inf")
    for t in candidates:
        probs = softmax_np(val_logits / t)
        p = np.clip(probs[np.arange(len(val_labels)), val_labels], 1e-12, 1.0)
        nll = float(-np.mean(np.log(p)))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def apply_minority_threshold(probs: np.ndarray, base_preds: np.ndarray, minority_idx: int, threshold: float) -> np.ndarray:
    preds = base_preds.copy()
    preds[probs[:, minority_idx] >= threshold] = minority_idx
    return preds


def check_model_available(model_name: str) -> str:
    try:
        _ = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        return ""
    except Exception as e:
        return str(e)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = {"value", "GAD_1", "GAD_2", "PHQ_1", "PHQ_2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["value"] = df["value"].fillna("").astype(str)
    df["anxiety"] = ((df["GAD_1"] > 0) | (df["GAD_2"] > 0)).astype(int)
    df["depression"] = ((df["PHQ_1"] > 0) | (df["PHQ_2"] > 0)).astype(int)
    df["joint_label"] = df["anxiety"] + 2 * df["depression"]
    df["label_combo"] = df["anxiety"].astype(str) + "_" + df["depression"].astype(str)
    return df


def load_data(train_path: str, test_path: str, val_size: float = 0.2, seed: int = 42) -> DatasetDict:
    train_df = preprocess_dataframe(pd.read_csv(train_path))
    test_df = preprocess_dataframe(pd.read_csv(test_path))

    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["label_combo"],
        random_state=seed,
    )
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_split.reset_index(drop=True), preserve_index=False),
            "validation": Dataset.from_pandas(val_split.reset_index(drop=True), preserve_index=False),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True), preserve_index=False),
        }
    )


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
        raise ValueError(f"window_overlap ({overlap}) must be smaller than window_size ({window_size})")
    if len(words) <= window_size:
        return [" ".join(words)]

    step = window_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        part = words[start : start + window_size]
        if not part:
            break
        chunks.append(" ".join(part))
        if start + window_size >= len(words):
            break
    return chunks


def expand_dataset_with_chunks(ds: DatasetDict, cfg: BertRunConfig) -> DatasetDict:
    out = {}
    for split in ["train", "validation", "test"]:
        rows = []
        for idx, ex in enumerate(ds[split]):
            chunks = chunk_text_words(ex.get("value", ""), cfg.window_size, cfg.window_overlap)
            for chunk_id, chunk in enumerate(chunks):
                row = dict(ex)
                row["value"] = chunk
                row["__example_id"] = idx
                row["__chunk_id"] = chunk_id
                rows.append(row)
        out[split] = Dataset.from_list(rows)
    return DatasetDict(out)


def compute_joint_class_metrics(y_true, y_pred):
    _, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=JOINT_CLASS_LABELS,
        zero_division=0,
    )
    out = {}
    for idx, label in enumerate(JOINT_CLASS_LABELS):
        name = JOINT_CLASS_NAMES[label]
        out[f"class_{name}_f1"] = float(f1[idx])
        out[f"class_{name}_recall"] = float(recall[idx])
        out[f"class_{name}_support"] = int(support[idx])
    return out


def evaluate_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    preds = np.argmax(logits, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "joint_exact_match": float(np.mean(preds == labels)),
    }
    metrics.update(compute_joint_class_metrics(labels, preds))
    return metrics


def evaluate_preds(preds: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "joint_exact_match": float(np.mean(preds == labels)),
    }
    metrics.update(compute_joint_class_metrics(labels, preds))
    return metrics


def evaluate_secondary_binary_from_joint(
    y_true_joint: np.ndarray,
    y_pred_joint: np.ndarray,
    joint_probs: np.ndarray | None = None,
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


def compute_class_weights(train_labels: np.ndarray, n_classes: int = 4, clip: float = 10.0) -> torch.Tensor:
    counts = np.bincount(train_labels, minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    n = float(np.sum(counts))
    weights = n / (n_classes * counts)
    weights = weights / np.mean(weights)
    if clip is not None and clip > 0:
        weights = np.clip(weights, 0.0, clip)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
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
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def run_one(cfg: BertRunConfig, full_ds: DatasetDict) -> Dict[str, Any]:
    if cfg.task_mode != "binary_derived":
        raise ValueError("BERT baseline currently supports only task_mode=binary_derived.")

    set_seed(cfg.seed)
    reset_cuda()
    start = time.time()

    ds = DatasetDict(
        {
            "train": subsample_split(full_ds["train"], cfg.smoke_fraction, cfg.seed),
            "validation": subsample_split(full_ds["validation"], cfg.eval_fraction, cfg.seed + 1),
            "test": subsample_split(full_ds["test"], cfg.eval_fraction, cfg.seed + 2),
        }
    )
    ds = expand_dataset_with_chunks(ds, cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=4,
        problem_type="single_label_classification",
    )

    def tokenize_fn(ex):
        out = tokenizer(
            ex["value"],
            truncation=True,
            max_length=cfg.max_length,
        )
        out["labels"] = int(ex["joint_label"])
        return out

    tokenized = DatasetDict(
        {
            split: ds[split].map(tokenize_fn, remove_columns=ds[split].column_names)
            for split in ["train", "validation", "test"]
        }
    )

    out_dir = os.path.join(cfg.output_dir_root, cfg.run_name)
    total_steps = max(
        1,
        int(
            np.ceil(len(tokenized["train"]) / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps))
            * cfg.num_train_epochs
        ),
    )
    warmup_steps = int(np.ceil(total_steps * cfg.warmup_ratio))

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        bf16=cfg.bf16 and torch.cuda.is_available(),
        fp16=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
    )

    class_weights = None
    if cfg.use_class_weights:
        train_labels = np.array(tokenized["train"]["labels"], dtype=int)
        class_weights = compute_class_weights(train_labels, n_classes=4, clip=cfg.class_weight_clip)
        print(f"[run:{cfg.run_name}] class_weights={class_weights.tolist()}")

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        class_weights=class_weights,
    )

    print(
        f"[run:{cfg.run_name}] model={cfg.model_name} "
        f"train={len(tokenized['train'])} val={len(tokenized['validation'])} test={len(tokenized['test'])} "
        f"window={cfg.window_size} overlap={cfg.window_overlap}"
    )
    train_out = trainer.train()

    val_out = trainer.predict(tokenized["validation"])
    test_out = trainer.predict(tokenized["test"])
    val_metrics = evaluate_logits(val_out.predictions, val_out.label_ids)
    test_metrics = evaluate_logits(test_out.predictions, test_out.label_ids)

    selected_temperature = 1.0
    selected_minority_threshold = 0.0
    selected_minority_min_recall = float(cfg.minority_min_recall)
    val_post_metrics = dict(val_metrics)
    test_post_metrics = dict(test_metrics)

    threshold_grid = parse_float_grid(cfg.minority_threshold_grid)
    minority_idx = {v: k for k, v in JOINT_CLASS_NAMES.items()}.get(cfg.minority_class_name, 2)
    if cfg.use_temperature_scaling:
        selected_temperature = fit_temperature_grid(val_out.predictions, val_out.label_ids)

    val_probs = softmax_np(val_out.predictions / selected_temperature)
    test_probs = softmax_np(test_out.predictions / selected_temperature)
    val_base_preds = np.argmax(val_probs, axis=1)
    test_base_preds = np.argmax(test_probs, axis=1)
    val_post_preds = val_base_preds.copy()
    test_post_preds = test_base_preds.copy()
    val_cal_metrics = calibration_metrics_from_probs(val_probs, val_out.label_ids, num_classes=4)
    test_cal_metrics = calibration_metrics_from_probs(test_probs, test_out.label_ids, num_classes=4)
    val_secondary_metrics = evaluate_secondary_binary_from_joint(val_out.label_ids, val_base_preds, val_probs, True)
    test_secondary_metrics = evaluate_secondary_binary_from_joint(test_out.label_ids, test_base_preds, test_probs, True)
    val_post_secondary_metrics = dict(val_secondary_metrics)
    test_post_secondary_metrics = dict(test_secondary_metrics)

    if threshold_grid:
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
                    m = evaluate_preds(val_preds_t, val_out.label_ids)
                    score = float(m["f1_macro"])
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

        val_post_preds = apply_minority_threshold(val_probs, val_base_preds, minority_idx, selected_minority_threshold)
        test_post_preds = apply_minority_threshold(test_probs, test_base_preds, minority_idx, selected_minority_threshold)
        val_post_metrics = evaluate_preds(val_post_preds, val_out.label_ids)
        test_post_metrics = evaluate_preds(test_post_preds, test_out.label_ids)
        val_post_secondary_metrics = evaluate_secondary_binary_from_joint(
            val_out.label_ids, val_post_preds, None, False
        )
        test_post_secondary_metrics = evaluate_secondary_binary_from_joint(
            test_out.label_ids, test_post_preds, None, False
        )

    elapsed = time.time() - start
    peak_mem = peak_gpu_mem_gb()
    row = {
        **asdict(cfg),
        "train_runtime_sec": elapsed,
        "train_loss": float(train_out.training_loss) if hasattr(train_out, "training_loss") else np.nan,
        "peak_gpu_mem_gb": peak_mem,
        "val_acc": val_metrics["accuracy"],
        "val_f1": val_metrics["f1_macro"],
        "val_f1_weighted": val_metrics["f1_weighted"],
        "val_minority_f1": val_metrics.get("class_0_1_f1", np.nan),
        "val_minority_recall": val_metrics.get("class_0_1_recall", np.nan),
        "test_acc": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_macro"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "val_precision_macro": val_metrics["precision_macro"],
        "val_precision_weighted": val_metrics["precision_weighted"],
        "val_recall_macro": val_metrics["recall_macro"],
        "val_recall_weighted": val_metrics["recall_weighted"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_precision_weighted": test_metrics["precision_weighted"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_recall_weighted": test_metrics["recall_weighted"],
        "val_ece": val_cal_metrics["ece"],
        "val_brier": val_cal_metrics["brier"],
        "test_ece": test_cal_metrics["ece"],
        "test_brier": test_cal_metrics["brier"],
        "test_minority_f1": test_metrics.get("class_0_1_f1", np.nan),
        "test_minority_recall": test_metrics.get("class_0_1_recall", np.nan),
        "val_valid_json_rate": np.nan,
        "test_valid_json_rate": np.nan,
        "selected_temperature": selected_temperature,
        "selected_minority_threshold": selected_minority_threshold,
        "selected_minority_min_recall": selected_minority_min_recall,
        "minority_min_recall": cfg.minority_min_recall,
        "minority_min_recall_grid": cfg.minority_min_recall_grid,
        "val_post_acc": val_post_metrics["accuracy"],
        "val_post_f1": val_post_metrics["f1_macro"],
        "val_post_f1_weighted": val_post_metrics["f1_weighted"],
        "val_post_minority_f1": val_post_metrics.get("class_0_1_f1", np.nan),
        "val_post_minority_recall": val_post_metrics.get("class_0_1_recall", np.nan),
        "test_post_acc": test_post_metrics["accuracy"],
        "test_post_f1": test_post_metrics["f1_macro"],
        "test_post_f1_weighted": test_post_metrics["f1_weighted"],
        "val_post_precision_macro": val_post_metrics["precision_macro"],
        "val_post_precision_weighted": val_post_metrics["precision_weighted"],
        "val_post_recall_macro": val_post_metrics["recall_macro"],
        "val_post_recall_weighted": val_post_metrics["recall_weighted"],
        "test_post_precision_macro": test_post_metrics["precision_macro"],
        "test_post_precision_weighted": test_post_metrics["precision_weighted"],
        "test_post_recall_macro": test_post_metrics["recall_macro"],
        "test_post_recall_weighted": test_post_metrics["recall_weighted"],
        "test_post_minority_f1": test_post_metrics.get("class_0_1_f1", np.nan),
        "test_post_minority_recall": test_post_metrics.get("class_0_1_recall", np.nan),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"val_{k}": v for k, v in val_secondary_metrics.items()},
        **{f"test_{k}": v for k, v in test_secondary_metrics.items()},
        **{f"val_post_{k}": v for k, v in val_post_metrics.items()},
        **{f"test_post_{k}": v for k, v in test_post_metrics.items()},
        **{f"val_post_{k}": v for k, v in val_post_secondary_metrics.items()},
        **{f"test_post_{k}": v for k, v in test_post_secondary_metrics.items()},
    }

    del trainer
    del model
    del tokenizer
    reset_cuda()
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="BERT baseline experiment")
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--task-mode", choices=["binary_derived"], default="binary_derived")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument(
        "--model-names",
        default=None,
        help="Comma-separated models, e.g. 'bert-base-uncased,roberta-base,microsoft/deberta-v3-base'",
    )
    parser.add_argument("--smoke-fraction", type=float, default=None)
    parser.add_argument("--eval-fraction", type=float, default=None)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--class-weight-clip", type=float, default=10.0)
    parser.add_argument("--use-temperature-scaling", action="store_true")
    parser.add_argument(
        "--minority-threshold-grid",
        default="",
        help="Comma-separated thresholds in [0,1], e.g. '0.2,0.3,0.4'. Empty disables threshold tuning.",
    )
    parser.add_argument("--minority-class-name", default="0_1", choices=["0_0", "1_0", "0_1", "1_1"])
    parser.add_argument(
        "--minority-min-recall",
        type=float,
        default=0.0,
        help="Constraint for threshold tuning: choose best macro F1 among thresholds with minority recall >= this value.",
    )
    parser.add_argument(
        "--minority-min-recall-grid",
        default="",
        help="Comma-separated recall constraints to try, e.g. '0.2,0.3'. Chooses strictest feasible target automatically.",
    )
    parser.add_argument(
        "--chunk-grid",
        default="94:24,128:32,192:48,256:64,384:96,512:128",
        help="Comma-separated window:overlap pairs, e.g. '94:24,192:48,384:96'",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="Override epochs (default: smoke=2, full=3).",
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


def main():
    args = parse_args()
    if args.stage == "smoke":
        smoke_fraction = args.smoke_fraction if args.smoke_fraction is not None else 0.10
        eval_fraction = args.eval_fraction if args.eval_fraction is not None else smoke_fraction
        epochs = 2
    else:
        smoke_fraction = args.smoke_fraction if args.smoke_fraction is not None else 1.0
        eval_fraction = args.eval_fraction if args.eval_fraction is not None else 1.0
        epochs = 3
    if args.num_train_epochs is not None:
        epochs = int(args.num_train_epochs)

    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"train CSV not found: {args.train_path}")
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"test CSV not found: {args.test_path}")
    ds = load_data(args.train_path, args.test_path, val_size=0.2, seed=42)

    os.makedirs("./bert_results", exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_log = f"./bert_results/bert_results_{run_stamp}.csv"
    jsonl_log = f"./bert_results/bert_results_{run_stamp}.jsonl"
    latest_csv = "./bert_results/bert_results_latest.csv"
    latest_jsonl = "./bert_results/bert_results_latest.jsonl"
    open(latest_jsonl, "w", encoding="utf-8").close()

    chunk_grid = parse_chunk_grid(args.chunk_grid)
    if args.model_names:
        model_names = [m.strip() for m in args.model_names.split(",") if m.strip()]
    else:
        model_names = [args.model_name.strip()]
    if not model_names:
        raise ValueError("No model names provided.")

    runs = []
    skipped_models: Dict[str, str] = {}
    idx = 0
    for model_name in model_names:
        model_error = check_model_available(model_name)
        if model_error:
            skipped_models[model_name] = model_error
            print(f"SKIP model={model_name} -> tokenizer/model init failed: {model_error}")
            continue
        model_label = make_model_label(model_name)
        for ws, ov in chunk_grid:
            name_prefix = (
                f"{args.stage}_{idx}_sf{frac_tag(smoke_fraction)}_ef{frac_tag(eval_fraction)}_ep{epochs}"
            )
            runs.append(
                BertRunConfig(
                    run_name=f"{name_prefix}_{model_label}_tmbinary_derived_ws{ws}_ov{ov}",
                    model_name=model_name,
                    model_label=model_label,
                    smoke_fraction=smoke_fraction,
                    eval_fraction=eval_fraction,
                    num_train_epochs=epochs,
                    task_mode=args.task_mode,
                    use_class_weights=args.use_class_weights,
                    class_weight_clip=args.class_weight_clip,
                    use_temperature_scaling=args.use_temperature_scaling,
                    minority_threshold_grid=args.minority_threshold_grid,
                    minority_class_name=args.minority_class_name,
                    minority_min_recall=args.minority_min_recall,
                    minority_min_recall_grid=args.minority_min_recall_grid,
                    window_size=ws,
                    window_overlap=ov,
                )
            )
            idx += 1

    print(
        f"Running BERT sweep: stage={args.stage}, task_mode={args.task_mode}, models={model_names}, n_runs={len(runs)}"
    )

    results = []
    for model_name, err in skipped_models.items():
        model_label = make_model_label(model_name)
        for ws, ov in chunk_grid:
            name_prefix = (
                f"{args.stage}_{idx}_sf{frac_tag(smoke_fraction)}_ef{frac_tag(eval_fraction)}_ep{epochs}"
            )
            row = {
                **asdict(
                    BertRunConfig(
                        run_name=f"{name_prefix}_{model_label}_tmbinary_derived_ws{ws}_ov{ov}",
                        model_name=model_name,
                        model_label=model_label,
                        smoke_fraction=smoke_fraction,
                        eval_fraction=eval_fraction,
                        num_train_epochs=epochs,
                        task_mode=args.task_mode,
                        use_class_weights=args.use_class_weights,
                        class_weight_clip=args.class_weight_clip,
                        use_temperature_scaling=args.use_temperature_scaling,
                        minority_threshold_grid=args.minority_threshold_grid,
                        minority_class_name=args.minority_class_name,
                        minority_min_recall=args.minority_min_recall,
                        minority_min_recall_grid=args.minority_min_recall_grid,
                        window_size=ws,
                        window_overlap=ov,
                    )
                ),
                "status": "failed",
                "error": f"preflight_tokenizer_failed: {err}",
                "stage": args.stage,
                "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            idx += 1
            results.append(row)
            append_jsonl(jsonl_log, row)
            append_jsonl(latest_jsonl, row)
    if results:
        pd.DataFrame(results).to_csv(csv_log, index=False)
        pd.DataFrame(results).to_csv(latest_csv, index=False)

    for cfg in runs:
        try:
            row = run_one(cfg, ds)
            row["status"] = "ok"
            row["error"] = ""
            row["stage"] = args.stage
            row["logged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            results.append(row)
            append_jsonl(jsonl_log, row)
            append_jsonl(latest_jsonl, row)
            pd.DataFrame(results).to_csv(csv_log, index=False)
            pd.DataFrame(results).to_csv(latest_csv, index=False)
            print(json.dumps({"run_name": cfg.run_name, "status": "ok", "test_f1": row.get("test_f1")}, indent=2))
        except Exception as e:
            row = {
                **asdict(cfg),
                "status": "failed",
                "error": str(e),
                "stage": args.stage,
                "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            results.append(row)
            append_jsonl(jsonl_log, row)
            append_jsonl(latest_jsonl, row)
            pd.DataFrame(results).to_csv(csv_log, index=False)
            pd.DataFrame(results).to_csv(latest_csv, index=False)
            print(f"FAILED: {cfg.run_name} -> {e}")

    print("Done.")


if __name__ == "__main__":
    main()
