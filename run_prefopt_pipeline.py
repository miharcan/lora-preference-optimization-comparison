import os
import gc
import json
import time
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer
from trl.experimental.orpo import ORPOConfig, ORPOTrainer
from trl.experimental.kto import KTOConfig, KTOTrainer

DEFAULT_TRAIN_PATH = "/mnt/3d56a808-6937-48b6-ba29-c2b68a3e2d94/datasets/train_dataset.csvV4_per_user"
DEFAULT_TEST_PATH = "/mnt/3d56a808-6937-48b6-ba29-c2b68a3e2d94/datasets/test_dataset.csvV4_per_user"


@dataclass
class PrefRunConfig:
    run_name: str
    method: Literal["dpo", "orpo", "kto"] = "dpo"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    mode: Literal["lora", "qlora"] = "lora"
    task_mode: Literal["binary_derived", "ordinal_raw"] = "binary_derived"

    max_length: int = 512
    max_new_tokens_eval: int = 64
    gen_eval_batch_size: int = 8

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    seed: int = 42
    bf16: bool = True
    smoke_fraction: float = 0.10
    eval_fraction: float = 0.10
    window_size: int = 512
    window_overlap: int = 128
    rebalance_train: bool = False
    rebalance_strategy: Literal["none", "sqrt", "max"] = "sqrt"

    output_dir_root: str = "./pref_results"


JOINT_CLASS_LABELS = [0, 1, 2, 3]
JOINT_CLASS_NAMES = {0: "0_0", 1: "1_0", 2: "0_1", 3: "1_1"}


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


def parse_chunk_grid(spec: str) -> List[Tuple[int, int]]:
    if spec is None or not spec.strip():
        return [(384, 96), (512, 128), (640, 128)]
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


def build_trl_args(config_cls, kwargs: Dict[str, Any]):
    sig = inspect.signature(config_cls.__init__)
    params = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return config_cls(**filtered)


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
    df["label_combo"] = df["anxiety"].astype(str) + "_" + df["depression"].astype(str)
    return df


def load_data(train_path: str, test_path: str, val_size=0.2, seed=42) -> DatasetDict:
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


def expand_dataset_with_chunks(ds: DatasetDict, cfg: PrefRunConfig) -> DatasetDict:
    out = {}
    for split in ["train", "validation", "test"]:
        rows: List[Dict[str, Any]] = []
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


def build_prompt(ex: Dict[str, Any], cfg: PrefRunConfig) -> str:
    if cfg.task_mode == "binary_derived":
        schema = '{"anxiety":0/1,"depression":0/1}'
    else:
        schema = '{"GAD_1":0..3,"GAD_2":0..3,"PHQ_1":0..3,"PHQ_2":0..3}'
    return (
        "Classify mental health signals.\n\n"
        f"Return JSON:\n{schema}\n\n"
        f"Text:\n{ex['value']}\n"
    )


def build_target(ex: Dict[str, Any], cfg: PrefRunConfig) -> Dict[str, int]:
    if cfg.task_mode == "binary_derived":
        return {"anxiety": int(ex["anxiety"]), "depression": int(ex["depression"])}
    return {
        "GAD_1": int(ex["GAD_1"]),
        "GAD_2": int(ex["GAD_2"]),
        "PHQ_1": int(ex["PHQ_1"]),
        "PHQ_2": int(ex["PHQ_2"]),
    }


def build_rejected_target(ex: Dict[str, Any], cfg: PrefRunConfig) -> Dict[str, int]:
    if cfg.task_mode == "binary_derived":
        gold = int(ex["anxiety"]) + 2 * int(ex["depression"])
        alternatives = [i for i in JOINT_CLASS_LABELS if i != gold]
        # deterministic but varied mapping
        rej = alternatives[(int(ex.get("__chunk_id", 0)) + int(ex.get("__example_id", 0))) % len(alternatives)]
        return {"anxiety": 1 if rej in [1, 3] else 0, "depression": 1 if rej in [2, 3] else 0}

    keys = ["GAD_1", "GAD_2", "PHQ_1", "PHQ_2"]
    tgt = build_target(ex, cfg)
    key = keys[(int(ex.get("__chunk_id", 0)) + int(ex.get("__example_id", 0))) % len(keys)]
    tgt[key] = int((tgt[key] + 1) % 4)
    return tgt


def build_preference_dataset(split_ds: Dataset, cfg: PrefRunConfig) -> Dataset:
    rows = []
    for ex in split_ds:
        prompt = build_prompt(ex, cfg)
        chosen = json.dumps(build_target(ex, cfg), separators=(",", ":"))
        rejected = json.dumps(build_rejected_target(ex, cfg), separators=(",", ":"))
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(rows)


def rebalance_binary_joint_dataset(split_ds: Dataset, cfg: PrefRunConfig) -> Dataset:
    if cfg.task_mode != "binary_derived" or cfg.rebalance_strategy == "none":
        return split_ds

    rows = [dict(ex) for ex in split_ds]
    if not rows:
        return split_ds

    buckets: Dict[int, List[Dict[str, Any]]] = {k: [] for k in JOINT_CLASS_LABELS}
    for ex in rows:
        label = int(ex["anxiety"]) + 2 * int(ex["depression"])
        buckets[label].append(ex)

    counts = {k: len(v) for k, v in buckets.items()}
    max_count = max(counts.values()) if counts else 0
    rng = random.Random(cfg.seed)
    balanced: List[Dict[str, Any]] = []
    for label in JOINT_CLASS_LABELS:
        pool = buckets[label]
        n = len(pool)
        if n == 0:
            continue
        if cfg.rebalance_strategy == "max":
            target = max_count
        else:
            target = int(np.ceil(np.sqrt(n * max_count)))
        target = max(n, target)
        if target <= n:
            picked = rng.sample(pool, target)
        else:
            picked = [pool[i % n] for i in range(target)]
            rng.shuffle(picked)
        balanced.extend(picked)

    rng.shuffle(balanced)
    print(f"[rebalance:{cfg.run_name}] strategy={cfg.rebalance_strategy} before={counts} after={len(balanced)}")
    return Dataset.from_list(balanced)


def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model_and_tokenizer(cfg: PrefRunConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if cfg.mode == "qlora":
        kwargs["quantization_config"] = make_bnb_config()
    else:
        kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    if cfg.mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def parse_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


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


def evaluate_secondary_binary_from_joint(y_true_joint, y_pred_joint) -> Dict[str, Any]:
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
    out["anxiety_ece"] = np.nan
    out["anxiety_brier"] = np.nan
    out["depression_ece"] = np.nan
    out["depression_brier"] = np.nan
    return out


@torch.inference_mode()
def eval_generative(model, tokenizer, ds: Dataset, cfg: PrefRunConfig) -> Dict[str, Any]:
    n = len(ds)
    if n == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_macro": 0.0,
            "recall_weighted": 0.0,
            "ece": np.nan,
            "brier": np.nan,
            "valid_json_rate": 0.0,
        }

    valid_json = 0
    raw_keys = ["GAD_1", "GAD_2", "PHQ_1", "PHQ_2"]
    if cfg.task_mode == "binary_derived":
        y_true, y_pred = [], []
    else:
        gold = {k: [] for k in raw_keys}
        pred = {k: [] for k in raw_keys}

    model.eval()
    batch_size = max(1, int(cfg.gen_eval_batch_size))
    for start_idx in range(0, n, batch_size):
        batch = [ds[i] for i in range(start_idx, min(start_idx + batch_size, n))]
        prompt_texts = [build_prompt(ex, cfg) for ex in batch]
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
            gen = tokenizer.decode(outputs[i][int(input_lens[i]) :], skip_special_tokens=True)
            parsed = parse_json(gen)
            if cfg.task_mode == "binary_derived":
                g = int(ex["anxiety"]) + 2 * int(ex["depression"])
                y_true.append(g)
                if parsed is not None and "anxiety" in parsed and "depression" in parsed:
                    valid_json += 1
                    pa = 1 if int(parsed["anxiety"]) > 0 else 0
                    pd_ = 1 if int(parsed["depression"]) > 0 else 0
                else:
                    pa, pd_ = 0, 0
                y_pred.append(pa + 2 * pd_)
            else:
                for k in raw_keys:
                    gold[k].append(int(ex[k]))
                if parsed is not None and all(k in parsed for k in raw_keys):
                    valid_json += 1
                    for k in raw_keys:
                        v = int(parsed[k])
                        pred[k].append(max(0, min(3, v)))
                else:
                    for k in raw_keys:
                        pred[k].append(0)

    if cfg.task_mode == "binary_derived":
        y_true_np = np.asarray(y_true, dtype=int)
        y_pred_np = np.asarray(y_pred, dtype=int)
        out = {
            "valid_json_rate": valid_json / max(1, n),
            "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
            "f1_macro": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
            "recall_macro": float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
            "recall_weighted": float(recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
            "ece": np.nan,
            "brier": np.nan,
            "joint_exact_match": float(np.mean(y_true_np == y_pred_np)),
        }
        out.update(compute_joint_class_metrics(y_true_np, y_pred_np))
        out.update(evaluate_secondary_binary_from_joint(y_true_np, y_pred_np))
        return out

    per_key_f1 = [f1_score(gold[k], pred[k], average="macro", zero_division=0) for k in raw_keys]
    per_key_f1_weighted = [f1_score(gold[k], pred[k], average="weighted", zero_division=0) for k in raw_keys]
    per_key_precision = [precision_score(gold[k], pred[k], average="macro", zero_division=0) for k in raw_keys]
    per_key_precision_weighted = [precision_score(gold[k], pred[k], average="weighted", zero_division=0) for k in raw_keys]
    per_key_recall = [recall_score(gold[k], pred[k], average="macro", zero_division=0) for k in raw_keys]
    per_key_recall_weighted = [recall_score(gold[k], pred[k], average="weighted", zero_division=0) for k in raw_keys]
    per_key_accuracy = [accuracy_score(gold[k], pred[k]) for k in raw_keys]
    return {
        "valid_json_rate": valid_json / max(1, n),
        "accuracy": float(np.mean(per_key_accuracy)),
        "f1_macro": float(np.mean(per_key_f1)),
        "f1_weighted": float(np.mean(per_key_f1_weighted)),
        "precision_macro": float(np.mean(per_key_precision)),
        "precision_weighted": float(np.mean(per_key_precision_weighted)),
        "recall_macro": float(np.mean(per_key_recall)),
        "recall_weighted": float(np.mean(per_key_recall_weighted)),
        "ece": np.nan,
        "brier": np.nan,
        "raw_f1_macro_mean": float(np.mean(per_key_f1)),
        "raw_recall_macro_mean": float(np.mean(per_key_recall)),
        "raw_joint_exact_match": float(np.mean([all(gold[k][i] == pred[k][i] for k in raw_keys) for i in range(n)])),
    }


def make_trainer(cfg: PrefRunConfig, model, tokenizer, train_pref: Dataset, val_pref: Dataset):
    trainer_bs = max(2, cfg.per_device_train_batch_size) if cfg.method == "kto" else cfg.per_device_train_batch_size
    effective_batch = max(1, trainer_bs * max(1, cfg.gradient_accumulation_steps))
    steps_per_epoch = max(1, int(np.ceil(len(train_pref) / effective_batch)))
    total_steps = max(1, int(steps_per_epoch * cfg.num_train_epochs))
    warmup_steps = max(1, int(np.ceil(total_steps * cfg.warmup_fraction)))

    out_dir = os.path.join(cfg.output_dir_root, cfg.run_name)
    common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=trainer_bs,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=cfg.bf16,
        fp16=not cfg.bf16,
        remove_unused_columns=False,
        max_length=cfg.max_length,
    )

    if cfg.method == "dpo":
        args = build_trl_args(DPOConfig, common)
        return DPOTrainer(
            model=model,
            args=args,
            train_dataset=train_pref,
            eval_dataset=val_pref,
            processing_class=tokenizer,
        )
    if cfg.method == "orpo":
        args = build_trl_args(ORPOConfig, common)
        return ORPOTrainer(
            model=model,
            args=args,
            train_dataset=train_pref,
            eval_dataset=val_pref,
            processing_class=tokenizer,
        )
    if cfg.method == "kto":
        kto_common = dict(common)
        args = build_trl_args(KTOConfig, kto_common)
        return KTOTrainer(
            model=model,
            args=args,
            train_dataset=train_pref,
            eval_dataset=val_pref,
            processing_class=tokenizer,
        )
    raise ValueError(cfg.method)


def run_one(cfg: PrefRunConfig, full_ds: DatasetDict) -> Dict[str, Any]:
    set_seed(cfg.seed)
    reset_cuda()
    start = time.time()

    ds = deepcopy(full_ds)
    ds["train"] = subsample_split(ds["train"], cfg.smoke_fraction, cfg.seed)
    ds["validation"] = subsample_split(ds["validation"], cfg.eval_fraction, cfg.seed + 1)
    ds["test"] = subsample_split(ds["test"], cfg.eval_fraction, cfg.seed + 2)
    ds = expand_dataset_with_chunks(ds, cfg)
    if cfg.rebalance_train:
        ds["train"] = rebalance_binary_joint_dataset(ds["train"], cfg)

    train_pref = build_preference_dataset(ds["train"], cfg)
    val_pref = build_preference_dataset(ds["validation"], cfg)

    print(
        f"[run:{cfg.run_name}] method={cfg.method} mode={cfg.mode} task_mode={cfg.task_mode} "
        f"train_pref={len(train_pref)} val_pref={len(val_pref)} "
        f"window={cfg.window_size} overlap={cfg.window_overlap}"
    )

    model, tokenizer = load_model_and_tokenizer(cfg)
    trainer = make_trainer(cfg, model, tokenizer, train_pref, val_pref)
    train_out = trainer.train()

    val_metrics = eval_generative(model, tokenizer, ds["validation"], cfg)
    test_metrics = eval_generative(model, tokenizer, ds["test"], cfg)

    elapsed = time.time() - start
    peak_mem = peak_gpu_mem_gb()
    row = {
        **asdict(cfg),
        "train_runtime_sec": elapsed,
        "train_loss": float(train_out.training_loss) if hasattr(train_out, "training_loss") else np.nan,
        "peak_gpu_mem_gb": peak_mem,
        "val_acc": val_metrics.get("accuracy", np.nan),
        "val_f1": val_metrics.get("f1_macro", val_metrics.get("raw_f1_macro_mean", np.nan)),
        "val_f1_weighted": val_metrics.get("f1_weighted", np.nan),
        "val_minority_f1": val_metrics.get("class_0_1_f1", np.nan),
        "val_minority_recall": val_metrics.get("class_0_1_recall", np.nan),
        "test_acc": test_metrics.get("accuracy", np.nan),
        "test_f1": test_metrics.get("f1_macro", test_metrics.get("raw_f1_macro_mean", np.nan)),
        "test_f1_weighted": test_metrics.get("f1_weighted", np.nan),
        "val_precision_macro": val_metrics.get("precision_macro", np.nan),
        "val_precision_weighted": val_metrics.get("precision_weighted", np.nan),
        "val_recall_macro": val_metrics.get("recall_macro", np.nan),
        "val_recall_weighted": val_metrics.get("recall_weighted", np.nan),
        "test_precision_macro": test_metrics.get("precision_macro", np.nan),
        "test_precision_weighted": test_metrics.get("precision_weighted", np.nan),
        "test_recall_macro": test_metrics.get("recall_macro", np.nan),
        "test_recall_weighted": test_metrics.get("recall_weighted", np.nan),
        "val_ece": val_metrics.get("ece", np.nan),
        "val_brier": val_metrics.get("brier", np.nan),
        "test_ece": test_metrics.get("ece", np.nan),
        "test_brier": test_metrics.get("brier", np.nan),
        "test_minority_f1": test_metrics.get("class_0_1_f1", np.nan),
        "test_minority_recall": test_metrics.get("class_0_1_recall", np.nan),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    del trainer
    del model
    del tokenizer
    reset_cuda()
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Preference optimization experiment: DPO/ORPO/KTO")
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--task-mode", choices=["binary_derived", "ordinal_raw"], default="binary_derived")
    parser.add_argument("--methods", default="dpo,orpo,kto", help="Comma-separated subset of dpo,orpo,kto")
    parser.add_argument("--modes", default="lora,qlora", help="Comma-separated subset of lora,qlora")
    parser.add_argument("--smoke-fraction", type=float, default=None)
    parser.add_argument("--eval-fraction", type=float, default=None)
    parser.add_argument("--rebalance-train", action="store_true", help="Rebalance preference train split (binary_derived only).")
    parser.add_argument("--rebalance-strategy", choices=["none", "sqrt", "max"], default="sqrt")
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="Override epochs (default: smoke=2, full=3).",
    )
    parser.add_argument(
        "--chunk-grid",
        default="94:24,192:48,384:96,512:128",
        help="Comma-separated window:overlap pairs used for all method/mode runs.",
    )
    parser.add_argument(
        "--run-name-contains",
        type=str,
        default="",
        help="If set, run only configs whose run_name contains this substring.",
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
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

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

    chunk_grid = parse_chunk_grid(args.chunk_grid)
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"train CSV not found: {args.train_path}")
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"test CSV not found: {args.test_path}")
    ds = load_data(args.train_path, args.test_path, val_size=0.2, seed=42)

    os.makedirs("./pref_results", exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_log = f"./pref_results/pref_results_{run_stamp}.csv"
    jsonl_log = f"./pref_results/pref_results_{run_stamp}.jsonl"
    latest_csv = "./pref_results/pref_results_latest.csv"
    latest_jsonl = "./pref_results/pref_results_latest.jsonl"
    open(latest_jsonl, "w", encoding="utf-8").close()

    runs = []
    idx = 0
    for method in methods:
        for mode in modes:
            for ws, ov in chunk_grid:
                name_prefix = (
                    f"{args.stage}_{idx}_sf{frac_tag(smoke_fraction)}_ef{frac_tag(eval_fraction)}_ep{epochs}"
                )
                runs.append(
                    PrefRunConfig(
                        run_name=f"{name_prefix}_{method}_{mode}_tm{args.task_mode}_ws{ws}_ov{ov}",
                        method=method,
                        mode=mode,
                        task_mode=args.task_mode,
                        smoke_fraction=smoke_fraction,
                        eval_fraction=eval_fraction,
                        num_train_epochs=epochs,
                        rebalance_train=args.rebalance_train,
                        rebalance_strategy=args.rebalance_strategy,
                        window_size=ws,
                        window_overlap=ov,
                    )
                )
                idx += 1
    if args.run_name_contains:
        runs = [r for r in runs if args.run_name_contains in r.run_name]
        print(f"[filter] configs after run_name_contains='{args.run_name_contains}': {len(runs)}")

    print(
        f"Running preference sweep: stage={args.stage}, task_mode={args.task_mode}, "
        f"methods={methods}, modes={modes}, n_runs={len(runs)}"
    )

    results = []
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
