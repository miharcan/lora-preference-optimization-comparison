import os
import gc
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

DEFAULT_TRAIN_PATH = "/mnt/3d56a808-6937-48b6-ba29-c2b68a3e2d94/datasets/train_dataset.csvV4_per_user"
DEFAULT_TEST_PATH = "/mnt/3d56a808-6937-48b6-ba29-c2b68a3e2d94/datasets/test_dataset.csvV4_per_user"


JOINT_CLASS_LABELS = [0, 1, 2, 3]
JOINT_CLASS_NAMES = {0: "0_0", 1: "1_0", 2: "0_1", 3: "1_1"}


@dataclass
class XGBRunConfig:
    run_name: str
    model_name: str = "xgboost_tfidf"
    mode: str = "xgboost"
    objective: str = "discriminative"
    output_schema: str = "joint_label"
    task_mode: str = "binary_derived"
    smoke_fraction: float = 0.1
    eval_fraction: float = 0.1
    window_size: int = 384
    window_overlap: int = 96
    seed: int = 42
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    tfidf_max_features: int = 100000
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    output_dir_root: str = "./xgb_results"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def reset_mem():
    gc.collect()


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def frac_tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def parse_chunk_grid(spec: str) -> List[Tuple[int, int]]:
    if spec is None or not spec.strip():
        return [(94, 24), (128, 32), (192, 48), (256, 64), (384, 96), (512, 128)]
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


def expand_dataset_with_chunks(ds: DatasetDict, cfg: XGBRunConfig) -> DatasetDict:
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


def compute_joint_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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


def evaluate_secondary_binary_from_joint(
    y_true_joint: np.ndarray,
    y_pred_joint: np.ndarray,
    joint_probs: np.ndarray | None = None,
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

    if joint_probs is not None:
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


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ece": expected_calibration_error(y_prob, y_true, n_bins=15),
        "brier": multiclass_brier_score(y_prob, y_true, num_classes=4),
        "joint_exact_match": float(np.mean(y_pred == y_true)),
    }
    metrics.update(compute_joint_class_metrics(y_true, y_pred))
    metrics.update(evaluate_secondary_binary_from_joint(y_true, y_pred, y_prob))
    return metrics


def run_one(cfg: XGBRunConfig, full_ds: DatasetDict) -> Dict[str, Any]:
    set_seed(cfg.seed)
    reset_mem()
    start = time.time()

    ds = DatasetDict(
        {
            "train": subsample_split(full_ds["train"], cfg.smoke_fraction, cfg.seed),
            "validation": subsample_split(full_ds["validation"], cfg.eval_fraction, cfg.seed + 1),
            "test": subsample_split(full_ds["test"], cfg.eval_fraction, cfg.seed + 2),
        }
    )
    ds = expand_dataset_with_chunks(ds, cfg)

    train_texts = list(ds["train"]["value"])
    val_texts = list(ds["validation"]["value"])
    test_texts = list(ds["test"]["value"])
    y_train = np.array(ds["train"]["joint_label"], dtype=int)
    y_val = np.array(ds["validation"]["joint_label"], dtype=int)
    y_test = np.array(ds["test"]["joint_label"], dtype=int)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=cfg.tfidf_max_features,
        ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
        min_df=2,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    present_classes = sorted(np.unique(y_train).tolist())
    if len(present_classes) < 2:
        raise ValueError(f"Training split has <2 classes after subsampling: {present_classes}")

    class_to_local = {c: i for i, c in enumerate(present_classes)}
    local_to_class = {i: c for c, i in class_to_local.items()}
    y_train_local = np.array([class_to_local[int(y)] for y in y_train], dtype=int)

    if len(present_classes) == 2:
        objective = "binary:logistic"
        num_class = None
        eval_metric = "logloss"
    else:
        objective = "multi:softprob"
        num_class = len(present_classes)
        eval_metric = "mlogloss"

    model_kwargs = dict(
        objective=objective,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        min_child_weight=cfg.min_child_weight,
        eval_metric=eval_metric,
        random_state=cfg.seed,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )
    if num_class is not None:
        model_kwargs["num_class"] = num_class

    model = XGBClassifier(**model_kwargs)

    print(
        f"[run:{cfg.run_name}] train={x_train.shape[0]} val={x_val.shape[0]} "
        f"test={x_test.shape[0]} window={cfg.window_size} overlap={cfg.window_overlap}"
    )
    model.fit(x_train, y_train_local)

    val_prob_local = model.predict_proba(x_val)
    test_prob_local = model.predict_proba(x_test)
    if val_prob_local.ndim == 1:
        val_prob_local = np.stack([1.0 - val_prob_local, val_prob_local], axis=1)
    if test_prob_local.ndim == 1:
        test_prob_local = np.stack([1.0 - test_prob_local, test_prob_local], axis=1)

    val_prob = np.zeros((len(y_val), 4), dtype=np.float64)
    test_prob = np.zeros((len(y_test), 4), dtype=np.float64)
    for local_idx, cls in local_to_class.items():
        val_prob[:, cls] = val_prob_local[:, local_idx]
        test_prob[:, cls] = test_prob_local[:, local_idx]

    val_pred = np.argmax(val_prob, axis=1)
    test_pred = np.argmax(test_prob, axis=1)

    val_metrics = evaluate_predictions(y_val, val_pred, val_prob)
    test_metrics = evaluate_predictions(y_test, test_pred, test_prob)

    elapsed = time.time() - start
    row = {
        **asdict(cfg),
        "train_runtime_sec": elapsed,
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "vocab_size": int(x_train.shape[1]),
        "val_acc": val_metrics["accuracy"],
        "val_f1": val_metrics["f1_macro"],
        "val_f1_weighted": val_metrics["f1_weighted"],
        "val_precision_macro": val_metrics["precision_macro"],
        "val_precision_weighted": val_metrics["precision_weighted"],
        "val_recall_macro": val_metrics["recall_macro"],
        "val_recall_weighted": val_metrics["recall_weighted"],
        "val_ece": val_metrics["ece"],
        "val_brier": val_metrics["brier"],
        "test_acc": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_macro"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_precision_weighted": test_metrics["precision_weighted"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_recall_weighted": test_metrics["recall_weighted"],
        "test_ece": test_metrics["ece"],
        "test_brier": test_metrics["brier"],
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost TF-IDF baseline experiment")
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--smoke-fraction", type=float, default=None)
    parser.add_argument("--eval-fraction", type=float, default=None)
    parser.add_argument(
        "--chunk-grid",
        default="94:24,128:32,192:48,256:64,384:96,512:128",
        help="Comma-separated window:overlap pairs, e.g. '94:24,192:48'",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--tfidf-max-features", type=int, default=100000)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
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
    else:
        smoke_fraction = args.smoke_fraction if args.smoke_fraction is not None else 1.0
        eval_fraction = args.eval_fraction if args.eval_fraction is not None else 1.0

    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"train CSV not found: {args.train_path}")
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"test CSV not found: {args.test_path}")
    ds = load_data(args.train_path, args.test_path, val_size=0.2, seed=42)

    os.makedirs("./xgb_results", exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_log = f"./xgb_results/xgb_results_{run_stamp}.csv"
    jsonl_log = f"./xgb_results/xgb_results_{run_stamp}.jsonl"
    latest_csv = "./xgb_results/xgb_results_latest.csv"
    latest_jsonl = "./xgb_results/xgb_results_latest.jsonl"
    open(latest_jsonl, "w", encoding="utf-8").close()

    chunk_grid = parse_chunk_grid(args.chunk_grid)
    runs = []
    for idx, (ws, ov) in enumerate(chunk_grid):
        name_prefix = f"{args.stage}_{idx}_sf{frac_tag(smoke_fraction)}_ef{frac_tag(eval_fraction)}"
        runs.append(
            XGBRunConfig(
                run_name=f"{name_prefix}_xgb_tfidf_tmbinary_derived_ws{ws}_ov{ov}",
                smoke_fraction=smoke_fraction,
                eval_fraction=eval_fraction,
                window_size=ws,
                window_overlap=ov,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_lambda=args.reg_lambda,
                reg_alpha=args.reg_alpha,
                min_child_weight=args.min_child_weight,
                tfidf_max_features=args.tfidf_max_features,
                tfidf_ngram_min=args.tfidf_ngram_min,
                tfidf_ngram_max=args.tfidf_ngram_max,
            )
        )

    print(f"Running XGBoost sweep: stage={args.stage}, n_runs={len(runs)}")

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
