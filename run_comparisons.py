#!/usr/bin/env python3
"""Rerun comparison baselines with the ablation combo configuration.

The script mirrors the training recipe used in
``run_combo_prev_current_repeats.py``:
    - dataset: /home/quanzb23/JASIST/exp1010/scibert7.5:1:1.5/data75:1:15
    - features: prev/current/citing_title/cited_abstract (hybrid fusion)
    - training: SciBERT-style fusion encoder, Focal loss (gamma 1.3),
      epochs=15, patience=3, batch size=8, warmup ratio=0.1

Each selected model is evaluated with seeds 42-46. Results are appended to
``comparison_results.jsonl`` so the process can resume safely after an
interruption.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Environment preparation (match run_combo_prev_current_repeats setup)
# ---------------------------------------------------------------------------
_preferred_libstd = "/home/quanzb23/.conda/envs/hgt310/lib/libstdc++.so.6"
_lib_candidates = [
    _preferred_libstd,
    "/home/quanzb23/.conda/envs/hgt310/lib/libstdc++.so.6.0.29",
    os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib", "libstdc++.so.6"),
    "/opt/anaconda3/lib/libstdc++.so.6",
]


def _load_conda_libstdcxx() -> str:
    tried: List[str] = []
    for candidate in _lib_candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        tried.append(candidate)
        try:
            import ctypes

            ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", ctypes.RTLD_LOCAL))
            return candidate
        except OSError:
            continue

    prefix_list = [
        os.environ.get("CONDA_PREFIX"),
        getattr(sys, "base_prefix", None),
        sys.prefix,
        sys.exec_prefix,
        os.path.dirname(os.path.dirname(sys.executable)),
    ]
    for prefix in prefix_list:
        if not prefix or not os.path.isdir(prefix):
            continue
        lib_dir = os.path.join(prefix, "lib")
        if not os.path.isdir(lib_dir):
            continue
        for name in sorted(os.listdir(lib_dir), reverse=True):
            if not name.startswith("libstdc++.so.6"):
                continue
            candidate = os.path.join(lib_dir, name)
            if not os.path.isfile(candidate):
                continue
            tried.append(candidate)
            try:
                import ctypes

                ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", ctypes.RTLD_LOCAL))
                return candidate
            except OSError:
                continue

    raise RuntimeError(
        "无法加载合适的 libstdc++.so.6，请确认 Conda 环境包含该库。尝试过: "
        + ", ".join(tried)
    )


_loaded_libstd = _load_conda_libstdcxx()
lib_dir = os.path.dirname(_loaded_libstd)
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
parts = [p for p in current_ld.split(":" ) if p] if current_ld else []
if lib_dir not in parts:
    parts.insert(0, lib_dir)
os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

PROXY = "http://172.16.134.238:7890"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# ---------------------------------------------------------------------------
# Imports (after env preparation)
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# Paths and experiment constants
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/quanzb23/JASIST/exp1010/scibert7.5:1:1.5/data75:1:15"
TRAIN_PATH = Path(DATA_ROOT) / "train_split.csv"
DEV_PATH = Path(DATA_ROOT) / "val_split.csv"
TEST_PATH = Path(DATA_ROOT) / "test_split.csv"

OUTPUT_DIR = Path("/home/quanzb23/JASIST/开源代码/对比实验")
RESULTS_PATH = OUTPUT_DIR / "comparison_results.jsonl"

SEEDS = [42, 43, 44, 45, 46]
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 15
PATIENCE = 3
WARMUP_RATIO = 0.1
LOSS_NAME = "Focal_1.3"
USE_GPU = os.environ.get("USE_GPU", "1") != "0"

CANDIDATE_FEATURES: Dict[str, str] = {
    "section": "citation_section",
    "context": "citation_context",
    "prev": "prev_sentence",
    "current": "current_sentence",
    "next": "next_sentence",
    "citing_title": "citing_paper_title",
    "citing_authors": "citing_paper_authors",
    "citing_abstract": "citing_paper_abstract",
    "cited_title": "cited_paper_title",
    "cited_authors": "cited_paper_authors",
    "cited_abstract": "cited_paper_abstract",
    "period": "period",
}

FEATURE_KEYS = ["prev", "current", "citing_title", "cited_abstract"]
N_FEATURES = len(FEATURE_KEYS)

MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "bert": {"type": "transformer", "pretrained": "bert-base-uncased", "fusion": "hybrid"},
    "scibert": {"type": "transformer", "pretrained": "allenai/scibert_scivocab_uncased", "fusion": "hybrid"},
    "scincl": {"type": "transformer", "pretrained": "malteos/scincl", "fusion": "hybrid"},
    "specter": {"type": "transformer", "pretrained": "allenai/specter", "fusion": "hybrid"},
    "specter2_base": {"type": "transformer", "pretrained": "allenai/specter2_base", "fusion": "hybrid"},
    "scincl_att_focal_gamma1p3": {"type": "transformer", "pretrained": "malteos/scincl", "fusion": "hybrid_att"},
    "specter_att_focal_gamma1p3": {"type": "transformer", "pretrained": "allenai/specter", "fusion": "hybrid_att"},
    "specter2_base_att_focal_gamma1p3": {"type": "transformer", "pretrained": "allenai/specter2_base", "fusion": "hybrid_att"},
    "random_forest": {"type": "random_forest"},
    "svm": {"type": "svm"},
    "xgboost": {"type": "xgboost"},
}

MODEL_ORDER = [
    "bert",
    "scibert",
    "random_forest",
    "svm",
    "xgboost",
    "scincl",
    "specter",
    "specter2_base",
    "scincl_att_focal_gamma1p3",
    "specter_att_focal_gamma1p3",
    "specter2_base_att_focal_gamma1p3",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_value(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "[EMPTY]"
    text = str(value).strip()
    return text if text else "[EMPTY]"


def row_feature_texts(row: pd.Series) -> List[str]:
    texts: List[str] = []
    for key in FEATURE_KEYS:
        col = CANDIDATE_FEATURES[key]
        val = clean_value(row.get(col))
        texts.append(f"[{col}] {val}")
    return texts


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    texts = []
    joined = []
    for _, row in df.iterrows():
        feats = row_feature_texts(row)
        texts.append(feats)
        joined.append(" </s> ".join(feats))
    enriched = df.copy()
    enriched["_feature_texts"] = texts
    enriched["_joined_text"] = joined
    return enriched


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = len(labels) / (num_classes * counts.astype(np.float32))
    return weights


def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
    }


# ---------------------------------------------------------------------------
# Dataset & collate
# ---------------------------------------------------------------------------
class FusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: np.ndarray) -> None:
        self.feature_texts = df["_feature_texts"].tolist()
        self.joined_texts = df["_joined_text"].tolist()
        self.labels = labels.astype(np.int64)
        self.n_features = len(self.feature_texts[0]) if self.feature_texts else N_FEATURES

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "feature_texts": self.feature_texts[idx],
            "joined_text": self.joined_texts[idx],
            "label": int(self.labels[idx]),
        }


def make_collate_fn(tokenizer, max_length: int, n_features: int):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        joined_texts = [item["joined_text"] for item in batch]

        joined_enc = tokenizer(
            joined_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        feature_texts: List[str] = []
        for item in batch:
            texts = item["feature_texts"]
            if len(texts) != n_features:
                raise ValueError(f"Expected {n_features} features, got {len(texts)}")
            feature_texts.extend(texts)

        feature_enc = tokenizer(
            feature_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        seq_len = feature_enc["input_ids"].shape[-1]
        batch_size = len(batch)
        feature_input_ids = feature_enc["input_ids"].view(batch_size, n_features, seq_len)
        feature_attention_mask = feature_enc["attention_mask"].view(batch_size, n_features, seq_len)

        return {
            "labels": labels,
            "joined_input_ids": joined_enc["input_ids"],
            "joined_attention_mask": joined_enc["attention_mask"],
            "feature_input_ids": feature_input_ids,
            "feature_attention_mask": feature_attention_mask,
        }

    return collate


# ---------------------------------------------------------------------------
# Fusion model + loss
# ---------------------------------------------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Parameter(torch.randn(hidden_size))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.tanh(self.proj(features))
        scores = torch.matmul(scores, self.context)
        weights = torch.softmax(scores, dim=1)
        weighted_sum = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return weighted_sum, weights


class MultiFeatureFusionModel(nn.Module):
    def __init__(self, base_model_name: str, num_classes: int, n_features: int, strategy: str) -> None:
        super().__init__()
        self.strategy = strategy
        self.n_features = n_features
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size

        if strategy == "hybrid":
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
        elif strategy == "hybrid_att":
            self.attention = FeatureAttention(hidden_size)
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
        elif strategy == "late":
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")

    def encode_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_features, seq_len = input_ids.shape
        flat_ids = input_ids.view(batch_size * n_features, seq_len)
        flat_mask = attention_mask.view(batch_size * n_features, seq_len)
        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        return outputs.pooler_output.view(batch_size, n_features, -1)

    def encode_joined(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature_repr = self.encode_features(batch["feature_input_ids"], batch["feature_attention_mask"])
        pooled = feature_repr.mean(dim=1)
        if self.strategy == "hybrid":
            joined = self.encode_joined(batch["joined_input_ids"], batch["joined_attention_mask"])
            logits = self.classifier(torch.cat([joined, pooled], dim=1))
        elif self.strategy == "hybrid_att":
            joined = self.encode_joined(batch["joined_input_ids"], batch["joined_attention_mask"])
            att_pooled, _ = self.attention(feature_repr)
            logits = self.classifier(torch.cat([joined, att_pooled], dim=1))
        else:  # late
            logits = self.classifier(pooled)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_probs, target, weight=weight, reduction="none")
        pt = torch.exp(-loss)
        focal = ((1 - pt) ** self.gamma) * loss
        return focal.mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_datasets() -> Dict[str, object]:
    train_df = enrich_dataframe(pd.read_csv(TRAIN_PATH))
    dev_df = enrich_dataframe(pd.read_csv(DEV_PATH))
    test_df = enrich_dataframe(pd.read_csv(TEST_PATH))

    all_labels = pd.concat(
        [train_df["KC"], dev_df["KC"], test_df["KC"]], ignore_index=True
    ).fillna("unknown")
    label_encoder = LabelEncoder().fit(all_labels)

    def encode(df: pd.DataFrame) -> np.ndarray:
        return label_encoder.transform(df["KC"].fillna("unknown"))

    train_labels = encode(train_df)
    dev_labels = encode(dev_df)
    test_labels = encode(test_df)

    class_weights = compute_class_weights(train_labels, len(label_encoder.classes_))

    return {
        "frames": {"train": train_df, "dev": dev_df, "test": test_df},
        "labels": {"train": train_labels, "dev": dev_labels, "test": test_labels},
        "texts": {
            split: frame["_joined_text"].tolist()
            for split, frame in {"train": train_df, "dev": dev_df, "test": test_df}.items()
        },
        "num_classes": len(label_encoder.classes_),
        "class_weights": class_weights,
        "n_features": N_FEATURES,
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    losses: List[float] = []
    preds: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["labels"])
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    metrics = compute_metrics(preds, labels)
    return avg_loss, metrics


def train_transformer_model(model_name: str, spec: Dict[str, object], seed: int, data_bundle: Dict[str, object]) -> Dict[str, object]:
    set_seed(seed)
    if USE_GPU and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(spec["pretrained"])
    n_features = data_bundle["n_features"]
    collate_fn = make_collate_fn(tokenizer, MAX_LENGTH, n_features)

    train_loader = DataLoader(
        FusionDataset(data_bundle["frames"]["train"], data_bundle["labels"]["train"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        FusionDataset(data_bundle["frames"]["dev"], data_bundle["labels"]["dev"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        FusionDataset(data_bundle["frames"]["test"], data_bundle["labels"]["test"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = MultiFeatureFusionModel(
        base_model_name=spec["pretrained"],
        num_classes=data_bundle["num_classes"],
        n_features=n_features,
        strategy=spec.get("fusion", "hybrid"),
    ).to(device)

    lr = 2e-5
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = max(len(train_loader), 1) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    weight_tensor = torch.tensor(data_bundle["class_weights"], dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=1.3, weight=weight_tensor)

    best_dev = -float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0

    progress = tqdm(range(1, EPOCHS + 1), desc=f"{model_name}", unit="epoch", leave=False)
    for epoch in progress:
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(train_loader), 1)
        _, dev_metrics = evaluate_model(model, dev_loader, device)
        progress.set_postfix(train_loss=f"{avg_train_loss:.4f}", dev_f1=f"{dev_metrics['macro_f1']:.4f}")

        if dev_metrics["macro_f1"] > best_dev + 1e-6:
            best_dev = dev_metrics["macro_f1"]
            patience_counter = 0
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    progress.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_metrics = evaluate_model(model, test_loader, device)
    return {"metrics": test_metrics, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------
def train_random_forest(texts: Dict[str, List[str]], labels: Dict[str, np.ndarray], seed: int) -> Dict[str, float]:
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)),
        ]
    )
    pipeline.fit(texts["train"], labels["train"])
    preds = pipeline.predict(texts["test"])
    return compute_metrics(preds.tolist(), labels["test"].tolist())


def train_linear_svm(texts: Dict[str, List[str]], labels: Dict[str, np.ndarray]) -> Dict[str, float]:
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual=False)),
        ]
    )
    pipeline.fit(texts["train"], labels["train"])
    preds = pipeline.predict(texts["test"])
    return compute_metrics(preds.tolist(), labels["test"].tolist())


def train_xgboost(texts: Dict[str, List[str]], labels: Dict[str, np.ndarray], num_classes: int, seed: int) -> Dict[str, float]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    X_train = vectorizer.fit_transform(texts["train"])
    X_test = vectorizer.transform(texts["test"])

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_train, labels["train"])
    preds = model.predict(X_test)
    return compute_metrics(preds.tolist(), labels["test"].tolist())


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def load_existing_results(path: Path) -> Dict[Tuple[str, int], Dict[str, object]]:
    existing: Dict[Tuple[str, int], Dict[str, object]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (data.get("model"), data.get("seed"))
            if None in key:
                continue
            existing[(key[0], int(key[1]))] = data
    return existing


def append_result(record: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def run_models(selected_models: List[str], output_path: Path) -> None:
    data_bundle = load_datasets()
    existing = load_existing_results(output_path)

    for model_name in selected_models:
        if model_name not in MODEL_SPECS:
            print(f"未知模型 {model_name}，跳过")
            continue
        spec = MODEL_SPECS[model_name]
        for seed in SEEDS:
            key = (model_name, seed)
            if key in existing:
                print(f"跳过 {model_name} seed={seed} (已存在)")
                continue
            print(f"开始 {model_name} seed={seed}")
            start = time.time()
            if spec["type"] == "transformer":
                outcome = train_transformer_model(model_name, spec, seed, data_bundle)
                metrics = outcome["metrics"]
                record = {
                    "model": model_name,
                    "seed": seed,
                    "test_macro_precision": metrics["macro_precision"],
                    "test_macro_recall": metrics["macro_recall"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_accuracy": metrics["accuracy"],
                    "best_epoch": outcome["best_epoch"],
                }
            elif spec["type"] == "random_forest":
                metrics = train_random_forest(data_bundle["texts"], data_bundle["labels"], seed)
                record = {
                    "model": model_name,
                    "seed": seed,
                    "test_macro_precision": metrics["macro_precision"],
                    "test_macro_recall": metrics["macro_recall"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_accuracy": metrics["accuracy"],
                }
            elif spec["type"] == "svm":
                metrics = train_linear_svm(data_bundle["texts"], data_bundle["labels"])
                record = {
                    "model": model_name,
                    "seed": seed,
                    "test_macro_precision": metrics["macro_precision"],
                    "test_macro_recall": metrics["macro_recall"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_accuracy": metrics["accuracy"],
                }
            elif spec["type"] == "xgboost":
                metrics = train_xgboost(data_bundle["texts"], data_bundle["labels"], data_bundle["num_classes"], seed)
                record = {
                    "model": model_name,
                    "seed": seed,
                    "test_macro_precision": metrics["macro_precision"],
                    "test_macro_recall": metrics["macro_recall"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_accuracy": metrics["accuracy"],
                }
            else:
                print(f"未知模型类型: {spec['type']}")
                continue

            runtime = time.time() - start
            record["runtime_sec"] = runtime
            append_result(record, output_path)
            existing[key] = record
            print(
                f"完成 {model_name} seed={seed}: F1={record['test_macro_f1']:.4f}, "
                f"ACC={record['test_accuracy']:.4f}, runtime={runtime/60:.1f} min"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparison baselines with the ablation configuration.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="仅运行指定模型 (默认运行全部)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help="结果输出路径 (JSONL)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = args.models if args.models else MODEL_ORDER
    run_models(models, args.output)


if __name__ == "__main__":
    main()
