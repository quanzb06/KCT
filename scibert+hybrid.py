
from __future__ import annotations

import ctypes
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Environment preparation (ensure correct libstdc++ for PIL/transformers)
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
                ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", ctypes.RTLD_LOCAL))
                return candidate
            except OSError:
                continue

    raise RuntimeError(
        "Unable to load a compatible libstdc++.so.6. Checked: " + ", ".join(tried)
    )


_loaded_libstd = _load_conda_libstdcxx()
lib_dir = os.path.dirname(_loaded_libstd)
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
parts = [p for p in current_ld.split(":") if p] if current_ld else []
if lib_dir not in parts:
    parts.insert(0, lib_dir)
os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PROXY = "http://172.16.134.238:7890"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

USE_GPU = os.environ.get("USE_GPU", "1") != "0"

# ---------------------------------------------------------------------------
# Paths and experiment constants
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/quanzb23/JASIST/exp1010/scibert7.5:1:1.5/data75:1:15"
TRAIN_PATH = os.path.join(DATA_ROOT, "train_split.csv")
DEV_PATH = os.path.join(DATA_ROOT, "val_split.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test_split.csv")

OUTPUT_DIR = "/home/quanzb23/JASIST/开源代码01/SCIBERT+HYBRID"
SUMMARY_RESULTS_PATH = os.path.join(OUTPUT_DIR, "scibert_hybrid_summary_results.csv")
RAW_RESULTS_PATH = os.path.join(OUTPUT_DIR, "scibert_hybrid_raw_results.csv")

SEEDS = [42, 43, 44, 45, 46]

MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 15
PATIENCE = 3
WARMUP_RATIO = 0.1
GLOBAL_SEED = 42

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

BASE_FEATURES = ["prev", "current", "citing_title", "cited_abstract"]

SCIBERT_HYBRID_CONFIG: Dict[str, object] = {
    "group": "baseline",
    "name": "combo_prev_current_citingtitle_citedabstract",
    "description": "Base combo prev/current/citing_title/cited_abstract",
    "feature_keys": BASE_FEATURES,
    "fusion": "hybrid",
    "loss": "Focal_1.3",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
    }


# ---------------------------------------------------------------------------
# Dataset & collate
# ---------------------------------------------------------------------------
class FusionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        labels: np.ndarray,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.labels = labels
        self.n_features = len(feature_cols)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        feature_texts = []
        for col in self.feature_cols:
            val = row.get(col, "")
            if pd.isna(val):
                val = "[EMPTY]"
            else:
                val = str(val).strip()
                if not val:
                    val = "[EMPTY]"
            feature_texts.append(f"[{col}] {val}")
        joined = " </s> ".join(feature_texts)
        label = int(self.labels[idx])
        return {
            "feature_texts": feature_texts,
            "joined_text": joined,
            "label": label,
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
                raise ValueError(
                    f"Expected {n_features} feature texts, got {len(texts)}"
                )
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
        feature_attention_mask = feature_enc["attention_mask"].view(
            batch_size, n_features, seq_len
        )

        return {
            "labels": labels,
            "joined_input_ids": joined_enc["input_ids"],
            "joined_attention_mask": joined_enc["attention_mask"],
            "feature_input_ids": feature_input_ids,
            "feature_attention_mask": feature_attention_mask,
        }

    return collate


# ---------------------------------------------------------------------------
# Fusion models
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
    def __init__(self, num_classes: int, n_features: int, strategy: str) -> None:
        super().__init__()
        self.strategy = strategy
        self.n_features = n_features
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size

        if strategy == "hybrid":
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
        elif strategy == "hybrid_att":
            self.attention = FeatureAttention(hidden_size)
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
        elif strategy == "late_att":
            self.per_feature_classifier = nn.Linear(hidden_size, num_classes)
            self.attention_linear = nn.Linear(hidden_size, 1)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)
            if "att" in strategy:
                self.attention = FeatureAttention(hidden_size)

    def encode_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, n_features, seq_len = input_ids.shape
        flat_ids = input_ids.view(batch_size * n_features, seq_len)
        flat_mask = attention_mask.view(batch_size * n_features, seq_len)
        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        pooled = outputs.pooler_output.view(batch_size, n_features, -1)
        return pooled

    def encode_joined(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature_repr = self.encode_features(
            batch["feature_input_ids"], batch["feature_attention_mask"]
        )

        if self.strategy == "early":
            joined = self.encode_joined(
                batch["joined_input_ids"], batch["joined_attention_mask"]
            )
            logits = self.classifier(joined)
        elif self.strategy == "late":
            pooled = feature_repr.mean(dim=1)
            logits = self.classifier(pooled)
        elif self.strategy == "hybrid":
            joined = self.encode_joined(
                batch["joined_input_ids"], batch["joined_attention_mask"]
            )
            pooled = feature_repr.mean(dim=1)
            logits = self.classifier(torch.cat([joined, pooled], dim=1))
        elif self.strategy == "early_att":
            pooled, _ = self.attention(feature_repr)
            logits = self.classifier(pooled)
        elif self.strategy == "late_att":
            per_feature_logits = self.per_feature_classifier(feature_repr)
            att_scores = self.attention_linear(feature_repr).squeeze(-1)
            weights = torch.softmax(att_scores, dim=1).unsqueeze(-1)
            logits = torch.sum(weights * per_feature_logits, dim=1)
        elif self.strategy == "hybrid_att":
            joined = self.encode_joined(
                batch["joined_input_ids"], batch["joined_attention_mask"]
            )
            pooled, _ = self.attention(feature_repr)
            logits = self.classifier(torch.cat([joined, pooled], dim=1))
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")

        return logits


# ---------------------------------------------------------------------------
# Loss definitions
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        gather_logp = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        if self.weight is not None:
            alpha = self.weight[targets]
        else:
            alpha = 1.0

        loss = -alpha * ((1 - pt) ** self.gamma) * gather_logp
        return loss.mean()


def build_loss(name: str, class_weights: np.ndarray, device: torch.device) -> nn.Module:
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    if name == "CE":
        return nn.CrossEntropyLoss().to(device)
    if name == "Weighted_CE":
        return nn.CrossEntropyLoss(weight=weight_tensor).to(device)
    if name.startswith("Focal_"):
        gamma = float(name.split("_")[1])
        return FocalLoss(gamma=gamma, weight=weight_tensor).to(device)
    raise ValueError(f"Unknown loss function: {name}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    preds_all: List[int] = []
    labels_all: List[int] = []
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        logits = model(batch)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(preds_all, labels_all)
    return avg_loss, metrics


def train_single_experiment(
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    n_features: int,
    fusion_strategy: str,
    loss_name: str,
    class_weights: np.ndarray,
    device: torch.device,
    seed: int,
) -> Dict[str, object]:
    model = MultiFeatureFusionModel(
        num_classes=num_classes,
        n_features=n_features,
        strategy=fusion_strategy,
    ).to(device)

    set_seed(seed)

    lr = 1e-5 if fusion_strategy in {"early_att", "late_att", "hybrid_att"} else 2e-5
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )
    criterion = build_loss(loss_name, class_weights, device)

    best_dev_f1 = -math.inf
    best_state = None
    best_epoch = -1
    patience_counter = 0

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc=f"Train {fusion_strategy}-{loss_name}", leave=False)
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        dev_loss, dev_metrics = evaluate(model, dev_loader, device)
        epoch_bar.set_postfix(
            {"train_loss": f"{avg_train_loss:.4f}", "dev_f1": f"{dev_metrics['macro_f1']:.4f}"}
        )

        if dev_metrics["macro_f1"] > best_dev_f1 + 1e-6:
            best_dev_f1 = dev_metrics["macro_f1"]
            patience_counter = 0
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "dev_loss": dev_loss,
                "dev_metrics": dev_metrics,
            }
            best_epoch = epoch
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    epoch_bar.close()

    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "epoch": EPOCHS,
            "dev_loss": float("nan"),
            "dev_metrics": {"macro_f1": float("nan")},
        }
        best_epoch = EPOCHS

    model.load_state_dict(best_state["model"])
    test_loss, test_metrics = evaluate(model, test_loader, device)

    torch.cuda.empty_cache()

    test_metrics["loss"] = test_loss
    return {
        "best_epoch": best_epoch,
        "dev_loss": best_state["dev_loss"],
        "dev_metrics": best_state["dev_metrics"],
        "test_metrics": test_metrics,
    }


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def feature_columns(keys: List[str]) -> List[str]:
    cols = []
    for key in keys:
        if key not in CANDIDATE_FEATURES:
            raise KeyError(f"Unknown feature key: {key}")
        cols.append(CANDIDATE_FEATURES[key])
    return cols


def load_datasets():
    train_df = load_table(TRAIN_PATH)
    dev_df = load_table(DEV_PATH)
    test_df = load_table(TEST_PATH)

    all_labels = pd.concat(
        [train_df["KC"], dev_df["KC"], test_df["KC"]],
        ignore_index=True,
    ).fillna("unknown")
    label_encoder = LabelEncoder().fit(all_labels)

    train_labels = label_encoder.transform(train_df["KC"].fillna("unknown"))
    dev_labels = label_encoder.transform(dev_df["KC"].fillna("unknown"))
    test_labels = label_encoder.transform(test_df["KC"].fillna("unknown"))

    class_weights = compute_class_weights(train_labels, len(label_encoder.classes_))

    return {
        "train_df": train_df,
        "dev_df": dev_df,
        "test_df": test_df,
        "train_labels": train_labels,
        "dev_labels": dev_labels,
        "test_labels": test_labels,
        "num_classes": len(label_encoder.classes_),
        "class_weights": class_weights,
    }


def load_raw_results() -> Tuple[pd.DataFrame, set]:
    if os.path.exists(RAW_RESULTS_PATH):
        df = pd.read_csv(RAW_RESULTS_PATH)
        done = set(df["seed"])
        return df, done
    columns = [
        "group",
        "name",
        "description",
        "features",
        "fusion",
        "loss",
        "seed",
        "best_epoch",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_accuracy",
        "runtime_sec",
    ]
    return pd.DataFrame(columns=columns), set()


def save_raw_results(df: pd.DataFrame) -> None:
    df.sort_values(by=["seed"], inplace=True)
    df.to_csv(RAW_RESULTS_PATH, index=False)


def load_summary_results() -> pd.DataFrame:
    if os.path.exists(SUMMARY_RESULTS_PATH):
        return pd.read_csv(SUMMARY_RESULTS_PATH)
    columns = [
        "group",
        "name",
        "description",
        "features",
        "fusion",
        "loss",
        "seeds",
        "macro_precision_mean",
        "macro_precision_std",
        "macro_recall_mean",
        "macro_recall_std",
        "macro_f1_mean",
        "macro_f1_std",
        "accuracy_mean",
        "accuracy_std",
        "runtime_sec_mean",
        "runtime_sec_std",
    ]
    return pd.DataFrame(columns=columns)


def save_summary_results(df: pd.DataFrame) -> None:
    df.sort_values(by=["group", "name"], inplace=True)
    df.to_csv(SUMMARY_RESULTS_PATH, index=False)


def aggregate_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    metrics_df = pd.DataFrame(records)
    return {
        "group": SCIBERT_HYBRID_CONFIG["group"],
        "name": SCIBERT_HYBRID_CONFIG["name"],
        "description": SCIBERT_HYBRID_CONFIG["description"],
        "features": "+".join(SCIBERT_HYBRID_CONFIG["feature_keys"]),
        "fusion": SCIBERT_HYBRID_CONFIG["fusion"],
        "loss": SCIBERT_HYBRID_CONFIG["loss"],
        "seeds": ",".join(str(s) for s in sorted(SEEDS)),
        "macro_precision_mean": metrics_df["test_macro_precision"].mean(),
        "macro_precision_std": metrics_df["test_macro_precision"].std(ddof=0),
        "macro_recall_mean": metrics_df["test_macro_recall"].mean(),
        "macro_recall_std": metrics_df["test_macro_recall"].std(ddof=0),
        "macro_f1_mean": metrics_df["test_macro_f1"].mean(),
        "macro_f1_std": metrics_df["test_macro_f1"].std(ddof=0),
        "accuracy_mean": metrics_df["test_accuracy"].mean(),
        "accuracy_std": metrics_df["test_accuracy"].std(ddof=0),
        "runtime_sec_mean": metrics_df["runtime_sec"].mean(),
        "runtime_sec_std": metrics_df["runtime_sec"].std(ddof=0),
    }


def run_single_seed(
    config: Dict[str, object],
    seed: int,
    data_bundle: Dict[str, object],
    tokenizer,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(seed)

    cols = feature_columns(config["feature_keys"])
    n_features = len(cols)

    train_ds = FusionDataset(data_bundle["train_df"], cols, data_bundle["train_labels"])
    dev_ds = FusionDataset(data_bundle["dev_df"], cols, data_bundle["dev_labels"])
    test_ds = FusionDataset(data_bundle["test_df"], cols, data_bundle["test_labels"])

    collate_fn = make_collate_fn(tokenizer, MAX_LENGTH, n_features)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    start = time.time()
    outcome = train_single_experiment(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        num_classes=data_bundle["num_classes"],
        n_features=n_features,
        fusion_strategy=config["fusion"],
        loss_name=config["loss"],
        class_weights=data_bundle["class_weights"],
        device=device,
        seed=seed,
    )
    elapsed = time.time() - start

    metrics = outcome["test_metrics"]
    return {
        "group": config["group"],
        "name": config["name"],
        "description": config["description"],
        "features": "+".join(config["feature_keys"]),
        "fusion": config["fusion"],
        "loss": config["loss"],
        "seed": seed,
        "best_epoch": outcome["best_epoch"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall": metrics["macro_recall"],
        "test_macro_f1": metrics["macro_f1"],
        "test_accuracy": metrics["accuracy"],
        "runtime_sec": elapsed,
    }


def main() -> None:
    ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if USE_GPU and not torch.cuda.is_available():
        raise RuntimeError("USE_GPU=1 but CUDA is unavailable. Set USE_GPU=0 or check the GPU.")
    device = torch.device("cuda" if USE_GPU else "cpu")
    data_bundle = load_datasets()
    raw_df, raw_done = load_raw_results()
    summary_df = load_summary_results()

    raw_records = raw_df.to_dict("records")

    progress = tqdm(SEEDS, desc="SciBERT+Hybrid Seeds", dynamic_ncols=True)
    for seed in progress:
        if seed in raw_done:
            continue
        progress.set_postfix({"current_seed": seed})
        record = run_single_seed(SCIBERT_HYBRID_CONFIG, seed, data_bundle, tokenizer, device)
        raw_records.append(record)
        raw_df = pd.DataFrame(raw_records)
        save_raw_results(raw_df)
        raw_done.add(seed)
        progress.write(
            f"Completed SciBERT+Hybrid seed {seed} -> MacroF1={record['test_macro_f1']:.4f}, "
            f"ACC={record['test_accuracy']:.4f}"
        )

    progress.close()

    if raw_done == set(SEEDS):
        summary_row = aggregate_records(raw_records)
        summary_df = summary_df[summary_df["name"] != SCIBERT_HYBRID_CONFIG["name"]]
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
        save_summary_results(summary_df)
        print(
            "Aggregated SciBERT+Hybrid -> "
            f"MacroF1={summary_row['macro_f1_mean']:.4f}±{summary_row['macro_f1_std']:.4f}, "
            f"ACC={summary_row['accuracy_mean']:.4f}±{summary_row['accuracy_std']:.4f}"
        )

    print(f"SciBERT+Hybrid raw results saved to {RAW_RESULTS_PATH}")
    print(f"SciBERT+Hybrid summary saved to {SUMMARY_RESULTS_PATH}")


if __name__ == "__main__":
    set_seed(GLOBAL_SEED)
    main()
