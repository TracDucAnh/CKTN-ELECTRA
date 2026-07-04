"""
training.py — CKTN-ELECTRA continued pre-training script.

Directory layout assumed:
    <project_root>/
        corpus/
            train/  cham_train.json  khmer_train.json  tay_nung_train.json
            dev/    cham_dev.json    khmer_dev.json    tay_nung_dev.json
        main/
            CKTN-ELECTRA.py   ← architecture
            training.py       ← this file
            checkpoint/
                discriminator/
                generator/
                graphs/
                report.json

Usage:
    python quarter_generator_variant/training.py
"""

import importlib.util
import hashlib
import json
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# ─────────────────────────────────────────────────────────────────────────────
# 0. Import architecture from CKTN-ELECTRA.py (filename has a hyphen)
# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent.resolve()
_ARCH_PATH = _HERE / "CKTN-ELECTRA.py"

spec = importlib.util.spec_from_file_location("cktn_electra", _ARCH_PATH)
cktn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cktn_module)

CKTNElectra = cktn_module.CKTNElectra
LinearLambdaScheduler = cktn_module.LinearLambdaScheduler
get_parameter_groups = cktn_module.get_parameter_groups
DISCRIMINATOR_CHECKPOINT = cktn_module.DISCRIMINATOR_CHECKPOINT
TRAINING_CONFIG = cktn_module.TRAINING_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# 1. Paths & hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

CORPUS_DIR   = _HERE.parent / "corpus"
TRAIN_DIR    = CORPUS_DIR / "train"
DEV_DIR      = CORPUS_DIR / "dev"
CKPT_DIR     = _HERE / "checkpoint"
CACHE_DIR    = _HERE / "cache"
CHUNK_CACHE_DIR = CACHE_DIR / "chunks"
GRAPHS_DIR   = CKPT_DIR / "graphs"
DISC_DIR     = CKPT_DIR / "discriminator"
GEN_DIR      = CKPT_DIR / "generator"
STATE_DIR    = CKPT_DIR / "state"
REPORT_PATH  = CKPT_DIR / "report.json"

for d in (CKPT_DIR, CACHE_DIR, CHUNK_CACHE_DIR, GRAPHS_DIR, DISC_DIR, GEN_DIR, STATE_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = [
    TRAIN_DIR / "cham_train.json",
    TRAIN_DIR / "khmer_train.json",
    TRAIN_DIR / "tay_nung_train.json",
]
DEV_FILES = [
    DEV_DIR / "cham_dev.json",
    DEV_DIR / "khmer_dev.json",
    DEV_DIR / "tay_nung_dev.json",
]

# Training hyper-parameters (from paper / TRAINING_CONFIG)
TOTAL_EPOCHS   = TRAINING_CONFIG["total_epochs"]          # 6
MASK_RATE      = TRAINING_CONFIG["mask_rate"]              # 0.15
SEQ_LEN        = TRAINING_CONFIG["seq_len"]                # 512
LR             = TRAINING_CONFIG["lr"]                     # 2e-5
WARMUP_RATIO   = TRAINING_CONFIG["warmup_ratio"]           # 0.06
WEIGHT_DECAY   = TRAINING_CONFIG["weight_decay"]           # 0.01
GRAD_NORM      = TRAINING_CONFIG["grad_norm"]              # 1.0
LAMBDA_MAX     = TRAINING_CONFIG["lambda_max"]             # 50.0

NUM_WORKERS    = 4
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Tokenizer + vocab-size check
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer_and_check() -> AutoTokenizer:
    """
    Load slow tokenizer from the discriminator checkpoint.
    Verify that tokenizer.vocab_size matches the model config vocab_size.
    Raises RuntimeError on mismatch.
    """
    print(f"[Tokenizer] Loading slow tokenizer from '{DISCRIMINATOR_CHECKPOINT}' ...")
    tokenizer = AutoTokenizer.from_pretrained(
        DISCRIMINATOR_CHECKPOINT, use_fast=False
    )

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(DISCRIMINATOR_CHECKPOINT)
    model_vocab_size = config.vocab_size
    tok_vocab_size   = tokenizer.vocab_size

    print(f"[Tokenizer] tokenizer.vocab_size = {tok_vocab_size}")
    print(f"[Config]    model config vocab_size = {model_vocab_size}")

    if tok_vocab_size != model_vocab_size:
        raise RuntimeError(
            f"Vocab size mismatch: tokenizer has {tok_vocab_size} tokens "
            f"but model config expects {model_vocab_size}. "
            "Check that you are using the correct tokenizer / checkpoint."
        )
    print("[Tokenizer] Vocab size check passed.")
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_contents(json_files: List[Path]) -> List[str]:
    """Read 'content' field from a list of JSON files (each file is a list of objects)."""
    contents: List[str] = []
    for path in json_files:
        if not path.exists():
            print(f"[Warning] File not found, skipping: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                text = item.get("content", "").strip()
                if text:
                    contents.append(text)
        elif isinstance(data, dict):
            text = data.get("content", "").strip()
            if text:
                contents.append(text)
        print(f"[Data] Loaded {path.name}: {len(contents)} total contents so far")
    return contents


def tokenize_and_chunk(
    contents: List[str],
    tokenizer: AutoTokenizer,
    max_len: int = SEQ_LEN,
) -> List[List[int]]:
    """
    Tokenize each content string and split into non-overlapping chunks of
    at most max_len tokens (special tokens [CLS]/[SEP] are added per chunk).
    Returns a list of token-id lists, each of length <= max_len.
    """
    # Reserve 2 positions for [CLS] and [SEP]
    effective_len = max_len - 2

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    chunks: List[List[int]] = []
    for text in tqdm(contents, desc="  Tokenizing", leave=False):
        # Tokenize without special tokens; truncation=False to keep all tokens
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        # Split into non-overlapping chunks
        for start in range(0, len(token_ids), effective_len):
            chunk = token_ids[start : start + effective_len]
            if not chunk:
                continue
            # Add [CLS] and [SEP]
            full_chunk = [cls_id] + chunk + [sep_id]
            chunks.append(full_chunk)

    print(f"[Data] Total chunks: {len(chunks)}")
    return chunks


def corpus_cache_fingerprint(
    json_files: List[Path],
    tokenizer: AutoTokenizer,
    max_len: int,
) -> str:
    payload = {
        "checkpoint": DISCRIMINATOR_CHECKPOINT,
        "tokenizer_name": getattr(tokenizer, "name_or_path", DISCRIMINATOR_CHECKPOINT),
        "vocab_size": tokenizer.vocab_size,
        "max_len": max_len,
        "files": [],
    }

    for path in json_files:
        entry = {"path": str(path.relative_to(_HERE.parent))}
        if path.exists():
            stat = path.stat()
            entry["size"] = stat.st_size
            entry["mtime_ns"] = stat.st_mtime_ns
        else:
            entry["missing"] = True
        payload["files"].append(entry)

    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def chunks_cache_path(
    split: str,
    json_files: List[Path],
    tokenizer: AutoTokenizer,
    max_len: int,
) -> Path:
    fingerprint = corpus_cache_fingerprint(json_files, tokenizer, max_len)
    return CHUNK_CACHE_DIR / f"{split}_{fingerprint}.pkl"


def load_chunks_cache(cache_path: Path) -> List[List[int]]:
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    return cached["chunks"]


def save_chunks_cache(cache_path: Path, chunks: List[List[int]]):
    payload = {"chunks": chunks}
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Cache] Saved tokenized chunks: {cache_path}")


def load_or_build_chunks(
    split: str,
    json_files: List[Path],
    tokenizer: AutoTokenizer,
    max_len: int,
    rebuild_cache: bool = False,
) -> List[List[int]]:
    cache_path = chunks_cache_path(split, json_files, tokenizer, max_len)
    if cache_path.exists() and not rebuild_cache:
        chunks = load_chunks_cache(cache_path)
        print(f"[Cache] Loaded {split} chunks: {len(chunks)} from {cache_path}")
        return chunks

    print(f"\n[Data] Loading {split} data ...")
    contents = read_contents(json_files)
    chunks = tokenize_and_chunk(contents, tokenizer, max_len)
    save_chunks_cache(cache_path, chunks)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ELECTRADataset(Dataset):
    """
    Each item is a padded sequence with MLM masking applied.
    Returns:
        input_ids      : masked token ids           [seq_len]
        attention_mask : 1 for real tokens, 0 pad   [seq_len]
        token_type_ids : all zeros                  [seq_len]
        labels         : original ids at masked pos, -100 elsewhere [seq_len]
    """

    def __init__(
        self,
        chunks: List[List[int]],
        tokenizer: AutoTokenizer,
        max_len: int = SEQ_LEN,
        mask_rate: float = MASK_RATE,
    ):
        self.chunks    = chunks
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.mask_rate = mask_rate

        self.pad_id  = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id

        # Identify special token ids to avoid masking them
        self.special_ids = set(tokenizer.all_special_ids)

    def __len__(self) -> int:
        return len(self.chunks)

    def _apply_mlm(
        self, token_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Apply ELECTRA generator masking:
            selected positions → [MASK]
        Returns (masked_ids, labels) where labels[i] = original id if masked,
        else -100.
        """
        masked_ids = list(token_ids)
        labels     = [-100] * len(token_ids)

        # Eligible positions: non-special tokens
        eligible = [
            i for i, tid in enumerate(token_ids)
            if tid not in self.special_ids
        ]
        if not eligible:
            return masked_ids, labels

        n_mask = max(1, int(round(len(eligible) * self.mask_rate)))
        mask_indices = np.random.choice(eligible, size=n_mask, replace=False).tolist()

        for idx in mask_indices:
            labels[idx] = token_ids[idx]
            masked_ids[idx] = self.mask_id

        return masked_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]

        masked_ids, labels = self._apply_mlm(chunk)

        # Pad to max_len
        seq_len       = len(masked_ids)
        pad_len       = self.max_len - seq_len

        input_ids      = masked_ids + [self.pad_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_type_ids = [0] * self.max_len
        label_ids      = labels + [-100] * pad_len

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels":         torch.tensor(label_ids,      dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: CKTNElectra,
    dataloader: DataLoader,
    device: str,
    lam: float = LAMBDA_MAX,
) -> Dict[str, float]:
    """
    Evaluate on dev set.
    Returns MLM/RTD losses and diagnostics from the actual sampled corruption.
    RTD labels are 1=original and 0=replaced; F1 uses replaced as the positive class.
    """
    model.eval()

    total_mlm_loss = 0.0
    total_rtd_loss = 0.0
    total_replacement_rate = 0.0
    total_valid_candidate_rate = 0.0
    total_disc_confidence = 0.0
    total_rtd_entropy = 0.0
    all_rtd_preds = []
    all_rtd_labels = []
    n_batches = 0

    for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(
            input_ids      = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            token_type_ids = batch["token_type_ids"],
            labels         = batch["labels"],
            lam            = lam,
        )

        total_mlm_loss += out["loss_mlm"].item()
        total_rtd_loss += out["loss_disc"].item()
        total_replacement_rate += out["replacement_rate"].item()
        total_valid_candidate_rate += out["valid_candidate_rate"].item()
        total_disc_confidence += out["avg_disc_confidence"].item()
        total_rtd_entropy += out["rtd_entropy"].item()
        n_batches += 1

        if out["disc_logits"] is not None:
            disc_logits = out["disc_logits"].squeeze(-1)  # [B, L]
            preds = (torch.sigmoid(disc_logits) > 0.5).long()
            eval_mask = out["rtd_label_mask"]

            if eval_mask.any():
                preds_flat  = preds[eval_mask].cpu().numpy()
                labels_flat = out["rtd_labels"][eval_mask].long().cpu().numpy()

                all_rtd_preds.append(preds_flat)
                all_rtd_labels.append(labels_flat)

    avg_mlm = total_mlm_loss / max(n_batches, 1)
    avg_rtd = total_rtd_loss / max(n_batches, 1)

    if all_rtd_preds:
        y_pred = np.concatenate(all_rtd_preds)
        y_true = np.concatenate(all_rtd_labels)
        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    else:
        acc, f1 = 0.0, 0.0

    batch_count = max(n_batches, 1)
    model.train()
    return {
        "avg_mlm_loss": avg_mlm,
        "avg_rtd_loss": avg_rtd,
        "accuracy":     acc,
        "f1":           f1,
        "replacement_rate": total_replacement_rate / batch_count,
        "valid_candidate_rate": total_valid_candidate_rate / batch_count,
        "disc_confidence": total_disc_confidence / batch_count,
        "rtd_entropy": total_rtd_entropy / batch_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Checkpoint saving
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: CKTNElectra,
    epoch: int,
    metrics: Dict[str, float],
    is_best: bool,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    global_step: int,
    best_f1: float,
    report: List[Dict],
):
    """Save discriminator and generator state dicts."""
    suffix = f"epoch{epoch}"

    disc_path = DISC_DIR / f"discriminator_{suffix}.pt"
    gen_path  = GEN_DIR  / f"generator_{suffix}.pt"
    state_path = STATE_DIR / f"training_state_{suffix}.pt"

    disc_state = {
        "encoder":              model.discriminator.encoder.state_dict(),
        "rtd_head":             model.discriminator.rtd_head.state_dict(),
        "shared_embeddings":    model.shared_embeddings.state_dict(),
        "epoch":                epoch,
        "metrics":              metrics,
    }
    torch.save(disc_state, disc_path)

    gen_state = {
        "generator":  model.generator.state_dict(),
        "epoch":      epoch,
        "metrics":    metrics,
    }
    torch.save(gen_state, gen_path)

    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_f1": best_f1,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "report": report,
    }
    torch.save(training_state, state_path)
    torch.save(training_state, STATE_DIR / "training_state_latest.pt")

    if is_best:
        torch.save(disc_state, DISC_DIR / "discriminator_best.pt")
        torch.save(gen_state,  GEN_DIR  / "generator_best.pt")
        print(f"  [Checkpoint] Best model saved (F1={metrics['f1']:.4f})")

    print(f"  [Checkpoint] Saved: {disc_path.name}, {gen_path.name}")


def checkpoint_epoch(path: Path, prefix: str) -> int:
    match = re.fullmatch(rf"{prefix}_epoch(\d+)", path.stem)
    if match is None:
        return -1
    return int(match.group(1))


def available_checkpoint_epochs() -> List[int]:
    disc_epochs = {
        checkpoint_epoch(path, "discriminator")
        for path in DISC_DIR.glob("discriminator_epoch*.pt")
    }
    gen_epochs = {
        checkpoint_epoch(path, "generator")
        for path in GEN_DIR.glob("generator_epoch*.pt")
    }
    return sorted(epoch for epoch in disc_epochs & gen_epochs if epoch > 0)


def resolve_resume_epoch(resume: str) -> int:
    epochs = available_checkpoint_epochs()
    if not epochs:
        raise FileNotFoundError(
            f"No epoch checkpoints found in {DISC_DIR} and {GEN_DIR}."
        )
    if resume == "latest":
        return epochs[-1]
    if resume.isdigit():
        epoch = int(resume)
        if epoch in epochs:
            return epoch
    raise ValueError(
        f"Unknown resume target '{resume}'. Available epochs: {epochs}"
    )


def load_epoch_checkpoint(model: CKTNElectra, epoch: int, device: str):
    disc_path = DISC_DIR / f"discriminator_epoch{epoch}.pt"
    gen_path = GEN_DIR / f"generator_epoch{epoch}.pt"

    disc_state = torch.load(disc_path, map_location=device)
    gen_state = torch.load(gen_path, map_location=device)

    model.discriminator.encoder.load_state_dict(disc_state["encoder"])
    model.discriminator.rtd_head.load_state_dict(disc_state["rtd_head"])
    model.shared_embeddings.load_state_dict(disc_state["shared_embeddings"])
    model.generator.load_state_dict(gen_state["generator"])
    print(f"[Resume] Loaded model weights from epoch {epoch}.")


def load_report_for_resume(epoch: int) -> List[Dict]:
    if not REPORT_PATH.exists():
        return []
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    return [row for row in report if row["epoch"] <= epoch]


def best_f1_from_report(report: List[Dict]) -> float:
    return max((row["f1"] for row in report), default=-1.0)


def load_training_state(
    epoch: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    device: str,
) -> Dict:
    state_path = STATE_DIR / f"training_state_epoch{epoch}.pt"
    if not state_path.exists():
        return {}

    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    print(f"[Resume] Loaded optimizer/scheduler state from epoch {epoch}.")
    return state


def advance_scheduler(lr_scheduler, optimizer: torch.optim.Optimizer, steps: int):
    if steps <= 0:
        return
    lr_scheduler.last_epoch = steps
    if hasattr(lr_scheduler, "_step_count"):
        lr_scheduler._step_count = steps + 1

    lrs = [
        base_lr * lr_lambda(steps)
        for base_lr, lr_lambda in zip(lr_scheduler.base_lrs, lr_scheduler.lr_lambdas)
    ]
    for param_group, lr in zip(optimizer.param_groups, lrs):
        param_group["lr"] = lr
    if hasattr(lr_scheduler, "_last_lr"):
        lr_scheduler._last_lr = lrs


# ─────────────────────────────────────────────────────────────────────────────
# 7. Graph plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_and_save_graphs(report: List[Dict]):
    """
    Generate and save training graphs after all epochs.
    Plots losses, RTD quality, replacement calibration, and gradient balance.
    Font size ~20, labels in English, no bold.
    """
    epochs = [r["epoch"] for r in report]

    plot_specs = [
        ("avg_mlm_loss", "MLM Loss", "MLM Loss per Epoch", "Loss"),
        ("avg_rtd_loss", "RTD Loss", "RTD Loss per Epoch", "Loss"),
        ("accuracy", "Accuracy", "RTD Accuracy per Epoch", "Accuracy"),
        ("f1", "F1 Score", "RTD Replaced-Token F1 per Epoch", "F1"),
        ("replacement_rate", "Replacement Rate", "Replacement Rate per Epoch", "Rate"),
        ("valid_candidate_rate", "Valid Candidate Rate", "Valid Candidates per Epoch", "Rate"),
        ("disc_confidence", "Confidence", "Discriminator Confidence per Epoch", "Confidence"),
        ("rtd_entropy", "Entropy", "RTD Prediction Entropy per Epoch", "Entropy"),
        ("grad_norm_ratio", "Grad Ratio", "Generator/Discriminator Grad Ratio", "Ratio"),
    ]

    n_cols = 3
    n_rows = math.ceil(len(plot_specs) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle("CKTN-ELECTRA Training Metrics", fontsize=22)

    for ax, (key, label, title, ylabel) in zip(axes.flat, plot_specs):
        values = [r[key] for r in report]
        ax.plot(epochs, values, marker="o", linewidth=2, markersize=7, label=label)
        ax.set_title(title,  fontsize=20)
        ax.set_xlabel("Epoch", fontsize=20)
        ax.set_ylabel(ylabel,  fontsize=20)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=16)

    for ax in axes.flat[len(plot_specs):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = GRAPHS_DIR / "training_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Graph] Saved: {out_path}")

    for key, label, title, ylabel in plot_specs:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        values = [r[key] for r in report]
        ax2.plot(epochs, values, marker="o", linewidth=2, markersize=7)
        ax2.set_title(title,  fontsize=20)
        ax2.set_xlabel("Epoch", fontsize=20)
        ax2.set_ylabel(ylabel,  fontsize=20)
        ax2.tick_params(labelsize=16)
        ax2.grid(True, alpha=0.4)
        plt.tight_layout()
        single_path = GRAPHS_DIR / f"{key}.png"
        plt.savefig(single_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Graph] Saved: {single_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CKTN-ELECTRA continued pre-training")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size for both training and evaluation (default: 8)",
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Rebuild tokenized train/dev chunk caches before training",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from 'latest' or a specific epoch number, e.g. --resume 4",
    )
    return parser.parse_args()


def module_grad_norm(module: nn.Module) -> float:
    squared_norm = 0.0
    for param in module.parameters():
        if param.grad is not None:
            param_norm = param.grad.detach().norm(2).item()
            squared_norm += param_norm * param_norm
    return math.sqrt(squared_norm)


def generator_discriminator_grad_ratio(model: CKTNElectra) -> float:
    gen_norm = module_grad_norm(model.generator)
    disc_norm = module_grad_norm(model.discriminator)
    return gen_norm / disc_norm if disc_norm > 0.0 else 0.0


def main():
    args       = parse_args()
    batch_size = args.batch_size

    print("=" * 70)
    print("CKTN-ELECTRA — Training Script")
    print(f"Batch size: {batch_size}")
    if args.resume is not None:
        print(f"Resume: {args.resume}")
    print("=" * 70)

    # ── 8.1 Tokenizer & vocab check ──────────────────────────────────────────
    tokenizer = load_tokenizer_and_check()

    # ── 8.2 Load cached chunks or prepare data ───────────────────────────────
    train_chunks = load_or_build_chunks(
        "train", TRAIN_FILES, tokenizer, SEQ_LEN, args.rebuild_cache
    )
    dev_chunks = load_or_build_chunks(
        "dev", DEV_FILES, tokenizer, SEQ_LEN, args.rebuild_cache
    )

    train_dataset = ELECTRADataset(train_chunks, tokenizer, SEQ_LEN, MASK_RATE)
    dev_dataset   = ELECTRADataset(dev_chunks,   tokenizer, SEQ_LEN, MASK_RATE)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE == "cuda"),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE == "cuda"),
    )

    # ── 8.3 Model ─────────────────────────────────────────────────────────────
    print("\n[Model] Instantiating CKTNElectra (load_pretrained=True) ...")
    model = CKTNElectra(load_pretrained=True, tokenizer=tokenizer)
    model.to(DEVICE)
    print(f"[Model] Running on device: {DEVICE}")

    # ── 8.4 Optimizer & LR scheduler ─────────────────────────────────────────
    param_groups    = get_parameter_groups(model, WEIGHT_DECAY)
    optimizer       = torch.optim.AdamW(param_groups, lr=LR)

    steps_per_epoch = len(train_loader)
    total_steps     = TOTAL_EPOCHS * steps_per_epoch
    warmup_steps    = int(total_steps * WARMUP_RATIO)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── 8.5 Lambda scheduler ─────────────────────────────────────────────────
    lambda_scheduler = LinearLambdaScheduler(
        lambda_max        = LAMBDA_MAX,
        zero_until_epoch  = TRAINING_CONFIG["lambda_zero_until_epoch"],   # 2
        ramp_until_epoch  = TRAINING_CONFIG["lambda_ramp_until_epoch"],   # 3
        total_epochs      = TOTAL_EPOCHS,
        steps_per_epoch   = steps_per_epoch,
    )

    print(f"\n[Training] Steps per epoch : {steps_per_epoch}")
    print(f"[Training] Total steps     : {total_steps} | Warmup: {warmup_steps}")
    print(f"[Training] Lambda schedule : 0 until epoch {TRAINING_CONFIG['lambda_zero_until_epoch']}, "
          f"ramp to {LAMBDA_MAX} by epoch {TRAINING_CONFIG['lambda_ramp_until_epoch']}, "
          f"fixed afterwards.\n")

    # ── 8.6 Training state ────────────────────────────────────────────────────
    report: List[Dict] = []
    best_f1     = -1.0
    global_step = 0
    start_epoch = 1

    if args.resume is not None:
        resume_epoch = resolve_resume_epoch(args.resume)
        load_epoch_checkpoint(model, resume_epoch, DEVICE)
        report = load_report_for_resume(resume_epoch)
        best_f1 = best_f1_from_report(report)
        global_step = resume_epoch * steps_per_epoch
        training_state = load_training_state(
            resume_epoch, optimizer, lr_scheduler, DEVICE
        )
        if training_state:
            global_step = training_state["global_step"]
            best_f1 = training_state["best_f1"]
            report = training_state["report"]
        if not training_state:
            advance_scheduler(lr_scheduler, optimizer, global_step)
        start_epoch = resume_epoch + 1
        print(
            f"[Resume] Continuing at epoch {start_epoch}/{TOTAL_EPOCHS} "
            f"from global_step={global_step}."
        )

    model.train()

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{TOTAL_EPOCHS}")
        print(f"{'='*60}")

        epoch_mlm_loss = 0.0
        epoch_rtd_loss = 0.0
        epoch_replacement_rate = 0.0
        epoch_grad_ratio = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc="  Training", leave=True)
        for batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            lam   = lambda_scheduler.get_lambda(global_step)

            outputs = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                token_type_ids = batch["token_type_ids"],
                labels         = batch["labels"],
                lam            = lam,
            )

            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            grad_ratio = generator_discriminator_grad_ratio(model)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step()
            lr_scheduler.step()

            epoch_mlm_loss += outputs["loss_mlm"].item()
            epoch_rtd_loss += outputs["loss_disc"].item()
            epoch_replacement_rate += outputs["replacement_rate"].item()
            epoch_grad_ratio += grad_ratio
            n_batches += 1
            global_step += 1

            pbar.set_postfix({
                "λ":    f"{lam:.1f}",
                "loss": f"{loss.item():.4f}",
                "mlm":  f"{outputs['loss_mlm'].item():.4f}",
                "rtd":  f"{outputs['loss_disc'].item():.4f}",
                "rep":  f"{outputs['replacement_rate'].item():.3f}",
            })

        # ── 8.7 Evaluation ────────────────────────────────────────────────────
        eval_metrics = evaluate(model, dev_loader, DEVICE, lam=LAMBDA_MAX)

        avg_mlm = epoch_mlm_loss / max(n_batches, 1)
        avg_rtd = epoch_rtd_loss / max(n_batches, 1)
        avg_replacement_rate = epoch_replacement_rate / max(n_batches, 1)
        avg_grad_ratio = epoch_grad_ratio / max(n_batches, 1)

        epoch_report = {
            "epoch":        epoch,
            "avg_mlm_loss": round(avg_mlm,                    6),
            "avg_rtd_loss": round(avg_rtd,                    6),
            "accuracy":     round(eval_metrics["accuracy"],   6),
            "f1":           round(eval_metrics["f1"],         6),
            "replacement_rate": round(eval_metrics["replacement_rate"], 6),
            "valid_candidate_rate": round(eval_metrics["valid_candidate_rate"], 6),
            "disc_confidence": round(eval_metrics["disc_confidence"], 6),
            "rtd_entropy": round(eval_metrics["rtd_entropy"], 6),
            "grad_norm_ratio": round(avg_grad_ratio, 6),
        }
        report.append(epoch_report)

        print(
            f"[Epoch {epoch}] "
            f"Train MLM={avg_mlm:.4f} | Train RTD={avg_rtd:.4f} | "
            f"Train Rep={avg_replacement_rate:.3f} | "
            f"Dev Acc={eval_metrics['accuracy']:.4f} | "
            f"Dev F1={eval_metrics['f1']:.4f} | "
            f"Dev Rep={eval_metrics['replacement_rate']:.3f} | "
            f"Dev Conf={eval_metrics['disc_confidence']:.3f}"
        )

        # ── 8.8 Checkpoint ────────────────────────────────────────────────────
        is_best = eval_metrics["f1"] > best_f1
        if is_best:
            best_f1 = eval_metrics["f1"]

        save_checkpoint(
            model,
            epoch,
            epoch_report,
            is_best,
            optimizer,
            lr_scheduler,
            global_step,
            best_f1,
            report,
        )

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  [Report] Updated: {REPORT_PATH}")

    # ── 8.9 Final graphs ──────────────────────────────────────────────────────
    print("\n[Graph] Generating training graphs ...")
    plot_and_save_graphs(report)

    print("\n[Done] Training complete.")
    print(f"  Best F1 on dev : {best_f1:.4f}")
    print(f"  Report saved to: {REPORT_PATH}")
    print(f"  Graphs saved to: {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
