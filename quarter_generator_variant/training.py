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
    cd main
    python training.py
"""

import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
GRAPHS_DIR   = CKPT_DIR / "graphs"
DISC_DIR     = CKPT_DIR / "discriminator"
GEN_DIR      = CKPT_DIR / "generator"
REPORT_PATH  = CKPT_DIR / "report.json"

for d in (CKPT_DIR, GRAPHS_DIR, DISC_DIR, GEN_DIR):
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
        self.vocab_size = tokenizer.vocab_size

        # Identify special token ids to avoid masking them
        self.special_ids = set(tokenizer.all_special_ids)

    def __len__(self) -> int:
        return len(self.chunks)

    def _apply_mlm(
        self, token_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Apply BERT-style MLM masking:
            80% → [MASK]
            10% → random token
            10% → unchanged
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
        n_mask = max(1, int(round(len(eligible) * self.mask_rate)))
        mask_indices = np.random.choice(eligible, size=n_mask, replace=False).tolist()

        for idx in mask_indices:
            labels[idx] = token_ids[idx]
            r = np.random.random()
            if r < 0.80:
                masked_ids[idx] = self.mask_id
            elif r < 0.90:
                # Random token (from whole vocab)
                masked_ids[idx] = np.random.randint(0, self.vocab_size)
            # else: keep original (10%)

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
    Returns dict with: avg_mlm_loss, avg_rtd_loss, accuracy, f1
    RTD accuracy/F1 are computed only over positions that were originally masked.
    """
    model.eval()

    total_mlm_loss  = 0.0
    total_rtd_loss  = 0.0
    all_rtd_preds   = []
    all_rtd_labels  = []
    n_batches       = 0

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
        n_batches += 1

        # Collect RTD predictions for masked positions
        if out["disc_logits"] is not None:
            disc_logits = out["disc_logits"].squeeze(-1)  # [B, L]
            preds = (torch.sigmoid(disc_logits) > 0.5).long()

            labels_tensor = batch["labels"]  # [B, L]   -100 = not masked
            is_masked = labels_tensor != -100  # [B, L]

            attn_mask = batch["attention_mask"].bool()  # [B, L]
            eval_mask = is_masked & attn_mask

            if eval_mask.any():
                gen_logits   = out["gen_logits"]           # [B, L, V]
                gen_pred_ids = gen_logits.argmax(dim=-1)   # [B, L]

                original_at_mask = labels_tensor.clone()
                original_at_mask[~is_masked] = batch["input_ids"][~is_masked]

                rtd_true = (gen_pred_ids != original_at_mask).long()  # [B, L]

                preds_flat  = preds[eval_mask].cpu().numpy()
                labels_flat = rtd_true[eval_mask].cpu().numpy()

                all_rtd_preds.append(preds_flat)
                all_rtd_labels.append(labels_flat)

    avg_mlm = total_mlm_loss / max(n_batches, 1)
    avg_rtd = total_rtd_loss / max(n_batches, 1)

    if all_rtd_preds:
        y_pred = np.concatenate(all_rtd_preds)
        y_true = np.concatenate(all_rtd_labels)
        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    else:
        acc, f1 = 0.0, 0.0

    model.train()
    return {
        "avg_mlm_loss": avg_mlm,
        "avg_rtd_loss": avg_rtd,
        "accuracy":     acc,
        "f1":           f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Checkpoint saving
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: CKTNElectra,
    epoch: int,
    metrics: Dict[str, float],
    is_best: bool,
):
    """Save discriminator and generator state dicts."""
    suffix = f"epoch{epoch}"

    disc_path = DISC_DIR / f"discriminator_{suffix}.pt"
    gen_path  = GEN_DIR  / f"generator_{suffix}.pt"

    disc_state = {
        "encoder":              model.discriminator.encoder.state_dict(),
        "rtd_head":             model.discriminator.rtd_head.state_dict(),
        "embeddings_project":   model.discriminator.embeddings_project.state_dict(),
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

    if is_best:
        torch.save(disc_state, DISC_DIR / "discriminator_best.pt")
        torch.save(gen_state,  GEN_DIR  / "generator_best.pt")
        print(f"  [Checkpoint] Best model saved (F1={metrics['f1']:.4f})")

    print(f"  [Checkpoint] Saved: {disc_path.name}, {gen_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Graph plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_and_save_graphs(report: List[Dict]):
    """
    Generate and save training graphs after all epochs.
    Plots: MLM Loss, RTD Loss, Accuracy, F1.
    Font size ~20, labels in English, no bold.
    """
    epochs = [r["epoch"] for r in report]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CKTN-ELECTRA Training Metrics", fontsize=22)

    plot_specs = [
        ("avg_mlm_loss", "MLM Loss",  "MLM Loss per Epoch",     "Loss"),
        ("avg_rtd_loss", "RTD Loss",  "RTD Loss per Epoch",     "Loss"),
        ("accuracy",     "Accuracy",  "RTD Accuracy per Epoch", "Accuracy"),
        ("f1",           "F1 Score",  "RTD F1 Score per Epoch", "F1"),
    ]

    for ax, (key, label, title, ylabel) in zip(axes.flat, plot_specs):
        values = [r[key] for r in report]
        ax.plot(epochs, values, marker="o", linewidth=2, markersize=7, label=label)
        ax.set_title(title,  fontsize=20)
        ax.set_xlabel("Epoch", fontsize=20)
        ax.set_ylabel(ylabel,  fontsize=20)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=16)

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
    return parser.parse_args()


def main():
    args       = parse_args()
    batch_size = args.batch_size

    print("=" * 70)
    print("CKTN-ELECTRA — Training Script")
    print(f"Batch size: {batch_size}")
    print("=" * 70)

    # ── 8.1 Tokenizer & vocab check ──────────────────────────────────────────
    tokenizer = load_tokenizer_and_check()

    # ── 8.2 Load & prepare data ──────────────────────────────────────────────
    print("\n[Data] Loading training data ...")
    train_contents = read_contents(TRAIN_FILES)
    train_chunks   = tokenize_and_chunk(train_contents, tokenizer, SEQ_LEN)

    print("\n[Data] Loading dev data ...")
    dev_contents = read_contents(DEV_FILES)
    dev_chunks   = tokenize_and_chunk(dev_contents, tokenizer, SEQ_LEN)

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
    model = CKTNElectra(load_pretrained=True)
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

    model.train()

    for epoch in range(1, TOTAL_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{TOTAL_EPOCHS}")
        print(f"{'='*60}")

        epoch_mlm_loss = 0.0
        epoch_rtd_loss = 0.0
        n_batches      = 0

        pbar = tqdm(train_loader, desc=f"  Training", leave=True)
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
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step()
            lr_scheduler.step()

            epoch_mlm_loss += outputs["loss_mlm"].item()
            epoch_rtd_loss += outputs["loss_disc"].item()
            n_batches      += 1
            global_step    += 1

            pbar.set_postfix({
                "λ":    f"{lam:.1f}",
                "loss": f"{loss.item():.4f}",
                "mlm":  f"{outputs['loss_mlm'].item():.4f}",
                "rtd":  f"{outputs['loss_disc'].item():.4f}",
            })

        # ── 8.7 Evaluation ────────────────────────────────────────────────────
        eval_metrics = evaluate(model, dev_loader, DEVICE, lam=LAMBDA_MAX)

        avg_mlm = epoch_mlm_loss / max(n_batches, 1)
        avg_rtd = epoch_rtd_loss / max(n_batches, 1)

        epoch_report = {
            "epoch":        epoch,
            "avg_mlm_loss": round(avg_mlm,                    6),
            "avg_rtd_loss": round(avg_rtd,                    6),
            "accuracy":     round(eval_metrics["accuracy"],   6),
            "f1":           round(eval_metrics["f1"],         6),
        }
        report.append(epoch_report)

        print(
            f"[Epoch {epoch}] "
            f"Train MLM={avg_mlm:.4f} | Train RTD={avg_rtd:.4f} | "
            f"Dev Acc={eval_metrics['accuracy']:.4f} | Dev F1={eval_metrics['f1']:.4f}"
        )

        # ── 8.8 Checkpoint ────────────────────────────────────────────────────
        is_best = eval_metrics["f1"] > best_f1
        if is_best:
            best_f1 = eval_metrics["f1"]

        save_checkpoint(model, epoch, epoch_report, is_best)

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