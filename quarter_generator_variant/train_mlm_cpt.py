import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
CORPUS_DIR = ROOT / "corpus"
DEFAULT_OUTPUT_ROOT = HERE / "cpt_baselines"
DEFAULT_CACHE_DIR = HERE / "mlm_cache"

BASELINE_MODELS = {
    "rembert": "google/rembert",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continued pre-training for baseline encoders with MLM."
    )
    parser.add_argument(
        "--model_key",
        choices=sorted(BASELINE_MODELS.keys()),
        required=True,
        help="Baseline model to continue pre-train.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Override the Hugging Face checkpoint for --model_key.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--corpus_dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["title", "summary", "tags", "content"],
        help="JSON fields to concatenate for MLM text.",
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--min_tokens", type=int, default=8)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluate MLM loss/perplexity without running CPT.",
    )
    parser.add_argument(
        "--save_each_epoch",
        action="store_true",
        help="Also save checkpoint-epochN directories.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenizer_uses_fast(model_key: str) -> bool:
    return model_key != "rembert"


def field_to_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def record_to_text(record: Dict, fields: List[str]) -> str:
    pieces = [field_to_text(record.get(field, "")) for field in fields]
    return " ".join(piece for piece in pieces if piece)


def iter_split_texts(corpus_dir: Path, split: str, fields: List[str]) -> Iterable[str]:
    split_dir = corpus_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing corpus split directory: {split_dir}")

    paths = sorted(split_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files found under: {split_dir}")

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            text = record_to_text(record, fields)
            if text:
                yield text


def cache_path(args: argparse.Namespace, split: str, model_id: str) -> Path:
    safe_model = model_id.replace("/", "__")
    fields = "-".join(args.fields)
    name = (
        f"{args.model_key}_{safe_model}_{split}_seq{args.max_seq_length}"
        f"_min{args.min_tokens}_{fields}.pt"
    )
    return args.cache_dir / name


def build_token_chunks(
    tokenizer,
    texts: Iterable[str],
    max_seq_length: int,
    min_tokens: int,
) -> List[List[int]]:
    special_tokens = count_manual_special_tokens(tokenizer)
    block_size = max_seq_length - special_tokens
    if block_size <= 0:
        raise ValueError(
            f"--max_seq_length {max_seq_length} is too small for this tokenizer."
        )

    examples: List[List[int]] = []
    for text in tqdm(texts, desc="Tokenizing", unit="doc"):
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        for start in range(0, len(token_ids), block_size):
            chunk = token_ids[start : start + block_size]
            if len(chunk) < min_tokens:
                continue
            examples.append(add_special_tokens(tokenizer, chunk, max_seq_length))
    return examples


def add_special_tokens(tokenizer, token_ids: List[int], max_seq_length: int) -> List[int]:
    prefix = []
    suffix = []
    if getattr(tokenizer, "cls_token_id", None) is not None:
        prefix.append(tokenizer.cls_token_id)
    elif getattr(tokenizer, "bos_token_id", None) is not None:
        prefix.append(tokenizer.bos_token_id)

    if getattr(tokenizer, "sep_token_id", None) is not None:
        suffix.append(tokenizer.sep_token_id)
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        suffix.append(tokenizer.eos_token_id)

    room = max_seq_length - len(prefix) - len(suffix)
    return prefix + token_ids[:room] + suffix


def count_manual_special_tokens(tokenizer) -> int:
    count = 0
    if getattr(tokenizer, "cls_token_id", None) is not None:
        count += 1
    elif getattr(tokenizer, "bos_token_id", None) is not None:
        count += 1

    if getattr(tokenizer, "sep_token_id", None) is not None:
        count += 1
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        count += 1
    return count


def load_or_build_examples(
    args: argparse.Namespace,
    tokenizer,
    split: str,
    model_id: str,
) -> List[List[int]]:
    path = cache_path(args, split, model_id)
    if path.exists() and not args.rebuild_cache:
        payload = torch.load(path, map_location="cpu")
        print(f"[Cache] Loaded {split} examples from {path}")
        return payload["examples"]

    print(f"[Data] Building {split} examples from {args.corpus_dir / split}")
    examples = build_token_chunks(
        tokenizer,
        iter_split_texts(args.corpus_dir, split, args.fields),
        args.max_seq_length,
        args.min_tokens,
    )
    if not examples:
        raise ValueError(f"No MLM examples built for split: {split}")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "examples": examples,
            "model_id": model_id,
            "split": split,
            "fields": args.fields,
            "max_seq_length": args.max_seq_length,
            "min_tokens": args.min_tokens,
        },
        path,
    )
    print(f"[Cache] Saved {len(examples)} {split} examples to {path}")
    return examples


class MLMDataset(Dataset):
    def __init__(self, examples: List[List[int]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return {"input_ids": self.examples[index]}


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: str,
    use_amp: bool,
    grad_accum_steps: int,
    max_grad_norm: float,
    logging_steps: int,
    epoch: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    update_step = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch} train", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum_steps

        should_step = step % grad_accum_steps == 0 or step == len(dataloader)
        if should_step:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

            if logging_steps > 0 and update_step % logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                progress.set_postfix(loss=f"{total_loss / step:.4f}", lr=f"{lr:.2e}")

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: str, use_amp: bool) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(**batch)
        total_loss += outputs.loss.item()

    loss = total_loss / max(len(dataloader), 1)
    perplexity = math.exp(loss) if loss < 20 else float("inf")
    return {"loss": loss, "perplexity": perplexity}


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_checkpoint(path: Path, model, tokenizer):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[Checkpoint] Saved {path}")


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / args.model_key
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.fp16 and device == "cuda"

    model_id = args.model_name_or_path or BASELINE_MODELS[args.model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=tokenizer_uses_fast(args.model_key),
    )
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(device)

    print(f"[Model] {args.model_key}: {model_id}")
    print(f"[Device] {device} | fp16={use_amp}")

    train_examples = load_or_build_examples(args, tokenizer, "train", model_id)
    dev_examples = load_or_build_examples(args, tokenizer, "dev", model_id)
    print(f"[Data] Train chunks: {len(train_examples)}")
    print(f"[Data] Dev chunks: {len(dev_examples)}")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    train_loader = DataLoader(
        MLMDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
    )
    dev_loader = DataLoader(
        MLMDataset(dev_examples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
    )

    if args.eval_only:
        metrics = evaluate(model, dev_loader, device, use_amp)
        row = {
            "model_key": args.model_key,
            "model_id": model_id,
            "eval_loss": round(metrics["loss"], 6),
            "perplexity": round(metrics["perplexity"], 6)
            if math.isfinite(metrics["perplexity"])
            else "inf",
        }
        save_json(args.output_dir / "eval_only_report.json", row)
        print(
            f"[Eval only] Eval loss={metrics['loss']:.4f} | "
            f"PPL={metrics['perplexity']:.4f}"
        )
        return

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.epochs * update_steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    report = []
    best_eval_loss = float("inf")
    best_dir = args.output_dir / "best"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            args.grad_accum_steps,
            args.max_grad_norm,
            args.logging_steps,
            epoch,
        )
        metrics = evaluate(model, dev_loader, device, use_amp)
        row = {
            "epoch": epoch,
            "model_key": args.model_key,
            "model_id": model_id,
            "train_loss": round(train_loss, 6),
            "eval_loss": round(metrics["loss"], 6),
            "perplexity": round(metrics["perplexity"], 6)
            if math.isfinite(metrics["perplexity"])
            else "inf",
        }
        report.append(row)
        save_json(args.output_dir / "report.json", report)
        print(
            f"[Epoch {epoch}] Train loss={train_loss:.4f} | "
            f"Eval loss={metrics['loss']:.4f} | PPL={metrics['perplexity']:.2f}"
        )

        if metrics["loss"] < best_eval_loss:
            best_eval_loss = metrics["loss"]
            save_checkpoint(best_dir, model, tokenizer)

        if args.save_each_epoch:
            save_checkpoint(args.output_dir / f"checkpoint-epoch{epoch}", model, tokenizer)

    save_checkpoint(args.output_dir, model, tokenizer)
    save_json(
        args.output_dir / "training_args.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    print(f"\n[Done] Best eval loss: {best_eval_loss:.4f}")
    print(f"[Done] Best checkpoint: {best_dir}")
    print(f"[Done] Final checkpoint: {args.output_dir}")


if __name__ == "__main__":
    main()
