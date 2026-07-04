import argparse
import inspect
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from downstream_common import (
    BENCHMARK_MODELS,
    DEFAULT_DISCRIMINATOR_CKPT,
    DOWNSTREAM_DIR,
    join_fields,
    load_encoder,
    load_tokenizer,
    masked_mean_pool,
    model_id_for_key,
    read_records,
    save_json,
    serializable_args,
    set_seed,
)


class CategoryDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        label_to_id: Dict[str, int],
        text_fields: List[str],
    ):
        self.examples = []
        for record in records:
            label = record.get("category", "").strip()
            text = join_fields(record, text_fields)
            if label in label_to_id and text:
                self.examples.append(
                    {"text": text, "label": label_to_id[label]}
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        return self.examples[index]


class CategoryCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [item["label"] for item in batch], dtype=torch.long
        )
        return encoded


class CategoryClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        dropout: float,
        pooling: str,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.accepts_token_type_ids = (
            "token_type_ids" in inspect.signature(encoder.forward).parameters
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.accepts_token_type_ids:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        pooled = outputs.last_hidden_state[:, 0]
        if self.pooling == "mean":
            pooled = masked_mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune CKTN-ELECTRA for category classification"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DISCRIMINATOR_CKPT)
    parser.add_argument(
        "--model_key",
        choices=list(BENCHMARK_MODELS.keys()),
        default="cktn",
        help="Encoder benchmark to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--text_fields", nargs="+", default=["title", "content"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="cls",
        help="Sequence pooling strategy for document classification",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class_weighting",
        choices=["none", "balanced"],
        default="balanced",
        help="Use balanced class weights for category cross-entropy",
    )
    parser.add_argument(
        "--diagnostics_top_k",
        type=int,
        default=20,
        help="Number of most common labels to print in diagnostics",
    )
    return parser.parse_args()


def build_label_map(records: List[Dict]) -> Dict[str, int]:
    labels = sorted(
        {
            record.get("category", "").strip()
            for record in records
            if record.get("category", "").strip()
        }
    )
    return {label: index for index, label in enumerate(labels)}


def category_counter(records: List[Dict]) -> Counter:
    return Counter(
        record.get("category", "").strip()
        for record in records
        if record.get("category", "").strip()
    )


def dataset_label_ids(dataset: CategoryDataset) -> List[int]:
    return [example["label"] for example in dataset.examples]


def print_label_diagnostics(
    label_to_id: Dict[str, int],
    train_records: List[Dict],
    dev_records: List[Dict],
    train_dataset: CategoryDataset,
    dev_dataset: CategoryDataset,
    top_k: int,
) -> Dict:
    train_counter = category_counter(train_records)
    dev_counter = category_counter(dev_records)
    unknown_dev = sorted(set(dev_counter) - set(label_to_id))
    train_ids = sorted(set(dataset_label_ids(train_dataset)))
    dev_ids = sorted(set(dataset_label_ids(dev_dataset)))

    train_top_label, train_top_count = train_counter.most_common(1)[0]
    dev_top_label, dev_top_count = dev_counter.most_common(1)[0]
    diagnostics = {
        "label_to_id": label_to_id,
        "train_label_ids": train_ids,
        "dev_label_ids": dev_ids,
        "raw_train_label_count": len(train_counter),
        "raw_dev_label_count": len(dev_counter),
        "unknown_dev_labels": unknown_dev,
        "unknown_dev_examples": sum(dev_counter[label] for label in unknown_dev),
        "train_distribution_top": train_counter.most_common(top_k),
        "dev_distribution_top": dev_counter.most_common(top_k),
        "train_top_ratio": train_top_count / max(sum(train_counter.values()), 1),
        "dev_top_ratio": dev_top_count / max(sum(dev_counter.values()), 1),
        "train_top_label": train_top_label,
        "dev_top_label": dev_top_label,
    }

    print("[Diagnostics] label_to_id:", label_to_id)
    print("[Diagnostics] train label ids:", train_ids)
    print("[Diagnostics] dev label ids:", dev_ids)
    print("[Diagnostics] unknown dev labels:", unknown_dev)
    print("[Diagnostics] unknown dev examples:", diagnostics["unknown_dev_examples"])
    print("[Diagnostics] train label distribution:", train_counter.most_common(top_k))
    print("[Diagnostics] dev label distribution:", dev_counter.most_common(top_k))
    print(
        f"[Diagnostics] train top ratio: {train_top_label} "
        f"{diagnostics['train_top_ratio']:.4f}"
    )
    print(
        f"[Diagnostics] dev top ratio: {dev_top_label} "
        f"{diagnostics['dev_top_ratio']:.4f}"
    )
    return diagnostics


def balanced_class_weights(dataset: CategoryDataset, num_labels: int) -> torch.Tensor:
    labels = dataset_label_ids(dataset)
    counter = Counter(labels)
    total = len(labels)
    weights = [
        total / (num_labels * counter[index])
        if counter[index] > 0
        else 0.0
        for index in range(num_labels)
    ]
    return torch.tensor(weights, dtype=torch.float)


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set:
    return {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }


def classifier_optimizer_diagnostics(
    model: CategoryClassifier,
    optimizer: torch.optim.Optimizer,
) -> List[Dict]:
    optimizer_ids = optimizer_parameter_ids(optimizer)
    rows = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            row = {
                "name": name,
                "requires_grad": param.requires_grad,
                "in_optimizer": id(param) in optimizer_ids,
                "norm": float(param.detach().norm().item()),
            }
            rows.append(row)
            print(
                "[Diagnostics] classifier param:",
                row["name"],
                "requires_grad=",
                row["requires_grad"],
                "in_optimizer=",
                row["in_optimizer"],
                "norm=",
                f"{row['norm']:.6f}",
            )
    return rows


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, dataloader, device: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []

    for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
        batch = move_batch(batch, device)
        outputs = model(**batch)
        total_loss += outputs["loss"].item()
        pred = outputs["logits"].argmax(dim=-1)
        preds.extend(pred.cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = {
        "loss": avg_loss,
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
    }
    model.train()
    return metrics


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: str,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    classifier_before = None
    classifier_update_reported = False

    for step, batch in enumerate(tqdm(dataloader, desc="  Training", leave=True), start=1):
        batch = move_batch(batch, device)
        outputs = model(**batch)
        loss = outputs["loss"] / grad_accum_steps
        if classifier_before is None:
            classifier_before = model.classifier.weight.detach().clone()
        loss.backward()
        total_loss += outputs["loss"].item()

        should_step = step % grad_accum_steps == 0 or step == len(dataloader)
        if should_step:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if not classifier_update_reported:
                classifier_after = model.classifier.weight.detach().clone()
                classifier_change = (
                    classifier_after - classifier_before
                ).abs().sum().item()
                print(
                    "[Diagnostics] classifier weight change after first step:",
                    f"{classifier_change:.8f}",
                )
                classifier_update_reported = True
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(len(dataloader), 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.output_dir is None:
        args.output_dir = (
            DOWNSTREAM_DIR
            / "category_classification"
            / args.pooling
            / args.model_key
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_key)
    train_records = read_records("train")
    dev_records = read_records("dev")
    label_to_id = build_label_map(train_records)
    id_to_label = {str(index): label for label, index in label_to_id.items()}

    collator = CategoryCollator(tokenizer, args.max_length)
    train_dataset = CategoryDataset(train_records, label_to_id, args.text_fields)
    dev_dataset = CategoryDataset(dev_records, label_to_id, args.text_fields)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
    )

    print(f"[Data] Train examples: {len(train_dataset)}")
    print(f"[Data] Dev examples: {len(dev_dataset)}")
    print(f"[Data] Labels: {len(label_to_id)}")
    diagnostics = print_label_diagnostics(
        label_to_id,
        train_records,
        dev_records,
        train_dataset,
        dev_dataset,
        args.diagnostics_top_k,
    )
    diagnostics["model_key"] = args.model_key
    diagnostics["model_id"] = model_id_for_key(args.model_key)
    diagnostics["pooling"] = args.pooling
    save_json(args.output_dir / "label_diagnostics.json", diagnostics)

    encoder = load_encoder(args.model_key, args.checkpoint)
    print(f"[Model] Loaded encoder: {args.model_key} ({model_id_for_key(args.model_key)})")
    if args.model_key == "cktn":
        print(f"[Model] Loaded discriminator checkpoint: {args.checkpoint}")

    class_weights = None
    if args.class_weighting == "balanced":
        class_weights = balanced_class_weights(train_dataset, len(label_to_id))
        print("[Diagnostics] class weights:", class_weights.tolist())

    model = CategoryClassifier(
        encoder, len(label_to_id), args.dropout, args.pooling, class_weights
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    classifier_diagnostics = classifier_optimizer_diagnostics(model, optimizer)
    save_json(args.output_dir / "classifier_diagnostics.json", classifier_diagnostics)
    if args.lr >= 5e-5:
        print(
            "[Warning] Full-encoder fine-tuning with lr >= 5e-5 can be unstable; "
            "use --lr 2e-5 if loss jumps or macro-F1 collapses."
        )
    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.epochs * update_steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0
    report = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            args.grad_accum_steps,
            args.max_grad_norm,
        )
        metrics = evaluate(model, dev_loader, device)
        row = {
            "epoch": epoch,
            "model_key": args.model_key,
            "model_id": model_id_for_key(args.model_key),
            "pooling": args.pooling,
            "train_loss": round(train_loss, 6),
            "dev_loss": round(metrics["loss"], 6),
            "accuracy": round(metrics["accuracy"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
            "micro_f1": round(metrics["micro_f1"], 6),
        }
        report.append(row)
        print(
            f"[Epoch {epoch}] Train loss={train_loss:.4f} | "
            f"Dev Acc={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f}"
        )

        is_best = metrics["macro_f1"] > best_macro_f1
        if is_best:
            best_macro_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "args": serializable_args(args),
                    "metrics": row,
                },
                args.output_dir / "best.pt",
            )
            print(f"[Checkpoint] Saved best model to {args.output_dir / 'best.pt'}")

        save_json(args.output_dir / "report.json", report)
        save_json(args.output_dir / "label_to_id.json", label_to_id)

    print(f"\n[Done] Best Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
