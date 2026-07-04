import argparse
import inspect
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
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


def select_query(
    record: Dict,
    query_fields: List[str],
    fallback_query_fields: List[str],
) -> tuple[str, str]:
    query = join_fields(record, query_fields)
    if query:
        return query, "primary"

    fallback_query = join_fields(record, fallback_query_fields)
    if fallback_query:
        return fallback_query, "fallback"

    return "", "missing"


def record_group(record: Dict, group_key: str) -> str:
    if group_key == "none":
        return "all"

    group = str(record.get(group_key, "")).strip()
    return group or "all"


class RetrievalPairDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        query_fields: List[str],
        fallback_query_fields: List[str],
        doc_fields: List[str],
        group_key: str,
    ):
        self.examples = []
        self.query_sources = Counter()
        self.groups = Counter()
        self.stats = Counter()
        for record in records:
            query, query_source = select_query(
                record, query_fields, fallback_query_fields
            )
            document = join_fields(record, doc_fields)
            self.stats["records"] += 1
            self.stats["empty_document"] += int(not document)
            self.query_sources[query_source] += 1
            if query and document:
                group = record_group(record, group_key)
                self.groups[group] += 1
                self.stats["pairs"] += 1
                self.examples.append(
                    {
                        "id": record["_id"],
                        "query": query,
                        "document": document,
                        "group": group,
                        "query_source": query_source,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        return self.examples[index]


class GroupedBatchSampler(Sampler[List[int]]):
    def __init__(self, examples: List[Dict], batch_size: int, drop_last: bool):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.group_to_indices = defaultdict(list)
        for index, example in enumerate(examples):
            self.group_to_indices[example["group"]].append(index)

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        for indices in self.group_to_indices.values():
            shuffled = indices[:]
            random.shuffle(shuffled)
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.group_to_indices.values():
            full_batches = len(indices) // self.batch_size
            partial_batch = int(
                not self.drop_last and len(indices) % self.batch_size > 0
            )
            total += full_batches + partial_batch
        return total

    def batch_counts(self) -> Dict[str, int]:
        return {
            group: len(indices) // self.batch_size
            for group, indices in self.group_to_indices.items()
        }


class RetrievalCollator:
    def __init__(self, tokenizer, max_query_length: int, max_doc_length: int):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def encode(self, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __call__(self, batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "query": self.encode(
                [item["query"] for item in batch], self.max_query_length
            ),
            "document": self.encode(
                [item["document"] for item in batch], self.max_doc_length
            ),
        }


class BiEncoderRetriever(nn.Module):
    def __init__(self, encoder: nn.Module, temperature: float):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.accepts_token_type_ids = (
            "token_type_ids" in inspect.signature(encoder.forward).parameters
        )

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.accepts_token_type_ids:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        pooled = masked_mean_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]):
        query_emb = self.encode(**query)
        doc_emb = self.encode(**document)
        logits = query_emb @ doc_emb.t() / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        query_loss = nn.CrossEntropyLoss()(logits, labels)
        doc_loss = nn.CrossEntropyLoss()(logits.t(), labels)
        loss = 0.5 * (query_loss + doc_loss)
        return {
            "loss": loss,
            "logits": logits,
            "query_emb": query_emb,
            "doc_emb": doc_emb,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune CKTN-ELECTRA for summary-to-document retrieval"
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
    parser.add_argument("--query_fields", nargs="+", default=["summary"])
    parser.add_argument("--fallback_query_fields", nargs="*", default=["tags"])
    parser.add_argument("--doc_fields", nargs="+", default=["title", "content"])
    parser.add_argument("--max_query_length", type=int, default=512)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch_group_key", type=str, default="_source")
    parser.add_argument("--diagnostic_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def move_encoded(encoded: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in encoded.items()}


def move_batch(batch: Dict[str, Dict[str, torch.Tensor]], device: str) -> Dict:
    return {
        "query": move_encoded(batch["query"], device),
        "document": move_encoded(batch["document"], device),
    }


def counter_payload(counter: Counter, top_k: int = 20) -> Dict[str, int]:
    return {key: count for key, count in counter.most_common(top_k)}


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not any(term in name for term in no_decay_terms)
    ]
    no_decay_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and any(term in name for term in no_decay_terms)
    ]
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def batch_similarity_diagnostics(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    scores = outputs["query_emb"].detach() @ outputs["doc_emb"].detach().t()
    diagonal = scores.diag()
    offdiag_mask = ~torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
    offdiag = scores[offdiag_mask]
    positive_rank = (scores > diagonal.unsqueeze(1)).sum(dim=1).float() + 1.0
    return {
        "loss": float(outputs["loss"].detach().item()),
        "diag_mean": float(diagonal.mean().item()),
        "offdiag_mean": float(offdiag.mean().item()),
        "offdiag_std": float(offdiag.std(unbiased=False).item()),
        "positive_rank_mean": float(positive_rank.mean().item()),
        "query_emb_std": float(outputs["query_emb"].detach().std().item()),
        "doc_emb_std": float(outputs["doc_emb"].detach().std().item()),
    }


def build_eval_sets(
    records: List[Dict],
    query_fields: List[str],
    fallback_query_fields: List[str],
    doc_fields: List[str],
) -> Dict:
    doc_records = []
    doc_index = {}
    for record in records:
        document = join_fields(record, doc_fields)
        if document:
            doc_index[record["_id"]] = len(doc_records)
            doc_records.append({"id": record["_id"], "text": document})

    query_records = []
    query_sources = Counter()
    for record in records:
        query, query_source = select_query(
            record, query_fields, fallback_query_fields
        )
        document = join_fields(record, doc_fields)
        query_sources[query_source] += 1
        if query and document and record["_id"] in doc_index:
            query_records.append(
                {
                    "id": record["_id"],
                    "text": query,
                    "positive_index": doc_index[record["_id"]],
                    "query_source": query_source,
                }
            )

    return {
        "queries": query_records,
        "documents": doc_records,
        "query_sources": dict(query_sources),
    }


@torch.no_grad()
def encode_texts(
    model: BiEncoderRetriever,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    model.eval()
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="  Encoding", leave=False):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = move_encoded(encoded, device)
        embeddings.append(model.encode(**encoded).cpu())
    model.train()
    return torch.cat(embeddings, dim=0)


def retrieval_metrics(scores: torch.Tensor, positives: torch.Tensor) -> Dict[str, float]:
    positive_scores = scores.gather(1, positives.unsqueeze(1))
    ranks = (scores > positive_scores).sum(dim=1) + 1
    ranks = ranks.float()

    mrr10 = torch.where(ranks <= 10, 1.0 / ranks, torch.zeros_like(ranks)).mean()
    recall1 = (ranks <= 1).float().mean()
    recall5 = (ranks <= 5).float().mean()
    recall10 = (ranks <= 10).float().mean()
    ndcg10 = torch.where(
        ranks <= 10,
        1.0 / torch.log2(ranks + 1.0),
        torch.zeros_like(ranks),
    ).mean()

    return {
        "mrr_at_10": float(mrr10.item()),
        "recall_at_1": float(recall1.item()),
        "recall_at_5": float(recall5.item()),
        "recall_at_10": float(recall10.item()),
        "ndcg_at_10": float(ndcg10.item()),
    }


@torch.no_grad()
def evaluate(
    model: BiEncoderRetriever,
    tokenizer,
    eval_sets: Dict,
    args,
    device: str,
) -> Dict[str, float]:
    queries = eval_sets["queries"]
    documents = eval_sets["documents"]
    query_emb = encode_texts(
        model,
        tokenizer,
        [item["text"] for item in queries],
        args.max_query_length,
        args.eval_batch_size,
        device,
    )
    doc_emb = encode_texts(
        model,
        tokenizer,
        [item["text"] for item in documents],
        args.max_doc_length,
        args.eval_batch_size,
        device,
    )
    scores = query_emb @ doc_emb.t()
    positives = torch.tensor(
        [item["positive_index"] for item in queries], dtype=torch.long
    )
    metrics = retrieval_metrics(scores, positives)
    metrics["num_queries"] = len(queries)
    metrics["num_documents"] = len(documents)
    return metrics


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: str,
    grad_accum_steps: int,
    max_grad_norm: float,
    diagnostic_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="  Training", leave=True), start=1):
        batch = move_batch(batch, device)
        outputs = model(**batch)
        loss = outputs["loss"] / grad_accum_steps
        loss.backward()
        total_loss += outputs["loss"].item()

        if step <= diagnostic_steps:
            diagnostics = batch_similarity_diagnostics(outputs)
            print(
                "[Diagnostics] "
                f"step={step} loss={diagnostics['loss']:.4f} "
                f"diag={diagnostics['diag_mean']:.4f} "
                f"offdiag={diagnostics['offdiag_mean']:.4f} "
                f"offdiag_std={diagnostics['offdiag_std']:.4f} "
                f"rank={diagnostics['positive_rank_mean']:.2f} "
                f"q_std={diagnostics['query_emb_std']:.4f} "
                f"d_std={diagnostics['doc_emb_std']:.4f}"
            )

        should_step = step % grad_accum_steps == 0 or step == len(dataloader)
        if should_step:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(len(dataloader), 1)


def main():
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("Retrieval training needs --batch_size >= 2.")
    set_seed(args.seed)
    if args.output_dir is None:
        args.output_dir = DOWNSTREAM_DIR / "information_retrieval" / args.model_key
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_key)
    train_records = read_records("train")
    dev_records = read_records("dev")

    train_dataset = RetrievalPairDataset(
        train_records,
        args.query_fields,
        args.fallback_query_fields,
        args.doc_fields,
        args.batch_group_key,
    )
    eval_sets = build_eval_sets(
        dev_records,
        args.query_fields,
        args.fallback_query_fields,
        args.doc_fields,
    )

    collator = RetrievalCollator(
        tokenizer, args.max_query_length, args.max_doc_length
    )
    batch_sampler = GroupedBatchSampler(
        train_dataset.examples,
        args.batch_size,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
    )

    print(f"[Data] Train pairs: {len(train_dataset)}")
    print(f"[Data] Dev queries: {len(eval_sets['queries'])}")
    print(f"[Data] Dev candidate docs: {len(eval_sets['documents'])}")
    print(f"[Data] Train query sources: {dict(train_dataset.query_sources)}")
    print(f"[Data] Dev query sources: {eval_sets['query_sources']}")
    print(f"[Data] Train groups: {dict(train_dataset.groups)}")
    print(f"[Data] Train grouped batches: {batch_sampler.batch_counts()}")
    save_json(
        args.output_dir / "data_diagnostics.json",
        {
            "model_key": args.model_key,
            "model_id": model_id_for_key(args.model_key),
            "train_stats": dict(train_dataset.stats),
            "train_query_sources": dict(train_dataset.query_sources),
            "dev_query_sources": eval_sets["query_sources"],
            "train_groups": dict(train_dataset.groups),
            "train_grouped_batches": batch_sampler.batch_counts(),
            "query_fields": args.query_fields,
            "fallback_query_fields": args.fallback_query_fields,
            "doc_fields": args.doc_fields,
            "batch_group_key": args.batch_group_key,
        },
    )

    encoder = load_encoder(args.model_key, args.checkpoint)
    print(f"[Model] Loaded encoder: {args.model_key} ({model_id_for_key(args.model_key)})")
    if args.model_key == "cktn":
        print(f"[Model] Loaded discriminator checkpoint: {args.checkpoint}")
    model = BiEncoderRetriever(encoder, args.temperature).to(device)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.epochs * update_steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_mrr = -1.0
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
            args.diagnostic_steps,
        )
        metrics = evaluate(model, tokenizer, eval_sets, args, device)
        row = {
            "epoch": epoch,
            "model_key": args.model_key,
            "model_id": model_id_for_key(args.model_key),
            "train_loss": round(train_loss, 6),
            "mrr_at_10": round(metrics["mrr_at_10"], 6),
            "recall_at_1": round(metrics["recall_at_1"], 6),
            "recall_at_5": round(metrics["recall_at_5"], 6),
            "recall_at_10": round(metrics["recall_at_10"], 6),
            "ndcg_at_10": round(metrics["ndcg_at_10"], 6),
            "num_queries": metrics["num_queries"],
            "num_documents": metrics["num_documents"],
        }
        report.append(row)
        print(
            f"[Epoch {epoch}] Train loss={train_loss:.4f} | "
            f"MRR@10={metrics['mrr_at_10']:.4f} | "
            f"R@1={metrics['recall_at_1']:.4f} | "
            f"R@10={metrics['recall_at_10']:.4f}"
        )

        is_best = metrics["mrr_at_10"] > best_mrr
        if is_best:
            best_mrr = metrics["mrr_at_10"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": serializable_args(args),
                    "metrics": row,
                },
                args.output_dir / "best.pt",
            )
            print(f"[Checkpoint] Saved best model to {args.output_dir / 'best.pt'}")

        save_json(args.output_dir / "report.json", report)

    print(f"\n[Done] Best MRR@10: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
