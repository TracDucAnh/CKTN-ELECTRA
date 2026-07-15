import argparse
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training import (
    DEV_FILES,
    DEVICE,
    DISC_DIR,
    GEN_DIR,
    MASK_RATE,
    NUM_WORKERS,
    SEQ_LEN,
    CKTNElectra,
    ELECTRADataset,
    load_or_build_chunks,
    load_tokenizer_and_check,
)


HERE = Path(__file__).parent.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CKTN-ELECTRA generator MLM loss/perplexity."
    )
    parser.add_argument(
        "--generator_checkpoint",
        type=Path,
        default=GEN_DIR / "generator_best.pt",
        help="Path to generator checkpoint saved by training.py.",
    )
    parser.add_argument(
        "--shared_embeddings_checkpoint",
        type=Path,
        default=DISC_DIR / "discriminator_best.pt",
        help="Discriminator checkpoint containing the paired shared embeddings.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=HERE / "checkpoint" / "generator" / "generator_ppl.json",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_generator_for_eval(args: argparse.Namespace, tokenizer) -> CKTNElectra:
    if not args.generator_checkpoint.exists():
        raise FileNotFoundError(f"Missing generator checkpoint: {args.generator_checkpoint}")
    if not args.shared_embeddings_checkpoint.exists():
        raise FileNotFoundError(
            "Missing shared embeddings checkpoint: "
            f"{args.shared_embeddings_checkpoint}"
        )

    print("[Model] Instantiating CKTNElectra for generator eval ...")
    model = CKTNElectra(load_pretrained=True, tokenizer=tokenizer)

    shared_state = torch.load(args.shared_embeddings_checkpoint, map_location="cpu")
    gen_state = torch.load(args.generator_checkpoint, map_location="cpu")

    model.shared_embeddings.load_state_dict(shared_state["shared_embeddings"])
    model.generator.load_state_dict(gen_state["generator"])
    model.to(DEVICE)
    model.eval()

    print(f"[Checkpoint] Generator: {args.generator_checkpoint}")
    print(f"[Checkpoint] Shared embeddings: {args.shared_embeddings_checkpoint}")
    return model


@torch.no_grad()
def evaluate_generator(model: CKTNElectra, dataloader: DataLoader) -> Dict[str, float]:
    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating generator", leave=False):
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            lam=0.0,
        )
        total_loss += outputs["loss_mlm"].item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {
        "generator_mlm_loss": avg_loss,
        "generator_perplexity": perplexity,
        "num_batches": total_batches,
    }


def save_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = load_tokenizer_and_check()
    dev_chunks = load_or_build_chunks(
        "dev", DEV_FILES, tokenizer, SEQ_LEN, args.rebuild_cache
    )
    dev_dataset = ELECTRADataset(dev_chunks, tokenizer, SEQ_LEN, MASK_RATE)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
    )

    model = load_generator_for_eval(args, tokenizer)
    metrics = evaluate_generator(model, dev_loader)
    payload = {
        "generator_checkpoint": str(args.generator_checkpoint),
        "shared_embeddings_checkpoint": str(args.shared_embeddings_checkpoint),
        "split": "dev",
        "mask_rate": MASK_RATE,
        "seq_len": SEQ_LEN,
        "batch_size": args.batch_size,
        "seed": args.seed,
        **metrics,
    }
    save_json(args.output_file, payload)

    print(f"[Result] Generator MLM loss: {metrics['generator_mlm_loss']:.6f}")
    print(f"[Result] Generator PPL: {metrics['generator_perplexity']:.6f}")
    print(f"[Result] Saved: {args.output_file}")


if __name__ == "__main__":
    main()
