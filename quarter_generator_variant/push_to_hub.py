"""
push_to_hub.py — Push CKTN-ELECTRA discriminator weights to Hugging Face Hub.

Loads the best discriminator checkpoint from:
    main/checkpoint/discriminator/discriminator_best.pt

Rebuilds a full RemBertModel with the trained weights, then pushes to:
    ducanhdinh/CKTN-ELECTRA

HF token is read from .env (project root):
    HUGGINGFACE_HUB = "hf_..."

Usage:
    cd main
    python push_to_hub.py
    python push_to_hub.py --checkpoint checkpoint/discriminator/discriminator_epoch3.pt
"""

import argparse
import importlib.util
import re
from pathlib import Path

import torch
from huggingface_hub import login
from transformers import AutoConfig, AutoTokenizer, RemBertModel


# ─────────────────────────────────────────────────────────────────────────────
# 0. Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).parent.resolve()
_ROOT        = _HERE.parent                         # project root (.env lives here)
_ENV_PATH    = _ROOT / ".env"
_ARCH_PATH   = _HERE / "CKTN-ELECTRA.py"

HUB_REPO     = "ducanhdinh/CKTN-ELECTRA"
DEFAULT_CKPT = _HERE / "checkpoint" / "discriminator" / "discriminator_best.pt"


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push CKTN-ELECTRA discriminator to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"Path to discriminator .pt checkpoint (default: discriminator_best.pt)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load HF token from .env
# ─────────────────────────────────────────────────────────────────────────────

def load_hf_token() -> str:
    """
    Parse HUGGINGFACE_HUB from .env at the project root.
    Accepted formats:
        HUGGINGFACE_HUB = "hf_abc123"
        HUGGINGFACE_HUB=hf_abc123
    """
    if not _ENV_PATH.exists():
        raise FileNotFoundError(
            f".env not found at: {_ENV_PATH}\n"
            'Create it and add:  HUGGINGFACE_HUB = "hf_..."'
        )

    with open(_ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^HUGGINGFACE_HUB\s*=\s*["\']?([^"\']+)["\']?$', line)
            if m:
                token = m.group(1).strip()
                print(f"[Auth] Token loaded from .env  ({token[:8]}...)")
                return token

    raise KeyError(
        "HUGGINGFACE_HUB not found in .env.\n"
        'Expected:  HUGGINGFACE_HUB = "hf_..."'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Import DISCRIMINATOR_CHECKPOINT name from architecture file
# ─────────────────────────────────────────────────────────────────────────────

def get_disc_checkpoint_name() -> str:
    spec   = importlib.util.spec_from_file_location("cktn_electra", _ARCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DISCRIMINATOR_CHECKPOINT          # "ducanhdinh/CKTN-EKECTRA"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rebuild RemBertModel from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_discriminator(ckpt_path: Path, base_ckpt: str) -> RemBertModel:
    """
    Instantiate RemBertModel from base_ckpt config, then overwrite its weights
    with the saved discriminator state (shared_embeddings + embeddings_project
    + encoder).
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Rebuild] Loading config from '{base_ckpt}' ...")
    config = AutoConfig.from_pretrained(base_ckpt)
    model  = RemBertModel(config)

    print(f"[Rebuild] Loading weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    # ── Shared embeddings ─────────────────────────────────────────────────────
    se  = state["shared_embeddings"]
    emb = model.embeddings
    emb.word_embeddings.weight      .data.copy_(se["word_embeddings.weight"])
    emb.position_embeddings.weight  .data.copy_(se["position_embeddings.weight"])
    emb.token_type_embeddings.weight.data.copy_(se["token_type_embeddings.weight"])
    emb.LayerNorm.weight            .data.copy_(se["LayerNorm.weight"])
    emb.LayerNorm.bias              .data.copy_(se["LayerNorm.bias"])

    # ── Embeddings projection (embedding_size → hidden_size) ──────────────────
    ep = state["embeddings_project"]
    model.embeddings_project.weight.data.copy_(ep["weight"])
    model.embeddings_project.bias  .data.copy_(ep["bias"])

    # ── Transformer encoder ───────────────────────────────────────────────────
    model.encoder.load_state_dict(state["encoder"])

    epoch   = state.get("epoch",   "?")
    metrics = state.get("metrics", {})
    print(
        f"[Rebuild] Done — epoch={epoch} | "
        f"F1={metrics.get('f1', 'N/A')} | "
        f"Acc={metrics.get('accuracy', 'N/A')}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Auth ──────────────────────────────────────────────────────────────────
    token = load_hf_token()
    login(token=token)
    print(f"[Hub] Logged in. Target repo: {HUB_REPO}")

    # ── Get base checkpoint name from architecture ────────────────────────────
    base_ckpt = get_disc_checkpoint_name()
    print(f"[Hub] Base model config: {base_ckpt}")

    # ── Rebuild model ─────────────────────────────────────────────────────────
    model     = rebuild_discriminator(args.checkpoint, base_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt, use_fast=False)

    # ── Push to Hub ───────────────────────────────────────────────────────────
    print(f"\n[Hub] Pushing model weights to '{HUB_REPO}' ...")
    model.push_to_hub(
        HUB_REPO,
        token          = token,
        commit_message = f"Upload discriminator weights ({args.checkpoint.name})",
    )

    print(f"[Hub] Pushing tokenizer to '{HUB_REPO}' ...")
    tokenizer.push_to_hub(
        HUB_REPO,
        token          = token,
        commit_message = "Upload tokenizer",
    )

    print(f"\n[Done] Discriminator pushed to https://huggingface.co/{HUB_REPO}")


if __name__ == "__main__":
    main()