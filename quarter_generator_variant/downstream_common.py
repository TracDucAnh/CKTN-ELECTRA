import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, RemBertModel


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
ARCH_PATH = HERE / "CKTN-ELECTRA.py"
CORPUS_DIR = ROOT / "corpus"
TRAIN_DIR = CORPUS_DIR / "train"
DEV_DIR = CORPUS_DIR / "dev"
DOWNSTREAM_DIR = HERE / "downstream"
DEFAULT_DISCRIMINATOR_CKPT = (
    HERE / "checkpoint" / "discriminator" / "discriminator_best.pt"
)


def load_arch_module():
    spec = importlib.util.spec_from_file_location("cktn_electra", ARCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARCH = load_arch_module()
BASE_CHECKPOINT = ARCH.DISCRIMINATOR_CHECKPOINT
BENCHMARK_MODELS = {
    "cktn": BASE_CHECKPOINT,
    "cktn_original": "ducanhdinh/CKTN-ELECTRA-original",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "rembert": "google/rembert",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_files(split: str) -> List[Path]:
    split_dir = TRAIN_DIR if split == "train" else DEV_DIR
    return [
        split_dir / f"cham_{split}.json",
        split_dir / f"khmer_{split}.json",
        split_dir / f"tay_nung_{split}.json",
    ]


def read_records(split: str) -> List[Dict]:
    records: List[Dict] = []
    for path in split_files(split):
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for index, item in enumerate(data):
            item = dict(item)
            item["_source"] = path.name
            item["_index"] = index
            item["_id"] = f"{path.stem}:{index}"
            records.append(item)
    return records


def field_to_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def join_fields(record: Dict, fields: List[str]) -> str:
    pieces = [field_to_text(record.get(field, "")) for field in fields]
    return "\n".join(piece for piece in pieces if piece)


def tokenizer_uses_slow(model_key: str) -> bool:
    return model_key in {"cktn", "cktn_original", "rembert"}


def model_id_for_key(model_key: str) -> str:
    return BENCHMARK_MODELS[model_key]


def load_tokenizer(model_key: str = "cktn") -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        model_id_for_key(model_key),
        use_fast=not tokenizer_uses_slow(model_key),
    )


def load_encoder(model_key: str, checkpoint_path: Path = DEFAULT_DISCRIMINATOR_CKPT):
    if model_key == "cktn":
        return rebuild_encoder(checkpoint_path)
    return AutoModel.from_pretrained(model_id_for_key(model_key))


def rebuild_encoder(checkpoint_path: Path) -> RemBertModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = AutoConfig.from_pretrained(BASE_CHECKPOINT)
    model = RemBertModel(config)
    state = torch.load(checkpoint_path, map_location="cpu")

    shared = state["shared_embeddings"]
    emb = model.embeddings
    emb.word_embeddings.weight.data.copy_(shared["word_embeddings.weight"])
    emb.position_embeddings.weight.data.copy_(shared["position_embeddings.weight"])
    emb.token_type_embeddings.weight.data.copy_(shared["token_type_embeddings.weight"])
    emb.LayerNorm.weight.data.copy_(shared["LayerNorm.weight"])
    emb.LayerNorm.bias.data.copy_(shared["LayerNorm.bias"])
    model.encoder.load_state_dict(state["encoder"])
    return model


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def serializable_args(args) -> Dict:
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload
