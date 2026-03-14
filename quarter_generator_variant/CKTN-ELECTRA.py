"""
CKTN-ELECTRA: ELECTRA-style continued pre-training architecture
for Vietnamese ethnic minority languages (Cham, Khmer, Tay-Nung).

Discriminator  : ducanhdinh/CKTN-EKECTRA (vocab-augmented RemBERT)
                 32 layers | hidden=1152 | heads=18 | vocab=254,513
Generator      : ~1/4 discriminator size (auto-computed)
                 ~8 layers | same hidden & heads (for embedding compatibility)
Shared         : token embeddings (E_token) + position embeddings (E_pos)
Lambda schedule: λ=0 (ep 1-2) → linear to λ_max=50 (ep 2-3) → fixed (ep 4-5)
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    RemBertConfig,
    RemBertModel,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config
# ──────────────────────────────────────────────────────────────────────────────

DISCRIMINATOR_CHECKPOINT = "ducanhdinh/CKTN-EKECTRA"

# Training hyper-parameters (from paper)
TRAINING_CONFIG = dict(
    total_epochs   = 5,
    mask_rate      = 0.15,
    seq_len        = 512,
    lr             = 2e-5,
    warmup_ratio   = 0.06,
    weight_decay   = 0.01,
    grad_norm      = 1.0,
    lambda_max     = 50.0,
    # Lambda schedule (epoch boundaries, 1-indexed)
    lambda_zero_until_epoch   = 2,   # epochs 1-2 : λ = 0
    lambda_ramp_until_epoch   = 3,   # epoch  2-3 : linear ramp
    lambda_fixed_from_epoch   = 3,   # epochs 4-5 : λ = λ_max
)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Linear λ scheduler
# ──────────────────────────────────────────────────────────────────────────────

class LinearLambdaScheduler:
    """
    Implements the paper's λ schedule (equation 8):

        λ(t) = 0                                      if t ≤ t_warmup
               λ_max * (t - t_warmup) / (T - t_warmup)  if t > t_warmup

    Here we track fractional epoch position so the scheduler works with
    any number of steps per epoch.

    Args:
        lambda_max         : maximum discriminator loss weight (default 50)
        zero_until_epoch   : λ stays 0 through this epoch (inclusive)
        ramp_until_epoch   : λ reaches λ_max at the end of this epoch
        total_epochs       : total training epochs
        steps_per_epoch    : number of optimizer steps in one epoch
    """

    def __init__(
        self,
        lambda_max: float,
        zero_until_epoch: int,
        ramp_until_epoch: int,
        total_epochs: int,
        steps_per_epoch: int,
    ):
        self.lambda_max        = lambda_max
        self.steps_per_epoch   = steps_per_epoch
        self.total_steps       = total_epochs * steps_per_epoch
        self.warmup_steps      = zero_until_epoch * steps_per_epoch
        self.ramp_end_steps    = ramp_until_epoch * steps_per_epoch

    def get_lambda(self, global_step: int) -> float:
        """Return λ value for the current global training step."""
        if global_step <= self.warmup_steps:
            return 0.0
        if global_step >= self.ramp_end_steps:
            return self.lambda_max
        # linear ramp
        ramp_len = self.ramp_end_steps - self.warmup_steps
        progress = (global_step - self.warmup_steps) / ramp_len
        return self.lambda_max * progress


# ──────────────────────────────────────────────────────────────────────────────
# 3. Shared Embedding Module
# ──────────────────────────────────────────────────────────────────────────────

class SharedEmbeddings(nn.Module):
    """
    Token + position embeddings shared between generator and discriminator.

    RemBERT uses *separate* input and output embedding matrices (tie_word_embeddings=False).
    This module covers only the *input* side (E_token, E_pos, LayerNorm, dropout)
    as described in the paper.  The output projection in the generator's MLM head
    is NOT shared (following standard ELECTRA practice).
    """

    def __init__(self, config: RemBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.input_embedding_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.input_embedding_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.input_embedding_size
        )
        self.LayerNorm = nn.LayerNorm(config.input_embedding_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.LayerNorm(x))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Generator
# ──────────────────────────────────────────────────────────────────────────────

def _make_generator_config(disc_config: RemBertConfig) -> RemBertConfig:
    """
    Build a generator config that is ~1/4 the size of the discriminator.
    Rule  : num_hidden_layers //= 4  (minimum 1)
    Fixed : hidden_size, num_attention_heads, intermediate_size, vocab_size
            must match the discriminator so the shared embeddings are compatible.

    The input projection (embedding_size → hidden_size) is kept per-model
    because generator hidden_size == discriminator hidden_size in this setup.
    """
    gen_cfg = RemBertConfig.from_pretrained(DISCRIMINATOR_CHECKPOINT)

    gen_num_layers = max(1, disc_config.num_hidden_layers // 4)
    print(
        f"[Generator] disc layers={disc_config.num_hidden_layers} "
        f"→ gen layers={gen_num_layers} (ratio ≈ 1/{disc_config.num_hidden_layers // gen_num_layers})"
    )

    gen_cfg.num_hidden_layers = gen_num_layers
    # Keep hidden_size, num_attention_heads, vocab_size identical.
    # (RemBERT uses embedding_size != hidden_size; we preserve both.)
    return gen_cfg


class GeneratorMLMHead(nn.Module):
    """Standard MLM prediction head for the generator."""

    def __init__(self, config: RemBertConfig):
        super().__init__()
        self.dense      = nn.Linear(config.hidden_size, config.input_embedding_size)
        self.act        = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.input_embedding_size, eps=config.layer_norm_eps)
        # Output projection: NOT shared with discriminator (standard ELECTRA)
        self.decoder    = nn.Linear(config.input_embedding_size, config.vocab_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(self.act(self.dense(hidden_states)))
        return self.decoder(x)


class Generator(nn.Module):
    """
    Small transformer encoder (~1/4 discriminator layers) trained from scratch.
    Receives embeddings from SharedEmbeddings (not internal ones).
    """

    def __init__(self, config: RemBertConfig):
        super().__init__()
        self.config = config
        # Input projection: embedding_size → hidden_size (same as disc)
        self.embeddings_project = nn.Linear(config.input_embedding_size, config.hidden_size)
        # Encoder layers only (no embedding table — that comes from SharedEmbeddings)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = config.hidden_size,
            nhead           = config.num_attention_heads,
            dim_feedforward = config.intermediate_size,
            dropout         = config.hidden_dropout_prob,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )
        self.mlm_head = GeneratorMLMHead(config)

    def forward(
        self,
        embedded:       torch.Tensor,           # [B, L, embedding_size] from SharedEmbeddings
        attention_mask: Optional[torch.Tensor] = None,  # [B, L]  1=real, 0=pad
    ) -> torch.Tensor:
        """Returns logits [B, L, vocab_size]."""
        x = self.embeddings_project(embedded)   # [B, L, hidden_size]

        # nn.TransformerEncoder expects src_key_padding_mask: True = ignore
        pad_mask = None
        if attention_mask is not None:
            pad_mask = attention_mask == 0       # [B, L]

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.mlm_head(x)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Discriminator
# ──────────────────────────────────────────────────────────────────────────────

class DiscriminatorRTDHead(nn.Module):
    """Binary classification head: original (0) vs replaced (1)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act   = nn.GELU()
        self.out   = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, L, 1]."""
        return self.out(self.act(self.dense(hidden_states)))


class Discriminator(nn.Module):
    """
    Full RemBERT encoder (32 layers) initialized from CKTN-EKECTRA checkpoint.
    Embeddings are replaced with the shared ones from CKTNElectra.

    Architecture:
      SharedEmbeddings output [B, L, input_embedding_size=256]
        → RemBertEncoder.embedding_hidden_mapping_in [256 → 1152]  (internal to encoder)
        → RemBertEncoder transformer layers [32 layers]
        → RTDHead → logits [B, L, 1]

    NOTE: RemBertEncoder already contains its own Linear projection
    (embedding_hidden_mapping_in: input_embedding_size → hidden_size).
    We do NOT add an extra embeddings_project — we pass the 256-dim
    SharedEmbeddings output directly into the encoder.
    """

    def __init__(self, config: RemBertConfig):
        super().__init__()
        self.config = config
        # RemBertEncoder includes embedding_hidden_mapping_in(256→1152) internally
        from transformers.models.rembert.modeling_rembert import RemBertEncoder
        self.encoder  = RemBertEncoder(config)
        self.rtd_head = DiscriminatorRTDHead(config.hidden_size)

    def forward(
        self,
        embedded:       torch.Tensor,           # [B, L, input_embedding_size=256]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns per-token RTD logits [B, L, 1]."""
        # Build 4-D attention mask for HF encoder: additive, 0=attend, -inf=ignore
        if attention_mask is not None:
            ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        else:
            ext_mask = None

        # encoder applies embedding_hidden_mapping_in(256→1152) on first line
        enc_out = self.encoder(hidden_states=embedded, attention_mask=ext_mask)
        hidden  = enc_out[0]                    # [B, L, hidden_size=1152]
        return self.rtd_head(hidden)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main CKTN-ELECTRA Model
# ──────────────────────────────────────────────────────────────────────────────

class CKTNElectra(nn.Module):
    """
    CKTN-ELECTRA model combining:
      - SharedEmbeddings   (E_token + E_pos, updated by both losses)
      - Generator          (~1/4 disc size, trained from scratch)
      - Discriminator      (init from CKTN-EKECTRA, 32-layer RemBERT)

    Training flow (per step):
      1. SharedEmbeddings encodes the (possibly masked) input.
      2. Generator produces MLM logits → LMLM.
      3. Generator samples replacement tokens → corrupted sequence.
      4. SharedEmbeddings encodes corrupted sequence.
      5. Discriminator produces RTD logits → LDisc.
      6. Total loss: L = LMLM + λ(t) * LDisc
         (gradients flow through shared embeddings from both terms)
    """

    def __init__(self, load_pretrained: bool = True):
        super().__init__()

        # ── Load discriminator config & weights ──────────────────────────────
        disc_config: RemBertConfig = AutoConfig.from_pretrained(
            DISCRIMINATOR_CHECKPOINT
        )
        print(f"[Discriminator] vocab_size={disc_config.vocab_size}, "
              f"layers={disc_config.num_hidden_layers}, "
              f"hidden={disc_config.hidden_size}, "
              f"embedding_size={disc_config.input_embedding_size}")

        # ── Shared embeddings (init from discriminator checkpoint) ────────────
        self.shared_embeddings = SharedEmbeddings(disc_config)

        # ── Generator (from scratch, 1/4 size) ───────────────────────────────
        gen_config = _make_generator_config(disc_config)
        self.generator = Generator(gen_config)

        # ── Discriminator ─────────────────────────────────────────────────────
        self.discriminator = Discriminator(disc_config)

        if load_pretrained:
            self._load_pretrained_discriminator(disc_config)
        else:
            print("[Warning] load_pretrained=False — discriminator initialized randomly.")

    # ── Weight loading ────────────────────────────────────────────────────────

    def _load_pretrained_discriminator(self, disc_config: RemBertConfig):
        """
        Load CKTN-EKECTRA weights into:
          - shared_embeddings        (word, position, token_type, LayerNorm)
          - discriminator.embeddings_project
          - discriminator.encoder
        Generator is intentionally left randomly initialized.
        """
        print(f"[Loading] {DISCRIMINATOR_CHECKPOINT} ...")
        base_model = AutoModel.from_pretrained(
            DISCRIMINATOR_CHECKPOINT
        )

        # ── Copy shared embeddings from checkpoint ────────────────────────────
        src_emb = base_model.embeddings
        dst_emb = self.shared_embeddings

        dst_emb.word_embeddings.weight.data.copy_(
            src_emb.word_embeddings.weight.data
        )
        dst_emb.position_embeddings.weight.data.copy_(
            src_emb.position_embeddings.weight.data
        )
        dst_emb.token_type_embeddings.weight.data.copy_(
            src_emb.token_type_embeddings.weight.data
        )
        dst_emb.LayerNorm.weight.data.copy_(src_emb.LayerNorm.weight.data)
        dst_emb.LayerNorm.bias.data.copy_(src_emb.LayerNorm.bias.data)

        # ── Copy encoder (includes embedding_hidden_mapping_in + 32 layers) ────
        self.discriminator.encoder.load_state_dict(
            base_model.encoder.state_dict()
        )

        print("[Loading] Done — discriminator weights loaded; generator is random.")
        del base_model

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,                    # [B, L]
        attention_mask: Optional[torch.Tensor] = None,   # [B, L]
        token_type_ids: Optional[torch.Tensor] = None,   # [B, L]
        labels:         Optional[torch.Tensor] = None,   # [B, L]  -100 = not masked
        lam:            float = 0.0,
    ) -> dict:
        """
        Args:
            input_ids      : masked input token ids
            attention_mask : 1 for real tokens, 0 for padding
            token_type_ids : segment ids (usually all zeros)
            labels         : original token ids at masked positions, -100 elsewhere
            lam            : current λ value from LinearLambdaScheduler

        Returns dict with keys:
            loss           : scalar combined loss (LMLM + λ * LDisc)
            loss_mlm       : generator MLM loss
            loss_disc      : discriminator RTD loss (0 if lam == 0)
            gen_logits     : [B, L, V]
            disc_logits    : [B, L, 1]  (None if lam == 0)
        """
        # ── Step 1: Embed masked input ────────────────────────────────────────
        embedded_masked = self.shared_embeddings(
            input_ids, token_type_ids=token_type_ids
        )

        # ── Step 2: Generator MLM ─────────────────────────────────────────────
        gen_logits = self.generator(embedded_masked, attention_mask)  # [B, L, V]

        loss_mlm = torch.tensor(0.0, device=input_ids.device)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_mlm = loss_fct(
                gen_logits.view(-1, gen_logits.size(-1)), labels.view(-1)
            )

        # ── Step 3: Sample replacement tokens ────────────────────────────────
        # We do this even when lam==0 so the graph is consistent, but
        # disc_loss will be zero-weighted and won't affect gradients.
        with torch.no_grad():
            gen_probs        = torch.softmax(gen_logits, dim=-1)
            sampled_ids      = torch.multinomial(
                gen_probs.view(-1, gen_probs.size(-1)), num_samples=1
            ).view(input_ids.shape)                          # [B, L]

        # Build corrupted input: replace masked positions with generator samples
        if labels is not None:
            is_masked       = labels != -100                 # [B, L]
            corrupted_ids   = torch.where(is_masked, sampled_ids, input_ids)
            # RTD labels: 1 = replaced, 0 = original
            rtd_labels      = (corrupted_ids != input_ids).float()
            # But we only label positions that were originally masked
            rtd_label_mask  = is_masked
        else:
            corrupted_ids  = input_ids
            rtd_labels     = torch.zeros_like(input_ids, dtype=torch.float)
            rtd_label_mask = torch.ones_like(input_ids, dtype=torch.bool)

        loss_disc  = torch.tensor(0.0, device=input_ids.device)
        disc_logits = None

        # ── Step 4 & 5: Discriminator RTD ────────────────────────────────────
        if lam > 0.0:
            embedded_corrupted = self.shared_embeddings(
                corrupted_ids, token_type_ids=token_type_ids
            )
            disc_logits = self.discriminator(
                embedded_corrupted, attention_mask
            )                                                # [B, L, 1]

            # Per-token BCE over ALL tokens (not just masked) — eq. (6)
            bce = nn.BCEWithLogitsLoss(reduction="none")
            disc_loss_all = bce(
                disc_logits.squeeze(-1),                     # [B, L]
                rtd_labels,
            )
            if attention_mask is not None:
                disc_loss_all = disc_loss_all * attention_mask.float()
            loss_disc = disc_loss_all.sum() / (
                attention_mask.float().sum() if attention_mask is not None
                else disc_loss_all.numel()
            )

        # ── Step 6: Combined loss ─────────────────────────────────────────────
        loss = loss_mlm + lam * loss_disc

        return dict(
            loss        = loss,
            loss_mlm    = loss_mlm,
            loss_disc   = loss_disc,
            gen_logits  = gen_logits,
            disc_logits = disc_logits,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7. Parameter group utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_parameter_groups(model: CKTNElectra, weight_decay: float = 0.01):
    """
    Returns optimizer parameter groups.
    - Generator + shared embeddings are always updated.
    - Discriminator encoder is always updated (gradients flow when lam > 0).
    - No weight decay on bias / LayerNorm parameters.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    def _split(named_params):
        decay, no_wd = [], []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_wd.append(param)
            else:
                decay.append(param)
        return decay, no_wd

    disc_decay, disc_no_wd = _split(model.discriminator.named_parameters())
    gen_decay,  gen_no_wd  = _split(model.generator.named_parameters())
    emb_decay,  emb_no_wd  = _split(model.shared_embeddings.named_parameters())

    return [
        {"params": disc_decay + gen_decay + emb_decay, "weight_decay": weight_decay},
        {"params": disc_no_wd + gen_no_wd + emb_no_wd, "weight_decay": 0.0},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 8. Training loop skeleton
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model:           CKTNElectra,
    dataloader,
    optimizer,
    lr_scheduler,
    lambda_scheduler: LinearLambdaScheduler,
    cfg:             dict,
    device:          str = "cuda",
):
    """
    Minimal training loop illustrating λ scheduling and gradient flow.

    Replace `dataloader` with a real DataLoader that yields batches of:
        {input_ids, attention_mask, token_type_ids, labels}
    """
    model.to(device)
    model.train()

    global_step = 0

    for epoch in range(1, cfg["total_epochs"] + 1):
        for batch in dataloader:
            batch       = {k: v.to(device) for k, v in batch.items()}
            lam         = lambda_scheduler.get_lambda(global_step)

            outputs     = model(**batch, lam=lam)
            loss        = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_norm"])
            optimizer.step()
            lr_scheduler.step()

            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"λ={lam:.1f} "
                    f"loss={loss.item():.4f} "
                    f"lm={outputs['loss_mlm'].item():.4f} "
                    f"disc={outputs['loss_disc'].item():.4f}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# 9. Quick smoke-test (no GPU, no real data needed)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("=" * 70)
    print("CKTN-ELECTRA — architecture smoke-test (load_pretrained = True)")
    print("=" * 70)

    # Instantiate without loading 0.6 B param checkpoint for quick test
    model = CKTNElectra(load_pretrained=True)

    disc_layers = model.discriminator.config.num_hidden_layers
    gen_layers  = model.generator.config.num_hidden_layers
    print(f"\nDisc layers : {disc_layers}")
    print(f"Gen  layers : {gen_layers}  (ratio ≈ 1/{disc_layers // gen_layers})")

    # Lambda scheduler
    steps_per_epoch = 1000          # placeholder
    sched = LinearLambdaScheduler(
        lambda_max        = TRAINING_CONFIG["lambda_max"],
        zero_until_epoch  = TRAINING_CONFIG["lambda_zero_until_epoch"],
        ramp_until_epoch  = TRAINING_CONFIG["lambda_ramp_until_epoch"],
        total_epochs      = TRAINING_CONFIG["total_epochs"],
        steps_per_epoch   = steps_per_epoch,
    )
    checkpoints = [0, 1000, 2000, 2500, 3000, 4000, 5000]
    print("\nλ schedule preview:")
    for s in checkpoints:
        print(f"  step {s:5d} → λ = {sched.get_lambda(s):.1f}")

    # Dummy forward pass
    B, L, V = 2, 16, model.shared_embeddings.word_embeddings.num_embeddings
    input_ids      = torch.randint(0, V, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    labels         = input_ids.clone()
    labels[:, ::3] = -100           # mask every 3rd token

    print("\nRunning dummy forward pass (λ=0) ...")
    out0 = model(input_ids, attention_mask, labels=labels, lam=0.0)
    print(f"  loss={out0['loss'].item():.4f}  loss_mlm={out0['loss_mlm'].item():.4f}")

    print("\nRunning dummy forward pass (λ=25) ...")
    out1 = model(input_ids, attention_mask, labels=labels, lam=25.0)
    print(f"  loss={out1['loss'].item():.4f}  loss_mlm={out1['loss_mlm'].item():.4f}  "
          f"loss_disc={out1['loss_disc'].item():.4f}")

    print("\nSmoke-test passed ✓")