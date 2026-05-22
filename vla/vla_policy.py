"""Portable PaliGemma feature extractor + LoRA adapter.

This module hosts the two pieces of the original Isaac-Sim VLA stack that the
new OpenFly training and RL code still depends on:

- ``LoRALinear`` — a tiny LoRA wrapper around a frozen ``nn.Linear`` (no PEFT
  dependency). Used by ``openfly/models/openfly_agent_rl.py`` to LoRA-adapt the
  OpenFly-Agent 7B policy for PPO.
- ``PaliGemmaFeatureExtractor`` — frozen ``google/paligemma-3b-pt-224`` with
  LoRA on ``q_proj`` / ``v_proj`` linears. Exposes ``preprocess_images``,
  ``forward_tokens`` / ``forward_tokens_with_grad`` (used by the OpenFly
  PaliGemma BC policy), plus the cached ``get_features`` / ``get_token_features``
  helpers and a single-token ``forward`` / ``forward_with_grad`` variant.

The legacy Isaac actor/critic classes that used to live in this file
(``VLAActorModel``, ``VLACriticModel``, ``HierarchicalVLAActor`` /
``HierarchicalVLACritic``) have been removed — the OpenFly stack does not
import them and the indoor curriculum they belonged to is no longer shipped.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA adapter (manual implementation, no PEFT dependency)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a frozen linear layer."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # A: Kaiming init for good gradient flow, B: zero init so LoRA starts as identity
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale


# ---------------------------------------------------------------------------
# PaliGemma feature extractor
# ---------------------------------------------------------------------------

class PaliGemmaFeatureExtractor(nn.Module):
    """Frozen PaliGemma 3B with LoRA adapters as a feature extractor.

    Takes raw RGB images and tokenized text, returns 2048-dim features
    from the last hidden state (mean-pooled over sequence length).
    """

    FEATURE_DIM = 2048  # PaliGemma's hidden size

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-pt-224",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        from transformers import PaliGemmaForConditionalGeneration

        print(f"[VLA] Loading PaliGemma from {model_name}...")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="eager",
        )

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # Apply LoRA to targeted layers
        self._apply_lora(lora_rank, lora_alpha, lora_targets)

        self._dtype = dtype
        self._img_size = 224

        n_lora = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[VLA] PaliGemma loaded. LoRA params: {n_lora:,} / {n_total:,} total ({100*n_lora/n_total:.2f}%)")

        # Feature cache (avoids double forward for actor + critic)
        self._cache_key: int | None = None
        self._cache_val: torch.Tensor | None = None

    def _apply_lora(self, rank: int, alpha: float, targets: tuple[str, ...]):
        """Replace targeted projection layers with LoRA-wrapped versions."""
        replaced = 0
        for name, module in list(self.model.named_modules()):
            if not any(t in name for t in targets):
                continue
            if not isinstance(module, nn.Linear):
                continue
            # Navigate to parent and replace
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(self.model.named_modules())[parts[0]]
                setattr(parent, parts[1], LoRALinear(module, rank, alpha))
                replaced += 1
        print(f"[VLA] Applied LoRA to {replaced} layers (rank={rank}, alpha={alpha})")

    def preprocess_images(self, rgb: torch.Tensor) -> torch.Tensor:
        """GPU-side image preprocessing.

        Args:
            rgb: (N, H, W, 3) float [0, 1]

        Returns:
            (N, 3, 224, 224) normalized for SigLIP
        """
        x = rgb.permute(0, 3, 1, 2)  # NHWC → NCHW
        if x.shape[-1] != self._img_size or x.shape[-2] != self._img_size:
            x = F.interpolate(x, size=(self._img_size, self._img_size), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5  # SigLIP uses [-1, 1] normalization
        return x.to(self._dtype)

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features from PaliGemma (always no_grad to avoid OOM).

        Returns:
            (batch_size, 2048) float32 tensor — mean-pooled last hidden state, detached.
        """
        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        # Last token hidden state — captures full image+text context via causal attention
        # (mean-pooling dilutes the text signal across 256 image tokens)
        hidden = outputs.last_hidden_state  # (B, seq_len, 2048)
        # Find last non-padding token per batch element
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        features = hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lengths]
        return features.float()

    def get_features(self, rgb: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Cached, mini-batched feature extraction (always detached)."""
        cache_key = rgb.data_ptr()
        if self._cache_key == cache_key and self._cache_val is not None:
            return self._cache_val

        pixel_values = self.preprocess_images(rgb)

        batch_size = pixel_values.shape[0]
        chunk_size = 64
        if batch_size <= chunk_size:
            features = self.forward(pixel_values, input_ids, attention_mask)
        else:
            chunks = []
            for i in range(0, batch_size, chunk_size):
                j = min(i + chunk_size, batch_size)
                chunk_feat = self.forward(
                    pixel_values[i:j], input_ids[i:j], attention_mask[i:j]
                )
                chunks.append(chunk_feat)
            features = torch.cat(chunks, dim=0)

        self._cache_key = cache_key
        self._cache_val = features.detach()
        return self._cache_val

    @torch.no_grad()
    def forward_tokens(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gemma last hidden state AND SigLIP vision features.

        Returns:
            gemma_features: (B, seq_len, 2048) — full sequence after Gemma (text-rich)
            siglip_features: (B, 256, 2048) — vision features before Gemma (spatially coherent)
        """
        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
        # Undo PaliGemma's sqrt(hidden_size) normalization so SigLIP features
        # are at the same scale as Gemma features (which image_proj expects)
        siglip_features = outputs.image_hidden_states * (2048 ** 0.5)  # (B, 256, 2048)
        return outputs.last_hidden_state.float(), siglip_features.float()

    def forward_tokens_with_grad(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like forward_tokens but with LoRA gradients enabled (no @torch.no_grad).

        Use only on small mini-batches for LoRA fine-tuning.
        """
        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
        siglip_features = outputs.image_hidden_states * (2048 ** 0.5)
        return outputs.last_hidden_state.float(), siglip_features.float()

    def get_token_features(self, rgb: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached, mini-batched token-level feature extraction.

        Returns:
            gemma_features: (B, seq_len, 2048) — Gemma last hidden state (text-rich)
            siglip_features: (B, 256, 2048) — SigLIP vision features (spatially coherent)
        """
        cache_key = ("tokens", rgb.data_ptr())
        if self._cache_key == cache_key and self._cache_val is not None:
            return self._cache_val

        pixel_values = self.preprocess_images(rgb)

        batch_size = pixel_values.shape[0]
        chunk_size = 32
        if batch_size <= chunk_size:
            features = self.forward_tokens(pixel_values, input_ids, attention_mask)
        else:
            gemma_chunks, siglip_chunks = [], []
            for i in range(0, batch_size, chunk_size):
                j = min(i + chunk_size, batch_size)
                gemma_c, siglip_c = self.forward_tokens(
                    pixel_values[i:j], input_ids[i:j], attention_mask[i:j]
                )
                gemma_chunks.append(gemma_c)
                siglip_chunks.append(siglip_c)
            features = (torch.cat(gemma_chunks, dim=0), torch.cat(siglip_chunks, dim=0))

        self._cache_key = cache_key
        self._cache_val = (features[0].detach(), features[1].detach())
        return self._cache_val

    def forward_with_grad(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features WITH gradient tracking for LoRA fine-tuning.

        Same as forward() but allows backpropagation through LoRA adapters.
        Only used during the LoRA update step on small mini-batches.
        """
        with torch.amp.autocast("cuda", dtype=self._dtype):
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        # Last token hidden state — same as forward() but with gradients
        hidden = outputs.last_hidden_state
        seq_lengths = attention_mask.sum(dim=1) - 1
        features = hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lengths]
        return features.float()  # NOT detached — gradients flow through LoRA

    def clear_cache(self):
        self._cache_key = None
        self._cache_val = None
