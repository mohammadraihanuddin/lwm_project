#!/usr/bin/env python3
"""
Minimal mobility_utils: prepare_model for LWM backbone (used by MoE EmbeddingExpert).

Updated version:
- supports standard prompt styles: deep, l2p, soft
- supports RFPrompt with configurable structured prompt groups:
    * global prompts
    * spectral prompts
    * temporal prompts
    * condition prompts
- supports optional internal RFPrompt router
- supports excluding prompt tokens from final pooling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent


def _split_rfprompt_groups(total_prompts: int) -> tuple[int, int, int, int]:
    """Split total RFPrompt count into (global, spectral, temporal, condition).

    Uses base ratio 1:2:1:1 and distributes remainders by largest fractions.
    Ensures each group is at least 1 when total >= 4.
    """
    total = max(4, int(total_prompts))
    weights = [1.0, 2.0, 1.0, 1.0]
    wsum = sum(weights)
    raw = [total * w / wsum for w in weights]
    counts = [int(v) for v in raw]
    frac_order = sorted(range(4), key=lambda i: raw[i] - counts[i], reverse=True)
    remain = total - sum(counts)
    for i in frac_order:
        if remain <= 0:
            break
        counts[i] += 1
        remain -= 1

    # Guarantee minimum 1 prompt/group when possible
    for i in range(4):
        if counts[i] == 0 and total >= 4:
            j = max(range(4), key=lambda k: counts[k])
            if counts[j] > 1:
                counts[j] -= 1
                counts[i] = 1

    return counts[0], counts[1], counts[2], counts[3]


def _patchify(spec: torch.Tensor, patch_rows: int = 4, patch_cols: int = 4) -> torch.Tensor:
    """(B, H, W) or (B, 1, H, W) -> (B, n_patches, patch_rows*patch_cols)."""
    if spec.dim() == 4:
        spec = spec.squeeze(1)
    B, H, W = spec.shape
    pr, pc = patch_rows, patch_cols
    n_r, n_c = H // pr, W // pc
    spec = spec[:, : n_r * pr, : n_c * pc]
    spec = spec.reshape(B, n_r, pr, n_c, pc).permute(0, 1, 3, 2, 4).reshape(B, n_r * n_c, pr * pc)
    return spec.float()


class L2PPromptPool(nn.Module):
    """L2P-style prompt pool: M prompts with keys, instance-wise top-N selection by query-key similarity."""

    def __init__(self, d_model: int = 128, pool_size: int = 10, prompt_len: int = 5, selection_size: int = 5) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.prompt_len = prompt_len
        self.selection_size = min(selection_size, pool_size)
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, d_model) * 0.02)
        self.keys = nn.Parameter(torch.randn(pool_size, d_model) * 0.02)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """query: (B, D). Returns similarity-weighted selected prompts (B, N*Lp, D)."""
        q = F.normalize(query, dim=-1)
        k = F.normalize(self.keys, dim=-1)
        sim = q @ k.T
        scores, idx = sim.topk(self.selection_size, dim=-1)
        weights = F.softmax(scores, dim=-1)
        selected = self.prompts[idx] * weights.unsqueeze(-1).unsqueeze(-1)
        return selected.reshape(query.size(0), -1, selected.size(-1))


class SoftPromptContextPool(nn.Module):
    """Soft prompt pool that produces a single context vector without increasing sequence length."""

    def __init__(self, d_model: int = 128, pool_size: int = 16, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.prompts = nn.Parameter(torch.randn(pool_size, d_model) * 0.02)
        self.keys = nn.Parameter(torch.randn(pool_size, d_model) * 0.02)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        q = F.normalize(query, dim=-1)
        k = F.normalize(self.keys, dim=-1)
        logits = (q @ k.T) / max(self.temperature, 1e-6)
        weights = F.softmax(logits, dim=-1)
        return weights @ self.prompts


class DeepPromptWrapper(nn.Module):
    """VPT-deep style: learnable prompts prepended at every transformer layer."""

    def __init__(self, backbone: nn.Module, prompt_len: int = 5, d_model: int = 128, init_scale: float = 0.02) -> None:
        super().__init__()
        self.backbone = backbone
        self.prompt_len = prompt_len
        n_layers = len(backbone.layers) if hasattr(backbone, "layers") else 12
        self.layer_prompts = nn.Parameter(torch.randn(n_layers, prompt_len, d_model) * init_scale)

    def forward_features(self, spec: torch.Tensor, *, return_tokens: bool = False, **kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = _patchify(spec)
        out = self.backbone.embedding(x)
        for i, layer in enumerate(self.backbone.layers):
            B = out.size(0)
            prompts = self.layer_prompts[i].unsqueeze(0).expand(B, -1, -1)
            combined = torch.cat([prompts, out], dim=1)
            combined, _ = layer(combined)
            out = combined[:, self.prompt_len :, :]
        pooled = out.mean(dim=1)
        if return_tokens:
            return pooled, out
        return pooled


class SoftPromptFiLMWrapper(nn.Module):
    """Prompt-conditioned feature modulation on the last few layers."""

    def __init__(
        self,
        backbone: nn.Module,
        pool_mod: SoftPromptContextPool,
        *,
        d_model: int = 128,
        conditioned_layers: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = pool_mod
        self.conditioned_layers = max(1, conditioned_layers)
        total_layers = len(backbone.layers) if hasattr(backbone, "layers") else 12
        self.start_layer = max(0, total_layers - self.conditioned_layers)
        self.modulators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, 2 * d_model),
                )
                for _ in range(total_layers - self.start_layer)
            ]
        )

    def forward_features(self, spec: torch.Tensor, *, return_tokens: bool = False, **kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = _patchify(spec)
        out = self.backbone.embedding(x)
        query = out.mean(dim=1)
        context = self.pool(query)

        mod_idx = 0
        for layer_idx, layer in enumerate(self.backbone.layers):
            layer_input = out
            if layer_idx >= self.start_layer:
                gamma_beta = self.modulators[mod_idx](context)
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                gamma = 0.1 * torch.tanh(gamma).unsqueeze(1)
                beta = 0.1 * torch.tanh(beta).unsqueeze(1)
                layer_input = out * (1.0 + gamma) + beta
                mod_idx += 1
            out, _ = layer(layer_input)

        pooled = out.mean(dim=1)
        if return_tokens:
            return pooled, out
        return pooled


class RFPromptWrapper(nn.Module):
    """
    RFPrompt: structured prompts for RF spectrogram transformers.

    Supports:
    - global prompts
    - spectral prompts
    - temporal prompts
    - condition-adaptive prompts
    - optional internal router
    - optional pooling over prompt tokens
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        d_model: int = 128,
        rfprompt_global: int = 4,
        rfprompt_spectral: int = 4,
        rfprompt_temporal: int = 2,
        rfprompt_condition: int = 2,
        rfprompt_use_router: bool = False,
        rfprompt_pool_prompts: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model

        self.rfprompt_config = {
            "global": int(rfprompt_global),
            "spectral": int(rfprompt_spectral),
            "temporal": int(rfprompt_temporal),
            "condition": int(rfprompt_condition),
            "use_router": bool(rfprompt_use_router),
            "pool_prompts": bool(rfprompt_pool_prompts),
        }

        self.num_global = max(0, int(rfprompt_global))
        self.num_spectral = max(0, int(rfprompt_spectral))
        self.num_temporal = max(0, int(rfprompt_temporal))
        self.num_condition = max(0, int(rfprompt_condition))
        self.use_router = bool(rfprompt_use_router)
        self.pool_prompts = bool(rfprompt_pool_prompts)

        if self.num_global > 0:
            self.Pg = nn.Parameter(torch.randn(self.num_global, d_model) * 0.02)
        else:
            self.register_parameter("Pg", None)

        if self.num_spectral > 0:
            self.Ps = nn.Parameter(torch.randn(self.num_spectral, d_model) * 0.02)
        else:
            self.register_parameter("Ps", None)

        if self.num_temporal > 0:
            self.Pt = nn.Parameter(torch.randn(self.num_temporal, d_model) * 0.02)
        else:
            self.register_parameter("Pt", None)

        if self.num_condition > 0:
            self.mlp_c = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.num_condition * d_model),
            )
        else:
            self.mlp_c = None

        if self.use_router:
            self.router = nn.Linear(d_model, 4)
        else:
            self.router = None

        self.total_prompts = self.num_global + self.num_spectral + self.num_temporal + self.num_condition

    def _scale_group(
        self,
        prompt_group: Optional[torch.Tensor],
        batch_size: int,
        alpha: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if prompt_group is None:
            return None
        out = prompt_group.unsqueeze(0).expand(batch_size, -1, -1)
        if alpha is not None:
            out = out * alpha.view(batch_size, 1, 1)
        return out

    def _build_condition_prompts(
        self,
        pooled_feature: torch.Tensor,
        alpha: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.mlp_c is None or self.num_condition <= 0:
            return None
        B, D = pooled_feature.shape
        Pc = self.mlp_c(pooled_feature).view(B, self.num_condition, D)
        if alpha is not None:
            Pc = Pc * alpha.view(B, 1, 1)
        return Pc

    def _build_prompts(self, xe: torch.Tensor) -> torch.Tensor:
        """
        xe: (B, N, D)
        returns: (B, M, D), where M = total structured prompts
        """
        B, _, D = xe.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        pooled = xe.mean(dim=1)

        if self.use_router and self.router is not None:
            alpha = torch.softmax(self.router(pooled), dim=-1)
            ag = alpha[:, 0] if self.num_global > 0 else None
            a_s = alpha[:, 1] if self.num_spectral > 0 else None
            a_t = alpha[:, 2] if self.num_temporal > 0 else None
            a_c = alpha[:, 3] if self.num_condition > 0 else None
        else:
            ag = a_s = a_t = a_c = None

        parts = []

        Pg = self._scale_group(self.Pg, B, ag)
        if Pg is not None:
            parts.append(Pg)

        Ps = self._scale_group(self.Ps, B, a_s)
        if Ps is not None:
            parts.append(Ps)

        Pt = self._scale_group(self.Pt, B, a_t)
        if Pt is not None:
            parts.append(Pt)

        Pc = self._build_condition_prompts(pooled, a_c)
        if Pc is not None:
            parts.append(Pc)

        if not parts:
            return torch.zeros(B, 0, D, device=xe.device, dtype=xe.dtype)

        return torch.cat(parts, dim=1)

    def forward_features(
        self,
        spec: torch.Tensor,
        *,
        return_tokens: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = _patchify(spec)
        xe = self.backbone.embedding(x)  # (B, N, D)

        prompts = self._build_prompts(xe)  # (B, M, D)
        xp = torch.cat([prompts, xe], dim=1) if self.total_prompts > 0 else xe

        out = xp
        for layer in self.backbone.layers:
            out, _ = layer(out)

        if self.total_prompts > 0:
            content = out[:, self.total_prompts :, :]
        else:
            content = out

        if self.pool_prompts:
            pooled = out.mean(dim=1)
        else:
            pooled = content.mean(dim=1)

        if return_tokens:
            return pooled, content
        return pooled


def prepare_model(
    checkpoint: Path | str,
    num_classes: int = 2,
    classifier_dim: int = 128,
    dropout: float = 0.0,
    trainable_layers: int = 0,
    projection_dim: int = 0,
    append_input_stats: bool = False,
    normalization_stats: Optional[Mapping[str, Any]] = None,
    head_type: str = "mlp",
    use_prompts: bool = False,
    num_prompts: int = 16,
    pool_size: int = 10,
    selection_size: int = 5,
    prompt_style: str = "deep",
    rfprompt_global: int = 4,
    rfprompt_spectral: int = 4,
    rfprompt_temporal: int = 2,
    rfprompt_condition: int = 2,
    rfprompt_use_router: bool = False,
    rfprompt_pool_prompts: bool = False,
) -> nn.Module:
    """
    Load LWM from checkpoint and return wrapper with .backbone and .forward_features(spec).
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from pretraining.pretrained_model import LWM

    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        state = {"state_dict": state}
    state_dict = state.get("state_dict", state)

    keys = list(state_dict.keys())
    if keys and (keys[0].startswith("module.") or keys[0].startswith("backbone.")):
        prefix = "module." if keys[0].startswith("module.") else "backbone."
        state_dict = {k.replace(prefix, "", 1): v for k, v in state_dict.items() if k.startswith(prefix)}
    if not state_dict:
        state_dict = state.get("state_dict", state)

    el = 16
    for k, v in state_dict.items():
        if "embedding.proj.weight" in k or (".proj.weight" in k and v.dim() == 2):
            el = int(v.shape[1])
            break

    backbone = LWM(
        element_length=el,
        d_model=128,
        n_layers=12,
        max_len=1025,
        n_heads=8,
        dropout=dropout,
    )
    backbone.load_state_dict(state_dict, strict=False)

    for p in backbone.parameters():
        p.requires_grad = False

    if trainable_layers > 0 and hasattr(backbone, "layers") and len(backbone.layers) >= trainable_layers:
        for layer in backbone.layers[-trainable_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    if use_prompts:
        style = (prompt_style or "").lower()

        if style == "rfprompt":
            # If caller did not override group sizes, derive them from total prompt count.
            if (rfprompt_global, rfprompt_spectral, rfprompt_temporal, rfprompt_condition) == (4, 4, 2, 2):
                rfprompt_global, rfprompt_spectral, rfprompt_temporal, rfprompt_condition = _split_rfprompt_groups(num_prompts)
            return RFPromptWrapper(
                backbone,
                d_model=128,
                rfprompt_global=rfprompt_global,
                rfprompt_spectral=rfprompt_spectral,
                rfprompt_temporal=rfprompt_temporal,
                rfprompt_condition=rfprompt_condition,
                rfprompt_use_router=rfprompt_use_router,
                rfprompt_pool_prompts=rfprompt_pool_prompts,
            )

        prompt_len = min(num_prompts, 20)
        prompt_len = max(5, prompt_len)

        if style == "l2p":
            sel = min(selection_size, pool_size)
            l2p_prompt_len = max(4, min(max(1, num_prompts // 2), 8))
            pool = L2PPromptPool(
                d_model=128,
                pool_size=pool_size,
                prompt_len=l2p_prompt_len,
                selection_size=sel,
            )

            class _L2PWrapper(nn.Module):
                def __init__(self, backbone: nn.Module, pool_mod: L2PPromptPool) -> None:
                    super().__init__()
                    self.backbone = backbone
                    self.pool = pool_mod
                    self._prompt_len = pool_mod.prompt_len * pool_mod.selection_size

                def forward_features(
                    self,
                    spec: torch.Tensor,
                    *,
                    return_tokens: bool = False,
                    **kwargs: Any,
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                    x = _patchify(spec)
                    xe = self.backbone.embedding(x)
                    query = xe.mean(dim=1)
                    prompts = self.pool(query)
                    xp = torch.cat([prompts, xe], dim=1)
                    out = xp
                    for layer in self.backbone.layers:
                        out, _ = layer(out)
                    content = out[:, self._prompt_len :, :]
                    pooled = out.mean(dim=1)
                    if return_tokens:
                        return pooled, content
                    return pooled

            return _L2PWrapper(backbone, pool)

        if style == "soft":
            effective_pool = max(pool_size, num_prompts)
            pool = SoftPromptContextPool(d_model=128, pool_size=effective_pool, temperature=1.0)
            conditioned_layers = 4 if num_prompts >= 16 else 2
            return SoftPromptFiLMWrapper(
                backbone,
                pool,
                d_model=128,
                conditioned_layers=conditioned_layers,
            )

        return DeepPromptWrapper(backbone, prompt_len=prompt_len, d_model=128, init_scale=0.02)

    class _Wrapper(nn.Module):
        def __init__(self, backbone: nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

        def forward_features(
            self,
            spec: torch.Tensor,
            *,
            return_tokens: bool = False,
            **kwargs: Any,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            x = _patchify(spec)
            out = self.backbone(x, masked_pos=None)
            if isinstance(out, tuple):
                out = out[1]
            pooled = out.mean(dim=1)
            if return_tokens:
                return pooled, out
            return pooled

    return _Wrapper(backbone)