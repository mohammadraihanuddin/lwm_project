import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LayerNormalization(nn.Module):
    """Layer norm with learnable scale and bias."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class Embedding(nn.Module):
    """Linear projection + positional embedding with optional max_len override."""

    def __init__(self, element_length: int, d_model: int, max_len: int | None = None) -> None:
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.max_len = max_len if max_len is not None else 1025

        self.proj = nn.Linear(element_length, d_model)
        self.pos_embed = nn.Embedding(self.max_len, d_model)
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}.")

        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_encodings = self.pos_embed(pos)
        tok_emb = self.proj(x.float())
        return self.norm(tok_emb + pos_encodings)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention."""

    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_k)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = Q
        batch_size = Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = self.scaled_dot_attn(q_s, k_s, v_s)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(output)
        return residual + self.dropout(output), attn


class PoswiseFeedForwardNet(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, enc_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        attn_outputs = self.norm1(attn_outputs)
        ff_outputs = self.pos_ffn(attn_outputs)
        enc_outputs = self.norm2(attn_outputs + ff_outputs)
        return enc_outputs, attn


class LWM(nn.Module):
    """Large Wireless Model (Transformer encoder)."""

    def __init__(
        self,
        element_length: int = 32,
        d_model: int = 128,
        n_layers: int = 12,
        max_len: int | None = None,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.element_length = element_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_len = max_len if max_len is not None else 1025
        self.n_heads = n_heads
        self.dropout = dropout

        self.embedding = Embedding(element_length, d_model, self.max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_model * 4, dropout) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d_model, d_model)
        self.norm = LayerNormalization(d_model)

        embed_weight = self.embedding.proj.weight
        _, n_dim = embed_weight.size()
        self.decoder = nn.Linear(d_model, n_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_dim))

    def forward(
        self,
        input_ids: torch.Tensor,
        masked_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        output = self.embedding(input_ids)

        for layer in self.layers:
            output, attn = layer(output)

        if masked_pos is not None:
            masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(F.relu(self.linear(h_masked)))
            logits_lm = self.decoder(h_masked) + self.decoder_bias
            return logits_lm, output

        return output


def lwm(*args, **kwargs) -> LWM:
    """Factory to preserve backward compatibility with older imports."""

    return LWM(*args, **kwargs)


class PretrainedLWM(LWM):
    """Alias retained for compatibility with existing inference scripts."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
