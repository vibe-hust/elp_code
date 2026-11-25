"""Self-contained PyTorch implementation & safetensors loader for Dory1.0-4B (nGPT looped LM)."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
import torch.nn.functional as F

try:
    from transformers import PretrainedConfig, PreTrainedModel
except ImportError:
    class PretrainedConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class PreTrainedModel(nn.Module):  # type: ignore[no-redef]
        def __init__(self, config: Any) -> None:
            super().__init__()
            self.config = config

        def post_init(self) -> None:
            return None

try:
    from transformers import GenerationMixin
except ImportError:
    try:
        from transformers.generation import GenerationMixin  # type: ignore[no-redef]
    except ImportError:
        class GenerationMixin:  # type: ignore[no-redef]
            pass

try:
    from transformers.cache_utils import Cache
except ImportError:
    class Cache:  # type: ignore[no-redef]
        pass

try:
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
except ImportError:
    @dataclass
    class BaseModelOutputWithPast:  # type: ignore[no-redef]
        last_hidden_state: torch.Tensor
        past_key_values: Any = None

    @dataclass
    class CausalLMOutputWithPast:  # type: ignore[no-redef]
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        past_key_values: Any = None


class DoryConfig(PretrainedConfig):
    """Configuration for Dory1.0-4B alternating/paired recurrent hyperspherical LM."""

    model_type = "dory"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 2560,
        intermediate_size: int = 7168,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.0177,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        sliding_window: int = 1024,
        layer_types: list[str] | None = None,
        ngpt_alpha_init: float = 0.05,
        n_input_layers: int = 8,
        n_recurrent_layers: int | None = None,
        n_output_layers: int = 8,
        n_recurrent_loops: int = 2,
        recurrent_kv_cache_mode: str = "per_loop",
        **kwargs: Any,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.use_cache = bool(use_cache)
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.mlp_bias = bool(mlp_bias)
        self.sliding_window = int(sliding_window)
        self.ngpt_alpha_init = float(ngpt_alpha_init)
        self.n_recurrent_loops = int(n_recurrent_loops)
        self.recurrent_kv_cache_mode = str(recurrent_kv_cache_mode)

        if layer_types is not None and "mlp" in layer_types:
            layer_types = list(layer_types)[::2]
            num_hidden_layers = int(num_hidden_layers) // 2
            n_input_layers = int(n_input_layers) // 2
            n_output_layers = int(n_output_layers) // 2
            if n_recurrent_layers is not None:
                n_recurrent_layers = int(n_recurrent_layers) // 2

        self.num_hidden_layers = int(num_hidden_layers)
        self.n_input_layers = int(n_input_layers)
        self.n_output_layers = int(n_output_layers)
        if n_recurrent_layers is None:
            n_recurrent_layers = self.num_hidden_layers - self.n_input_layers - self.n_output_layers
        self.n_recurrent_layers = int(n_recurrent_layers)

        if layer_types is None:
            layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            layer_types = list(layer_types)

        recurrent_end = self.n_input_layers + self.n_recurrent_layers
        self.layer_types = [
            f"recurrent_{lt[len('recurrent_'):]}"
            if lt.startswith("recurrent_")
            else (f"recurrent_{lt}" if self.n_input_layers <= i < recurrent_end else lt)
            for i, lt in enumerate(layer_types)
        ]

        if rope_parameters is None:
            rope_thetas = {"sliding_attention": 10_000.0, "full_attention": 10_000_000.0}
            base_types = {lt.removeprefix("recurrent_") for lt in self.layer_types}
            rope_parameters = {
                lt: {"rope_type": "default", "rope_theta": rope_thetas[lt], "partial_rotary_factor": 1.0}
                for lt in sorted(base_types)
            }
        else:
            rope_parameters = {k: dict(v) for k, v in rope_parameters.items()}
        for lt in sorted(set(self.layer_types)):
            base_lt = lt.removeprefix("recurrent_")
            if lt not in rope_parameters and base_lt in rope_parameters:
                rope_parameters[lt] = dict(rope_parameters[base_lt])
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.validate_architecture()

    def validate_architecture(self) -> None:
        if self.n_input_layers + self.n_recurrent_layers + self.n_output_layers != self.num_hidden_layers:
            raise ValueError("n_input_layers + n_recurrent_layers + n_output_layers must equal num_hidden_layers")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"len(layer_types)={len(self.layer_types)} != num_hidden_layers={self.num_hidden_layers}")
        if self.n_recurrent_loops < 1:
            raise ValueError(f"n_recurrent_loops must be >= 1, got {self.n_recurrent_loops}")
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError(f"Invalid recurrent_kv_cache_mode: {self.recurrent_kv_cache_mode!r}")


class DoryL2Norm(nn.Module):
    """Projects hidden states onto the unit hypersphere."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x)


class DoryRecurrentCache(Cache):
    """Cache supporting Dory `per_loop` and `last_loop` KV modes."""

    is_compileable = False

    def __init__(self, num_layers: int, sliding_window: int, recurrent_kv_cache_mode: str = "per_loop") -> None:
        self.num_layers = int(num_layers)
        self.sliding_window = int(sliding_window)
        self.recurrent_kv_cache_mode = str(recurrent_kv_cache_mode)
        self._store: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._seen_tokens: int = 0
        self.layers: list[Any] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self._seen_tokens

    def get_max_cache_shape(self, layer_idx: int = 0) -> int | None:
        del layer_idx
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        del layer_idx
        return self._seen_tokens + cache_position.shape[0], 0

    def __len__(self) -> int:
        return self.num_layers

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for k, (ks, vs) in list(self._store.items()):
            idx = beam_idx.to(ks.device)
            self._store[k] = (ks.index_select(0, idx), vs.index_select(0, idx))

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = max(0, self._seen_tokens + max_length)
        if self._seen_tokens <= max_length:
            return
        self._seen_tokens = max_length
        for k, (ks, vs) in list(self._store.items()):
            self._store[k] = (ks[..., :max_length, :], vs[..., :max_length, :])

    def snapshot(self) -> tuple[dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]], int]:
        copied = {k: (v[0].clone(), v[1].clone()) for k, v in self._store.items()}
        return copied, self._seen_tokens

    def restore(self, snap: tuple[dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]], int]) -> None:
        store, seen = snap
        self._store = {k: (v[0].clone(), v[1].clone()) for k, v in store.items()}
        self._seen_tokens = int(seen)

    def reset(self) -> None:
        self._store.clear()
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *,
        loop_idx: int = 0,
        is_last_loop: bool = True,
        is_recurrent: bool = False,
        is_sliding: bool = False,
        advance_seen: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if is_recurrent and self.recurrent_kv_cache_mode == "last_loop":
            cache_key = (layer_idx, 0)
            read_only = not is_last_loop
        else:
            cache_key = (layer_idx, loop_idx if is_recurrent else 0)
            read_only = False

        prev = self._store.get(cache_key)
        if prev is not None:
            full_k = torch.cat([prev[0], key_states], dim=-2)
            full_v = torch.cat([prev[1], value_states], dim=-2)
        else:
            full_k, full_v = key_states, value_states

        if is_sliding and full_k.shape[-2] > self.sliding_window:
            keep = max(self.sliding_window, self.sliding_window - 1 + key_states.shape[-2])
            if not read_only:
                self._store[cache_key] = (
                    full_k[..., -self.sliding_window :, :],
                    full_v[..., -self.sliding_window :, :],
                )
            full_k = full_k[..., -keep:, :]
            full_v = full_v[..., -keep:, :]
        elif not read_only:
            self._store[cache_key] = (full_k, full_v)

        if advance_seen:
            self._seen_tokens += key_states.shape[-2]
        return full_k, full_v


class DoryRotaryEmbedding(nn.Module):
    def __init__(self, config: DoryConfig, device: Any = None) -> None:
        super().__init__()
        self.config = config
        self.layer_types = sorted(set(config.layer_types))
        dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        for layer_type in self.layer_types:
            base = float(config.rope_parameters[layer_type]["rope_theta"])
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq").to(device=x.device, dtype=torch.float32)
        freqs = position_ids[..., None].float() * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    return (q * cos_u) + (_rotate_half(q) * sin_u), (k * cos_u) + (_rotate_half(k) * sin_u)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    )


def _build_sdpa_mask(
    q_len: int,
    kv_len: int,
    position_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor | None,
    *,
    sliding_window: int | None,
    device: torch.device,
) -> torch.Tensor:
    bsz = position_ids.shape[0]
    if kv_len == q_len:
        k_pos = position_ids[:, None, :]
    else:
        past_len = kv_len - q_len
        if past_len == int(position_ids[0, 0].item()):
            k_pos = torch.arange(kv_len, device=device, dtype=position_ids.dtype)[None, None, :].expand(bsz, 1, kv_len)
        else:
            start = position_ids[:, :1] - past_len
            k_pos = start[:, None, :] + torch.arange(kv_len, device=device, dtype=position_ids.dtype)[None, None, :]
    q_pos = position_ids[:, :, None]
    allowed = k_pos <= q_pos
    if sliding_window is not None:
        allowed = allowed & ((q_pos - k_pos) < sliding_window)
    if attention_mask_2d is not None:
        key_valid = attention_mask_2d[:, -kv_len:].bool()[:, None, :]
        allowed = allowed & key_valid
        has_any = allowed.any(dim=-1, keepdim=True)
        allowed = torch.where(has_any, allowed, k_pos == k_pos.min(dim=-1, keepdim=True).values)
    return allowed[:, None, :, :]


class DoryAttention(nn.Module):
    def __init__(self, config: DoryConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.q_norm = DoryL2Norm()
        self.k_norm = DoryL2Norm()
        self.attn_sub_norm = DoryL2Norm()
        self.sqk = nn.Parameter(torch.full((self.num_heads * self.head_dim,), config.initializer_range))
        self.s_gate = nn.Parameter(torch.full((self.num_heads * self.head_dim,), config.initializer_range))
        self.sqk_scale = (self.head_dim**0.5) / config.initializer_range
        self.s_gate_scale = (config.hidden_size**0.25) / config.initializer_range

        self.layer_type = config.layer_types[layer_idx]
        self.is_recurrent = self.layer_type.startswith("recurrent_")
        self.is_local_attention = self.layer_type.endswith("sliding_attention")
        self.sliding_window = config.sliding_window if self.is_local_attention else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        past_key_values: DoryRecurrentCache | None = None,
        loop_idx: int = 0,
        is_last_loop: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del kwargs
        bsz, q_len, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        gate_states = self.gate_proj(hidden_states)

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        q_scale = (self.sqk * self.sqk_scale).view(-1, 1, self.head_dim)
        q = (q * q_scale.square()).to(v.dtype)

        if past_key_values is not None:
            advance_seen = (self.layer_idx == self.config.num_hidden_layers - 1) and is_last_loop
            k, v = past_key_values.update(
                k,
                v,
                self.layer_idx,
                loop_idx=loop_idx,
                is_last_loop=is_last_loop,
                is_recurrent=self.is_recurrent,
                is_sliding=self.is_local_attention,
                advance_seen=advance_seen,
            )

        kv_len = k.shape[-2]
        k_rep = _repeat_kv(k, self.num_key_value_groups)
        v_rep = _repeat_kv(v, self.num_key_value_groups)

        if self.sliding_window is None and attention_mask_2d is None and kv_len == q_len and q_len > 1:
            attn_output = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=None, dropout_p=0.0, is_causal=True, scale=self.scaling)
        else:
            attn_mask = _build_sdpa_mask(
                q_len, kv_len, position_ids, attention_mask_2d, sliding_window=self.sliding_window, device=hidden_states.device
            )
            attn_output = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask, dropout_p=0.0, is_causal=False, scale=self.scaling)

        output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
        output = output * torch.sigmoid(gate_states * (self.s_gate * self.s_gate_scale).square())
        return self.o_proj(self.attn_sub_norm(output)), None


class DoryMLP(nn.Module):
    def __init__(self, config: DoryConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.suv_gate = nn.Parameter(torch.ones(config.intermediate_size))
        self.suv_up = nn.Parameter(torch.ones(config.intermediate_size))
        self.suv_scale = config.hidden_size**0.5
        self.ffn_sub_norm = DoryL2Norm()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states) * (self.suv_gate * self.suv_scale)
        up = self.up_proj(hidden_states) * (self.suv_up * self.suv_scale)
        return self.down_proj(self.ffn_sub_norm(F.silu(gate) * up))


class DoryDecoderLayer(nn.Module):
    def __init__(self, config: DoryConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = DoryAttention(config, layer_idx)
        self.mlp = DoryMLP(config)
        self.input_layernorm = DoryL2Norm()
        self.post_attention_layernorm = DoryL2Norm()
        self.pre_feedforward_layernorm = DoryL2Norm()
        self.post_feedforward_layernorm = DoryL2Norm()
        self.lambda_1 = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        self.lambda_2 = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        self.lambda_scale = config.ngpt_alpha_init / config.initializer_range

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask_2d: torch.Tensor | None = None,
        past_key_values: DoryRecurrentCache | None = None,
        loop_idx: int = 0,
        is_last_loop: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask_2d=attention_mask_2d,
            past_key_values=past_key_values,
            loop_idx=loop_idx,
            is_last_loop=is_last_loop,
            **kwargs,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        attn_rate = (self.lambda_1 * self.lambda_scale).abs()
        hidden_states = hidden_states + attn_rate * (attn_output - hidden_states)

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        mlp_output = self.post_feedforward_layernorm(self.mlp(hidden_states))
        mlp_rate = (self.lambda_2 * self.lambda_scale).abs()
        return hidden_states + mlp_rate * (mlp_output - hidden_states)


class DoryLMHead(nn.Linear):
    def __init__(self, config: DoryConfig) -> None:
        super().__init__(config.hidden_size, config.vocab_size, bias=False)
        self.logit_scale = nn.Parameter(torch.full((config.vocab_size,), config.initializer_range))
        self.logit_scale_factor = 1.0 / config.initializer_range

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden_states).float() * (self.logit_scale.float() * self.logit_scale_factor)


class DoryPreTrainedModel(PreTrainedModel):
    config_class = DoryConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["DoryDecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class DoryModel(DoryPreTrainedModel):
    def __init__(self, config: DoryConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DoryDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = DoryL2Norm()
        self.rotary_emb = DoryRotaryEmbedding(config=config)
        self._loop_runner: Callable[..., torch.Tensor] | None = None
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DoryRecurrentCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        del kwargs
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = bool(self.config.use_cache if use_cache is None else use_cache)
        if use_cache and (
            past_key_values is None
            or (not isinstance(past_key_values, DoryRecurrentCache) and past_key_values.get_seq_length() == 0)
        ):
            past_key_values = DoryRecurrentCache(
                num_layers=self.config.num_hidden_layers,
                sliding_window=self.config.sliding_window,
                recurrent_kv_cache_mode=self.config.recurrent_kv_cache_mode,
            )

        q_len = inputs_embeds.shape[1]
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            if attention_mask is not None and past_seen == 0:
                pos = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = pos.masked_fill(attention_mask == 0, 0)
            else:
                position_ids = torch.arange(q_len, device=inputs_embeds.device, dtype=torch.long).unsqueeze(0) + past_seen

        layer_inputs = {
            lt: {
                "position_embeddings": self.rotary_emb(inputs_embeds, position_ids, lt),
                "position_ids": position_ids,
                "attention_mask_2d": attention_mask,
            }
            for lt in set(self.config.layer_types)
        }

        recurrent_start = self.config.n_input_layers
        recurrent_end = recurrent_start + self.config.n_recurrent_layers
        output_start = self.config.num_hidden_layers - self.config.n_output_layers

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[:recurrent_start]:
            hidden_states = decoder_layer(hidden_states, past_key_values=past_key_values, **layer_inputs[decoder_layer.layer_type])

        input_hidden_states = self.norm(hidden_states)
        if self._loop_runner is not None:
            hidden_states = self._loop_runner(
                model=self,
                hidden_states=hidden_states,
                input_hidden_states=input_hidden_states,
                recurrent_layers=self.layers[recurrent_start:recurrent_end],
                layer_inputs=layer_inputs,
                past_key_values=past_key_values,
            )
        else:
            for loop_idx in range(self.config.n_recurrent_loops):
                hidden_states = self.norm(hidden_states) + input_hidden_states
                is_last = loop_idx == (self.config.n_recurrent_loops - 1)
                for decoder_layer in self.layers[recurrent_start:recurrent_end]:
                    hidden_states = decoder_layer(
                        hidden_states,
                        past_key_values=past_key_values,
                        loop_idx=loop_idx,
                        is_last_loop=is_last,
                        **layer_inputs[decoder_layer.layer_type],
                    )

        for decoder_layer in self.layers[output_start:]:
            hidden_states = decoder_layer(hidden_states, past_key_values=past_key_values, **layer_inputs[decoder_layer.layer_type])

        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states), past_key_values=past_key_values)


_CAUSAL_BASES: tuple[type, ...] = (
    (DoryPreTrainedModel,)
    if issubclass(DoryPreTrainedModel, GenerationMixin)
    else (DoryPreTrainedModel, GenerationMixin)
)


class DoryForCausalLM(*_CAUSAL_BASES):  # type: ignore[misc]
    _tied_weights_keys = None

    def __init__(self, config: DoryConfig) -> None:
        super().__init__(config)
        self.model = DoryModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = DoryLMHead(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: DoryRecurrentCache | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if past_key_values is not None and not isinstance(past_key_values, DoryRecurrentCache):
            if past_key_values.get_seq_length() == 0:
                past_key_values = DoryRecurrentCache(
                    num_layers=self.config.num_hidden_layers,
                    sliding_window=self.config.sliding_window,
                    recurrent_kv_cache_mode=self.config.recurrent_kv_cache_mode,
                )
        if past_key_values is not None and isinstance(past_key_values, DoryRecurrentCache):
            past_len = past_key_values.get_seq_length()
            if past_len > 0 and input_ids.shape[1] > past_len:
                input_ids = input_ids[:, past_len:]
            elif past_len > 0:
                input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            pos = attention_mask.long().cumsum(-1) - 1
            pos.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None and isinstance(past_key_values, DoryRecurrentCache) and past_key_values.get_seq_length() > 0:
                pos = pos[:, -input_ids.shape[1] :]
            position_ids = pos
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DoryRecurrentCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, torch.Tensor):
            hidden_states = hidden_states[:, logits_to_keep, :]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)


def remap_dory_state_dict(raw_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert alternating Dory checkpoint block keys (72 half-layers) to 36 paired layers."""
    mapped: dict[str, torch.Tensor] = {}
    for key, tensor in raw_state_dict.items():
        if key == "backbone.embeddings.weight":
            mapped["model.embed_tokens.weight"] = tensor
        elif key in {"lm_head.weight", "lm_head.logit_scale"}:
            mapped[key] = tensor
        elif key.startswith("backbone.layers."):
            parts = key.split(".")
            raw_idx = int(parts[2])
            layer_idx = raw_idx // 2
            is_attn = (raw_idx % 2) == 0
            suffix = ".".join(parts[3:])
            if is_attn:
                if suffix == "attn_alpha":
                    new_key = f"model.layers.{layer_idx}.lambda_1"
                elif suffix == "mixer.g_proj.weight":
                    new_key = f"model.layers.{layer_idx}.self_attn.gate_proj.weight"
                elif suffix.startswith("mixer."):
                    new_key = f"model.layers.{layer_idx}.self_attn.{suffix[len('mixer.'):]}"
                else:
                    raise KeyError(f"Unexpected Dory attention tensor key: {key}")
            else:
                if suffix == "mlp_alpha":
                    new_key = f"model.layers.{layer_idx}.lambda_2"
                elif suffix.startswith("mixer."):
                    new_key = f"model.layers.{layer_idx}.mlp.{suffix[len('mixer.'):]}"
                else:
                    raise KeyError(f"Unexpected Dory MLP tensor key: {key}")
            mapped[new_key] = tensor
        elif key.startswith("model."):
            mapped[key] = tensor
        else:
            raise KeyError(f"Unexpected Dory checkpoint tensor key: {key}")
    return mapped


def _resolve_model_dir(model_path: str | Path, *, local_files_only: bool) -> Path:
    candidate = Path(model_path)
    if candidate.is_dir():
        return candidate.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_path), local_files_only=local_files_only)).resolve()


def load_dory_model(
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    n_recurrent_loops: int | None = None,
    recurrent_kv_cache_mode: str = "last_loop",
    local_files_only: bool = True,
) -> tuple[DoryForCausalLM, Any]:
    """Load `DoryForCausalLM` + tokenizer from local directory or HF Hub ID."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model_dir = _resolve_model_dir(model_path, local_files_only=local_files_only)
    raw_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg["recurrent_kv_cache_mode"] = recurrent_kv_cache_mode
    if n_recurrent_loops is not None:
        raw_cfg["n_recurrent_loops"] = int(n_recurrent_loops)
    config = DoryConfig(**raw_cfg)

    raw_sd: dict[str, torch.Tensor] = {}
    st_single = model_dir / "model.safetensors"
    if st_single.is_file():
        raw_sd = load_file(str(st_single), device="cpu")
    else:
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors checkpoint found under {model_dir}")
        for shard in shards:
            raw_sd.update(load_file(str(shard), device="cpu"))

    remapped = remap_dory_state_dict(raw_sd)
    with torch.device("meta"):
        model = DoryForCausalLM(config)
    model.load_state_dict(remapped, strict=True, assign=True)
    for lt in model.model.rotary_emb.layer_types:
        base = float(config.rope_parameters[lt]["rope_theta"])
        dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
        model.model.rotary_emb.register_buffer(f"{lt}_inv_freq", inv_freq, persistent=False)
    model = model.to(device=device, dtype=dtype)
    for lt in model.model.rotary_emb.layer_types:
        buf = f"{lt}_inv_freq"
        setattr(model.model.rotary_emb, buf, getattr(model.model.rotary_emb, buf).float())
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=local_files_only)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


__all__ = [
    "DoryConfig",
    "DoryForCausalLM",
    "DoryModel",
    "DoryRecurrentCache",
    "load_dory_model",
    "remap_dory_state_dict",
]
