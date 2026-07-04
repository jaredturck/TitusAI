import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta=10_000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
            )
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def get_cos_sin(self, sequence_length, start_pos, device, dtype):
        positions = torch.arange(
            start_pos,
            start_pos + sequence_length,
            device=device,
            dtype=torch.float32,
        )
        frequencies = torch.outer(positions, self.inv_freq.to(device))
        cos = frequencies.cos().to(dtype)[None, None, :, :]
        sin = frequencies.sin().to(dtype)[None, None, :, :]
        return cos, sin

    def apply_rotary(self, hidden_states, cos, sin):
        first_half = hidden_states[..., :self.head_dim // 2]
        second_half = hidden_states[..., self.head_dim // 2:]
        rotated_first = first_half * cos - second_half * sin
        rotated_second = second_half * cos + first_half * sin
        return torch.cat((rotated_first, rotated_second), dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config, rotary_embedding):
        super().__init__()
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.num_kv_heads = config['num_kv_heads']
        self.head_dim = config['head_dim']
        self.dropout = config['dropout']
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.rotary_embedding = rotary_embedding

        assert self.num_heads % self.num_kv_heads == 0
        assert self.num_heads * self.head_dim == self.d_model

        self.q_proj = nn.Linear(
            self.d_model,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config['rms_norm_eps'])
        self.k_norm = RMSNorm(self.head_dim, config['rms_norm_eps'])

    def reshape_heads(self, hidden_states, num_heads):
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size,
            sequence_length,
            num_heads,
            self.head_dim,
        )
        return hidden_states.transpose(1, 2)

    def build_attention_mask(self, segment_ids, query_length, key_length, start_pos, device):
        query_positions = torch.arange(
            start_pos,
            start_pos + query_length,
            device=device,
        )
        key_positions = torch.arange(key_length, device=device)
        causal_mask = key_positions[None, :] <= query_positions[:, None]

        if segment_ids is None:
            return causal_mask[None, None, :, :]

        same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
        return same_segment[:, None, :, :] & causal_mask[None, None, :, :]

    def forward(self, hidden_states, segment_ids=None, kv_cache=None, start_pos=0, use_cache=False):
        batch_size, query_length, _ = hidden_states.shape

        queries = self.reshape_heads(self.q_proj(hidden_states), self.num_heads)
        keys = self.reshape_heads(self.k_proj(hidden_states), self.num_kv_heads)
        values = self.reshape_heads(self.v_proj(hidden_states), self.num_kv_heads)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        cos, sin = self.rotary_embedding.get_cos_sin(
            query_length,
            start_pos,
            hidden_states.device,
            hidden_states.dtype,
        )
        queries = self.rotary_embedding.apply_rotary(queries, cos, sin)
        keys = self.rotary_embedding.apply_rotary(keys, cos, sin)

        if kv_cache is not None:
            cached_keys, cached_values = kv_cache
            keys = torch.cat((cached_keys, keys), dim=2)
            values = torch.cat((cached_values, values), dim=2)

        new_cache = (keys, values) if use_cache else None
        key_length = keys.size(2)

        dropout_p = self.dropout if self.training else 0.0
        can_use_causal_kernel = segment_ids is None and kv_cache is None

        if can_use_causal_kernel:
            attention_output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=dropout_p,
                is_causal=True,
                enable_gqa=True,
            )
        else:
            attention_mask = self.build_attention_mask(
                segment_ids,
                query_length,
                key_length,
                start_pos,
                hidden_states.device,
            )
            attention_output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=False,
                enable_gqa=True,
            )

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size,
            query_length,
            self.d_model,
        )
        return self.o_proj(attention_output), new_cache


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config['d_model'],
            config['ffn_dim'],
            bias=False,
        )
        self.value_proj = nn.Linear(
            config['d_model'],
            config['ffn_dim'],
            bias=False,
        )
        self.down_proj = nn.Linear(
            config['ffn_dim'],
            config['d_model'],
            bias=False,
        )

    def forward(self, hidden_states):
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.value_proj(hidden_states)
        )


class TransformerBlock(nn.Module):
    def __init__(self, config, rotary_embedding):
        super().__init__()
        self.attention_norm = RMSNorm(
            config['d_model'],
            config['rms_norm_eps'],
        )
        self.attention = GroupedQueryAttention(config, rotary_embedding)
        self.ffn_norm = RMSNorm(
            config['d_model'],
            config['rms_norm_eps'],
        )
        self.feed_forward = SwiGLU(config)

    def forward(self, hidden_states, segment_ids=None, kv_cache=None, start_pos=0, use_cache=False):
        attention_output, new_cache = self.attention(
            self.attention_norm(hidden_states),
            segment_ids=segment_ids,
            kv_cache=kv_cache,
            start_pos=start_pos,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(
            self.ffn_norm(hidden_states)
        )
        return hidden_states, new_cache

    def forward_no_cache(self, hidden_states, segment_ids=None):
        hidden_states, _ = self.forward(
            hidden_states,
            segment_ids=segment_ids,
            kv_cache=None,
            start_pos=0,
            use_cache=False,
        )
        return hidden_states


class TitusModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_seq_len']
        self.loss_chunk_size = config['loss_chunk_size']
        self.gradient_checkpointing = False

        assert self.vocab_size is not None
        assert config['num_heads'] * config['head_dim'] == config['d_model']

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            config['d_model'],
        )
        self.rotary_embedding = RotaryEmbedding(
            config['head_dim'],
            config['max_seq_len'],
            config['rope_theta'],
        )
        self.layers = nn.ModuleList([
            TransformerBlock(config, self.rotary_embedding)
            for _ in range(config['num_layers'])
        ])
        self.final_norm = RMSNorm(
            config['d_model'],
            config['rms_norm_eps'],
        )

        self.apply(self.initialize_weights)
        self.initialize_residual_weights()

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def initialize_residual_weights(self):
        residual_std = 0.02 / math.sqrt(2 * self.config['num_layers'])
        for layer in self.layers:
            nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.feed_forward.down_proj.weight, mean=0.0, std=residual_std)

    def enable_gradient_checkpointing(self, enabled=True):
        self.gradient_checkpointing = enabled

    def compute_loss(self, hidden_states, labels):
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_labels = labels.reshape(-1)
        loss_sum = hidden_states.new_zeros((), dtype=torch.float32)
        valid_tokens = (flat_labels != -100).sum()

        for start in range(0, flat_hidden.size(0), self.loss_chunk_size):
            end = min(start + self.loss_chunk_size, flat_hidden.size(0))
            logits = F.linear(
                flat_hidden[start:end],
                self.token_embedding.weight,
            )
            loss_sum = loss_sum + F.cross_entropy(
                logits.float(),
                flat_labels[start:end],
                ignore_index=-100,
                reduction='sum',
            )

        loss = loss_sum / valid_tokens.clamp_min(1)
        return loss, loss_sum, valid_tokens

    def forward(self, input_ids, labels=None, segment_ids=None, kv_cache=None, start_pos=0, use_cache=False, return_logits=True):
        batch_size, sequence_length = input_ids.shape
        assert start_pos + sequence_length <= self.max_seq_len

        if kv_cache is not None:
            assert len(kv_cache) == len(self.layers)

        hidden_states = self.token_embedding(input_ids)
        new_kv_cache = [] if use_cache else None

        for layer_index, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[layer_index]

            if self.gradient_checkpointing and self.training and not use_cache:
                hidden_states = checkpoint(
                    layer.forward_no_cache,
                    hidden_states,
                    segment_ids,
                    use_reentrant=False,
                )
                new_layer_cache = None
            else:
                hidden_states, new_layer_cache = layer(
                    hidden_states,
                    segment_ids=segment_ids,
                    kv_cache=layer_cache,
                    start_pos=start_pos,
                    use_cache=use_cache,
                )

            if use_cache:
                new_kv_cache.append(new_layer_cache)

        hidden_states = self.final_norm(hidden_states)

        logits = None
        if return_logits:
            logits = F.linear(hidden_states, self.token_embedding.weight)

        loss = None
        loss_sum = None
        loss_tokens = None
        if labels is not None:
            loss, loss_sum, loss_tokens = self.compute_loss(
                hidden_states,
                labels,
            )

        return {
            'logits': logits,
            'loss': loss,
            'loss_sum': loss_sum,
            'loss_tokens': loss_tokens,
            'kv_cache': new_kv_cache,
        }

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
