# Model Sources

**Synchronization requirement:** this file must mirror `model.py`. Any future edit to `model.py` must update this file in the same commit.

The code below preserves every line of `model.py` in the same order. Every non-blank line is copied verbatim before its trailing source comment. Blank lines are preserved. Source 0 marks documentation or TitusAI-specific design choices rather than pretending those lines are dictated by external research.

`model.py` SHA-256: `690ca303e42900c3ab4b6d0cf213d2b35d320b8649595557e61e9fbb8daabbe3`

## Audited model mirror

```python
''' Define the language model architecture. '''  # Source 0 — TitusAI documentation

import torch  # Source 1 — PyTorch API
from torch import nn  # Source 1 — PyTorch API
from torch.nn import functional as F  # Source 1 — PyTorch API

def apply_rope(tensor, cos, sin):  # Sources 2, 5 — RoFormer; Meta Llama 3
    ''' Apply rotary position embeddings. '''  # Source 0 — TitusAI documentation
    even = tensor[..., ::2]  # Sources 2, 5 — RoFormer; Meta Llama 3
    odd = tensor[..., 1::2]  # Sources 2, 5 — RoFormer; Meta Llama 3
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)  # Sources 2, 5 — RoFormer; Meta Llama 3

class TransformerBlock(nn.Module):  # Sources 1, 5 — PyTorch Module; Meta Llama 3
    ''' Apply one decoder Transformer block. '''  # Source 0 — TitusAI documentation

    def __init__(self, d_model):  # Source 1 — PyTorch Module
        ''' Build attention and feed-forward layers. '''  # Source 0 — TitusAI documentation
        super().__init__()  # Source 1 — PyTorch Module

        self.attention_norm = nn.RMSNorm(d_model, eps=1e-6)  # Sources 3, 5, 7 — RMSNorm; Llama 3; Qwen3
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)  # Sources 5, 8 — Llama 3 attention; vLLM fused QKV
        self.attention_output = nn.Linear(d_model, d_model, bias=False)  # Sources 5, 7 — Llama 3; Qwen3

        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-6)  # Sources 3, 5, 7 — RMSNorm; Llama 3; Qwen3
        self.ffn_up = nn.Linear(d_model, d_model * 6, bias=False)  # Sources 4, 7, 8 — SwiGLU; Qwen3 3x MLP; fused gate/up
        self.ffn_down = nn.Linear(d_model * 3, d_model, bias=False)  # Sources 4, 5, 7 — SwiGLU; Llama 3; Qwen3

    def forward(self, hidden, cos, sin):  # Source 1 — PyTorch Module
        ''' Apply attention and SwiGLU transformations. '''  # Source 0 — TitusAI documentation
        batch, sequence, d_model = hidden.shape  # Source 1 — PyTorch Tensor API

        query, key, value = self.qkv(self.attention_norm(hidden)).chunk(3, dim=-1)  # Sources 5, 8 — Pre-norm attention; fused QKV
        query = query.view(batch, sequence, d_model // 64, 64).transpose(1, 2)  # Sources 6, 9 — Llama 3.2 64-dim heads; PyTorch SDPA layout
        key = key.view(batch, sequence, d_model // 64, 64).transpose(1, 2)  # Sources 6, 9 — Llama 3.2 64-dim heads; PyTorch SDPA layout
        value = value.view(batch, sequence, d_model // 64, 64).transpose(1, 2)  # Sources 6, 9 — Llama 3.2 64-dim heads; PyTorch SDPA layout

        cos = cos.to(query.dtype)  # Source 1 — PyTorch dtype conversion
        sin = sin.to(query.dtype)  # Source 1 — PyTorch dtype conversion
        query = apply_rope(query, cos, sin)  # Sources 2, 5 — RoPE on queries
        key = apply_rope(key, cos, sin)  # Sources 2, 5 — RoPE on keys

        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)  # Source 9 — PyTorch causal SDPA / FlashAttention dispatch
        attention = attention.transpose(1, 2).reshape(batch, sequence, d_model)  # Sources 1, 5 — Tensor reshape; Llama head merge
        hidden = hidden + self.attention_output(attention)  # Source 5 — Llama residual attention

        gate, value = self.ffn_up(self.ffn_norm(hidden)).chunk(2, dim=-1)  # Sources 4, 7, 8 — SwiGLU; Qwen3; fused gate/up
        return hidden + self.ffn_down(F.silu(gate) * value)  # Sources 4, 5 — SwiGLU; Llama residual FFN

class LanguageModel(nn.Module):  # Sources 1, 5 — PyTorch Module; decoder LM structure
    ''' Predict the next token with a decoder-only Transformer. '''  # Source 0 — TitusAI documentation

    def __init__(self):  # Source 1 — PyTorch Module
        ''' Build the language model. '''  # Source 0 — TitusAI documentation
        super().__init__()  # Source 1 — PyTorch Module

        d_model = 256  # Source 0 — TitusAI model-width choice

        self.token_embedding = nn.Embedding(50257, d_model)  # Sources 1, 10 — PyTorch Embedding; GPT-2 vocabulary

        self.layer_1 = TransformerBlock(d_model)  # Source 0 — TitusAI four-layer choice
        self.layer_2 = TransformerBlock(d_model)  # Source 0 — TitusAI four-layer choice
        self.layer_3 = TransformerBlock(d_model)  # Source 0 — TitusAI four-layer choice
        self.layer_4 = TransformerBlock(d_model)  # Source 0 — TitusAI four-layer choice

        self.norm = nn.RMSNorm(d_model, eps=1e-6)  # Sources 3, 5, 7 — Final RMSNorm precedent
        self.output = nn.Linear(d_model, 50257, bias=False)  # Sources 1, 11 — PyTorch Linear; OLMo 2 untied LM head

        self.register_buffer('rope_frequencies', 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64)), persistent=False)  # Sources 1, 2, 5 — register_buffer; RoPE frequencies; Llama implementation

    def forward(self, input_ids):  # Source 1 — PyTorch Module
        ''' Return vocabulary logits for each token. '''  # Source 0 — TitusAI documentation
        hidden = self.token_embedding(input_ids)  # Source 1 — PyTorch Embedding

        positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=self.rope_frequencies.dtype)  # Sources 1, 5 — torch.arange; Llama position construction
        frequencies = torch.outer(positions, self.rope_frequencies)  # Sources 1, 2, 5 — torch.outer; RoPE frequency construction
        cos = frequencies.cos()  # Sources 2, 5 — RoPE cosine coefficients
        sin = frequencies.sin()  # Sources 2, 5 — RoPE sine coefficients

        hidden = self.layer_1(hidden, cos, sin)  # Source 0 — Explicit TitusAI layer wiring
        hidden = self.layer_2(hidden, cos, sin)  # Source 0 — Explicit TitusAI layer wiring
        hidden = self.layer_3(hidden, cos, sin)  # Source 0 — Explicit TitusAI layer wiring
        hidden = self.layer_4(hidden, cos, sin)  # Source 0 — Explicit TitusAI layer wiring

        return self.output(self.norm(hidden))  # Sources 5, 11 — Final normalization and LM projection
```

## Sources

- **Source 0 — TitusAI local design.** Project-specific documentation and deliberately chosen scale values such as `d_model = 256`, four explicit blocks, and the decision to keep those choices literal rather than configurable. This source exists to distinguish local choices from research-backed architectural mechanics.
- **Source 1 — PyTorch 2.13 documentation.** Supports `nn.Module`, `nn.Embedding`, `nn.Linear`, `nn.RMSNorm`, tensor reshaping/dtype operations, `register_buffer`, `torch.arange`, `torch.outer`, and the standard `forward` module pattern. https://docs.pytorch.org/docs/stable/
- **Source 2 — RoFormer: Enhanced Transformer with Rotary Position Embedding.** Establishes RoPE, including rotating paired query/key coordinates with position-dependent sine and cosine terms. https://arxiv.org/abs/2104.09864
- **Source 3 — Root Mean Square Layer Normalization.** Establishes RMSNorm as a normalization method without mean-centering. https://arxiv.org/abs/1910.07467
- **Source 4 — GLU Variants Improve Transformer.** Establishes SwiGLU-style gated feed-forward layers using a SiLU/Swish gate multiplied by a second projected branch. https://arxiv.org/abs/2002.05202
- **Source 5 — Meta Llama 3 reference implementation.** Supports the modern decoder ordering used here: pre-RMSNorm attention, residual addition, pre-RMSNorm SwiGLU feed-forward, residual addition, RoPE on queries and keys, head reshaping, attention output projection, final RMSNorm, and vocabulary projection. https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py
- **Source 6 — Meta Llama 3.2 architecture definitions.** Meta's official 1B definition uses `dim = 2048` and `n_heads = 32`, giving 64 dimensions per attention head; this is the contemporary precedent for the 64-dimensional heads used here. https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py
- **Source 7 — Qwen3 implementation and 0.6B configuration.** Provides current corroboration for RMSNorm with `1e-6`, bias-free attention, zero attention dropout, SiLU, RoPE, and a 3x intermediate MLP width in Qwen3-0.6B. https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modular_qwen3.py and https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
- **Source 8 — vLLM Qwen3 implementation.** Demonstrates high-performance fused QKV projection and fused gate/up projection while preserving the same Q/K/V and SwiGLU mathematics. https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py
- **Source 9 — PyTorch scaled dot-product attention.** Defines the causal SDPA API used directly by the block. PyTorch documents automatic selection among supported attention backends, including FlashAttention-2 when inputs and hardware are eligible. https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- **Source 10 — GPT-2 configuration.** Confirms the GPT-2 tokenizer/model vocabulary size of 50,257 used by TitusAI's tokenizer and embedding/output dimensions. https://huggingface.co/openai-community/gpt2/blob/main/config.json
- **Source 11 — OLMo 2 1B configuration.** Provides a current production precedent for an untied language-model output head (`tie_word_embeddings: false`), bias-free attention, zero attention dropout, RMSNorm, RoPE, and SiLU. https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/config.json
