''' Define the language model architecture. '''

import torch
from torch import nn
from torch.nn import functional as F

def apply_rope(tensor, cos, sin):
    ''' Apply rotary position embeddings. '''
    even = tensor[..., ::2]
    odd = tensor[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

class TransformerBlock(nn.Module):
    ''' Apply one decoder Transformer block. '''

    def __init__(self, d_model):
        ''' Build attention and feed-forward layers. '''
        super().__init__()

        self.attention_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.attention_output = nn.Linear(d_model, d_model, bias=False)

        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.ffn_up = nn.Linear(d_model, d_model * 6, bias=False)
        self.ffn_down = nn.Linear(d_model * 3, d_model, bias=False)

    def forward(self, hidden, cos, sin):
        ''' Apply attention and SwiGLU transformations. '''
        batch, sequence, d_model = hidden.shape

        query, key, value = self.qkv(self.attention_norm(hidden)).chunk(3, dim=-1)
        query = query.view(batch, sequence, d_model // 64, 64).transpose(1, 2)
        key = key.view(batch, sequence, d_model // 64, 64).transpose(1, 2)
        value = value.view(batch, sequence, d_model // 64, 64).transpose(1, 2)

        cos = cos.to(query.dtype)
        sin = sin.to(query.dtype)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)

        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch, sequence, d_model)
        hidden = hidden + self.attention_output(attention)

        gate, value = self.ffn_up(self.ffn_norm(hidden)).chunk(2, dim=-1)
        return hidden + self.ffn_down(F.silu(gate) * value)

class LanguageModel(nn.Module):
    ''' Predict the next token with a decoder-only Transformer. '''

    def __init__(self):
        ''' Build the language model. '''
        super().__init__()

        d_model = 256

        self.token_embedding = nn.Embedding(50257, d_model)

        self.layer_1 = TransformerBlock(d_model)
        self.layer_2 = TransformerBlock(d_model)
        self.layer_3 = TransformerBlock(d_model)
        self.layer_4 = TransformerBlock(d_model)
        self.layer_5 = TransformerBlock(d_model)
        self.layer_6 = TransformerBlock(d_model)
        self.layer_7 = TransformerBlock(d_model)
        self.layer_8 = TransformerBlock(d_model)
        self.layer_9 = TransformerBlock(d_model)
        self.layer_10 = TransformerBlock(d_model)
        self.layer_11 = TransformerBlock(d_model)
        self.layer_12 = TransformerBlock(d_model)

        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.output = nn.Linear(d_model, 50257, bias=False)
        self.output.weight = self.token_embedding.weight

        self.register_buffer('rope_frequencies', 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64)), persistent=False)

    def forward(self, input_ids):
        ''' Return vocabulary logits for each token. '''
        hidden = self.token_embedding(input_ids)

        positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=self.rope_frequencies.dtype)
        frequencies = torch.outer(positions, self.rope_frequencies)
        cos = frequencies.cos()
        sin = frequencies.sin()

        hidden = self.layer_1(hidden, cos, sin)
        hidden = self.layer_2(hidden, cos, sin)
        hidden = self.layer_3(hidden, cos, sin)
        hidden = self.layer_4(hidden, cos, sin)
        hidden = self.layer_5(hidden, cos, sin)
        hidden = self.layer_6(hidden, cos, sin)
        hidden = self.layer_7(hidden, cos, sin)
        hidden = self.layer_8(hidden, cos, sin)
        hidden = self.layer_9(hidden, cos, sin)
        hidden = self.layer_10(hidden, cos, sin)
        hidden = self.layer_11(hidden, cos, sin)
        hidden = self.layer_12(hidden, cos, sin)

        return self.output(self.norm(hidden))
