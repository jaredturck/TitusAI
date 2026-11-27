import sqlite3, sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from torch.nn.functional import linear

sys.path.insert(0, ".")
from models import TitusAI

class ExportAttention(nn.Module):
    '''Express self-attention with sequence dimensions that stay dynamic in ONNX.'''

    def __init__(self, attention):
        '''Keep the trained attention weights and head dimensions.'''
        super().__init__()
        self.attention = attention
        self.batch_first = attention.batch_first
        self.heads = attention.num_heads
        self.head_width = attention.head_dim

    def forward(self, query, key, value, attn_mask, key_padding_mask, need_weights, is_causal):
        '''Apply the same masked attention using exportable tensor operations.'''
        batch, length, width = query.shape
        projected = linear(query, self.attention.in_proj_weight, self.attention.in_proj_bias)
        projected = projected.reshape(batch, length, 3, self.heads, self.head_width)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind()
        scores = (query @ key.transpose(-1, -2)) * self.head_width ** -0.5
        attended = (scores + attn_mask).softmax(-1) @ value
        hidden = attended.transpose(1, 2).reshape(batch, length, width)
        return self.attention.out_proj(hidden), None

class LastTokenProjection(nn.Module):
    '''Project the final position to vocabulary logits for browser generation.'''

    def __init__(self, projection):
        '''Keep the trained output projection.'''
        super().__init__()
        self.projection = projection

    def forward(self, hidden):
        '''Return the next-token logits.'''
        return self.projection(hidden[:, -1])

def export_model():
    '''Quantize model.pt and export its tokenizer for ONNX Runtime Web.'''
    assets = Path("frontend/public/model")
    with closing(sqlite3.connect("file:datasets.sqlite?mode=ro", uri=True)) as database:
        tokenizer = database.execute("SELECT value FROM build_info WHERE key = 'tokenizer'").fetchone()[0]

    model = TitusAI()
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    for layer in model.transformer.layers:
        layer.self_attn = ExportAttention(layer.self_attn)
    model.output = LastTokenProjection(model.output)
    model.eval()
    torch.backends.mha.set_fastpath_enabled(False)

    assets.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as directory:
        unquantized = Path(directory) / "titusai-fp32.onnx"
        torch.onnx.export(
            model, (torch.ones((1, 16), dtype=torch.long),), unquantized,
            input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {1: "sequence"}}, opset_version=17, dynamo=False,
        )
        quantize_dynamic(
            unquantized, assets / "titusai.onnx", op_types_to_quantize=["Gather", "MatMul"],
            weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
        )
    (assets / "tokenizer.model").write_bytes(tokenizer)

if __name__ == "__main__":
    export_model()
