import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tiny_config():
    return {
        'vocab_size': 128,
        'd_model': 64,
        'num_layers': 2,
        'num_heads': 4,
        'num_kv_heads': 2,
        'head_dim': 16,
        'ffn_dim': 128,
        'max_seq_len': 64,
        'rms_norm_eps': 1e-5,
        'rope_theta': 10_000.0,
        'dropout': 0.0,
        'loss_chunk_size': 16,
    }
