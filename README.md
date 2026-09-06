# TitusAI

TitusAI is a deliberately small decoder-only language model built to keep the complete data, model, and training pipeline easy to read and modify.

## Model

- FineWeb-Edu, Cosmopedia v2, and TinyStories pretraining with OpenThoughts-114k reasoning post-training
- GPT-2 tokenizer with a 50,257-token vocabulary
- 256 hidden dimensions
- 4 decoder Transformer blocks
- 4 attention heads with 64 dimensions per head
- RoPE, RMSNorm, and SwiGLU
- 256-token pretraining context and 1,024-token post-training context
- About 29 million parameters

## Files

```text
prepare_data.py    Prepare pretraining or post-training data
model.py           Language model architecture
README_SOURCES.md  Line-by-line sources for the model architecture
train.py           Shared two-GPU training loop
inference.py       Minimal inference script
requirements.txt   Python dependencies
weights/           Prepared datasets and model checkpoints
```

## Install

```bash
pip install -r requirements.txt
```

## Data

```bash
python prepare_data.py pretrain
python prepare_data.py posttrain
python prepare_data.py all
```

`pretrain` prepares the 460M-token FineWeb-Edu, Cosmopedia v2, and TinyStories corpus, `posttrain` prepares OpenThoughts-114k reasoning data, and `all` prepares both in sequence.

## Train

```bash
torchrun --standalone --nproc-per-node=2 train.py pretrain
torchrun --standalone --nproc-per-node=2 train.py posttrain
torchrun --standalone --nproc-per-node=2 train.py all
```

`pretrain` trains from scratch, `posttrain` continues from the newest pretraining checkpoint, and `all` runs both stages in sequence. Training uses two CUDA GPUs with DistributedDataParallel and keeps three rotating checkpoints per stage.

## Inference

```bash
python inference.py
```

The inference script loads the newest post-training checkpoint and starts an interactive `user>` prompt. Type `exit` or `quit` to stop.
