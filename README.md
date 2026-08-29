# TitusAI

TitusAI is a deliberately small decoder-only language model built to keep the complete data, model, and training pipeline easy to read and modify.

## Current model

- WikiText-103 raw data for pretraining
- OpenThoughts-114k reasoning data for post-training
- GPT-2 tokenizer with a 50,257-token vocabulary
- 256-token pretraining context
- 1,024-token post-training context
- 256 hidden dimensions
- 4 explicit decoder Transformer blocks
- 4 attention heads with 64 dimensions per head
- RoPE positional encoding
- RMSNorm
- SwiGLU feed-forward layers with a 768-dimension intermediate width
- PyTorch scaled dot-product causal attention
- No dropout
- About 29 million parameters

## Training

- Two CUDA GPUs with PyTorch DistributedDataParallel and NCCL
- 32 samples per GPU, giving a global batch size of 64
- AdamW optimizer
- Peak learning rate of `3e-4`
- 100-step linear warmup followed by cosine decay to `3e-5`
- One training epoch per stage
- Progress printed every 20 seconds
- Model weights saved every 10 minutes
- Three rotating checkpoint files are retained for each training stage

## Files

```text
prepare_data.py    Prepare WikiText pretraining or OpenThoughts post-training data
model.py           Language model architecture
README_SOURCES.md  Line-by-line sources for the model architecture
train.py           Shared two-GPU pretraining and post-training loop
inference.py       Minimal inference script; not yet updated for the current model
requirements.txt   Python dependencies
weights/           Prepared datasets and model checkpoints
```

## Install

```bash
pip install -r requirements.txt
```

## Prepare pretraining data

```bash
python prepare_data.py pretrain
```

This reconstructs WikiText-103 articles, inserts the GPT-2 end-of-text token between articles, and writes one flat packed `uint16` token stream to:

```text
weights/data.bin
```

## Prepare post-training data

```bash
python prepare_data.py posttrain
```

This downloads the `metadata` subset of `open-thoughts/OpenThoughts-114k` and formats each retained example as a user problem followed by an assistant `<think>...</think>` reasoning trace and final answer. Examples longer than the 1,024-token post-training context are skipped rather than truncated.

Post-training preparation writes fixed 1,025-token `uint16` samples and a matching assistant loss mask to:

```text
weights/posttrain.bin
weights/posttrain_mask.bin
```

Prompt and padding tokens are masked out so only the assistant reasoning and answer contribute to the post-training loss.

## Train

Training requires two CUDA GPUs and is launched with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=2 train.py pretrain
torchrun --standalone --nproc-per-node=2 train.py posttrain
torchrun --standalone --nproc-per-node=2 train.py all
```

Pretraining memory-maps the WikiText stream and creates 257-token samples with a stride of 256. Post-training initializes from the newest pretraining checkpoint and trains on the masked OpenThoughts reasoning samples. The `all` mode runs pretraining first and post-training second in the same launch.

Pretraining checkpoints rotate between:

```text
weights/model_1.pt
weights/model_2.pt
weights/model_3.pt
```

Post-training checkpoints rotate between:

```text
weights/posttrain_1.pt
weights/posttrain_2.pt
weights/posttrain_3.pt
```

## Inference

`inference.py` is still a minimal placeholder from the earlier baseline and has not yet been updated for the current model architecture and rotating checkpoint names.
