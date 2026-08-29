# TitusAI

TitusAI is a deliberately small decoder-only language model built to keep the complete data, model, and training pipeline easy to read and modify.

## Current model

- WikiText-103 raw training data
- GPT-2 tokenizer with a 50,257-token vocabulary
- 256-token training context
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
- One training epoch
- Progress printed every 20 seconds
- Model weights saved every 10 minutes
- Three rotating checkpoint files are retained

## Files

```text
prepare_data.py    Download, reconstruct, tokenize, and pack WikiText-103
model.py           Language model architecture
README_SOURCES.md  Line-by-line sources for the model architecture
train.py           Two-GPU distributed training loop
inference.py       Minimal inference script; not yet updated for the current model
requirements.txt   Python dependencies
weights/           Packed training tokens and model checkpoints
```

## Install

```bash
pip install -r requirements.txt
```

## Prepare data

```bash
python prepare_data.py
```

Hugging Face downloads WikiText-103 and the GPT-2 tokenizer through its normal cache. The preparation script reconstructs WikiText articles, inserts the GPT-2 end-of-text token between articles, and writes one flat packed `uint16` token stream to:

```text
weights/data.bin
```

No context windows are created during data preparation. Context length belongs to the training process.

## Train

Training requires two CUDA GPUs and is launched with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=2 train.py
```

The flat token stream is memory-mapped during training. Samples contain 257 tokens with a stride of 256: the first 256 tokens are the model input and the following 256 shifted tokens are the targets. Samples are shuffled and divided between the two distributed workers without both GPUs training on the same sample.

Checkpoints rotate between:

```text
weights/model_1.pt
weights/model_2.pt
weights/model_3.pt
```

Once all three exist, the oldest checkpoint is overwritten on the next save.

## Inference

`inference.py` is still a minimal placeholder from the earlier baseline and has not yet been updated for the current model architecture and rotating checkpoint names.
