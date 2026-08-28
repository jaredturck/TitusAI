# TitusAI

TitusAI is currently a deliberately minimal language-model training baseline. The goal is to keep the entire project easy to read before adding more advanced architecture, training, and post-training features.

## Baseline

- WikiText-103 raw training data
- GPT-2 tokenizer
- 256-token context
- 256 hidden dimensions
- 4 PyTorch Transformer layers
- 8 attention heads
- 1,024-dimension feed-forward layers
- About 29 million parameters
- Single-GPU training
- AdamW and cross-entropy loss
- Greedy next-token inference

## Files

```text
prepare_data.py  Download and tokenize WikiText-103
model.py         Language model definition
train.py         Single-GPU training loop
inference.py     Load weights and generate text
requirements.txt Python dependencies
weights/         Prepared tokens and trained model weights
```

## Install

```bash
pip install -r requirements.txt
```

## Prepare data

```bash
python prepare_data.py
```

Hugging Face handles the dataset and tokenizer downloads through its normal cache. The prepared token tensor is saved to `weights/data.pt`.

## Train

```bash
python train.py
```

Training runs for one epoch on one CUDA GPU and saves the finished model to `weights/model.pt`.

## Inference

```bash
python inference.py
```

The prompt and weight path are intentionally hardcoded in `inference.py` for now.
