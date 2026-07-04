# TitusAI

TitusAI is a readable, from-scratch PyTorch language model built for learning how a modern compact LLM works end to end.

The model, training loop, generation code, dataset pipeline, checkpointing, and CPU inference are implemented in this repository. Hugging Face is used only for the tokenizer and source datasets.

## Quick start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Authenticate with Hugging Face and accept access to `nvidia/Nemotron-CC-Math-v1`:

```bash
hf auth login
```

### 2. Verify the project

```bash
python check_setup.py
pytest -q
```

Optional dual-GPU smoke test:

```bash
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29671 --max_restarts=0 ddp_smoke.py
```

### 3. Prepare the dataset

Set the desired token budget in `PREPARE_CONFIG['max_total_tokens']` inside `config.py`, then run:

```bash
HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=120 PYTHONUNBUFFERED=1 python prepare_data.py
```

Prepared shards are written to:

```text
data/processed/train/
data/processed/validation/
```

Preparation is resumable. Existing valid shards are checked and reused, completed sources are skipped, and missing manifests are rebuilt automatically.

### 4. Configure Discord monitoring

Create a private Discord webhook and store it in `.env`:

```bash
cp .env.example .env
nano .env
chmod 600 .env
python notifications.py
```

`.env` should contain:

```text
STATUS_WEBHOOK=https://discord.com/api/webhooks/...
```

Discord receives a startup embed, progress updates roughly every ten minutes, validation results, completion, interruption, and fatal-error notifications.

### 5. Train on both GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 --max_restarts=0 train.py
```

Training uses native PyTorch DistributedDataParallel. Hugging Face Accelerate is not required.

Press `Ctrl+C` once to stop safely. Restart the same command to resume from the newest full checkpoint.

### 6. Inspect current weights

Rolling inference snapshots are saved approximately every ten minutes, with the newest ten retained.

Run CPU-only inference with:

```bash
python inference.py
```

Useful commands:

```text
/reload    Load the newest snapshot
/info      Show checkpoint information
/direct    Direct-answer mode
/reason    Reasoning mode
/clear     Clear conversation history
/exit      Exit
```

## Model architecture

| Component | Value |
|---|---:|
| Parameters | 158.6M |
| Decoder layers | 30 |
| Hidden dimension | 576 |
| Query / KV heads | 9 / 3 |
| Head dimension | 64 |
| Feed-forward | SwiGLU, width 2,000 |
| Normalization | Pre-RMSNorm and QK-RMSNorm |
| Position encoding | RoPE |
| Context length | 2,048 |
| Attention | Causal grouped-query attention |
| Output | Tied embeddings with full cross-entropy |

Training uses BF16, AdamW, warmup plus cosine decay, gradient accumulation, gradient clipping, shard-aware distributed sampling, rolling inference snapshots, and resumable checkpoints.

## Pretraining data

The configured mixture is:

| Source | Share |
|---|---:|
| DCLM 100BT shuffled | 80% |
| SwallowCode-v2 | 12% |
| Nemotron-CC-Math 4+ | 6% |
| Cosmopedia v2 | 2% |

Documents are deduplicated, split deterministically into training and validation sets, tokenized with the SmolLM2 tokenizer, separated with document-end tokens, and packed into 2,048-token sequences.

## Instruction tuning

Prepare Smol-SmolTalk with:

```bash
python prepare_instructions.py
```

Update the instruction-run paths and initial pretraining weights in `config.py`, then launch the same `torchrun` training command.

## Key files

```text
model.py                 Transformer architecture
train.py                 Dual-GPU training loop
prepare_data.py          Base-data preparation
prepare_instructions.py  Instruction-data preparation
dataset.py               Memory-mapped dataset loader
notifications.py         Discord embeds
checkpoint.py            Snapshots and resumable checkpoints
inference.py             CPU-only model inspection
config.py                Project configuration
```

See `DESIGN.md` for implementation details and `REFERENCES.md` for the research behind the architecture and dataset choices.

## Notes

- The final model quality depends heavily on the training-token budget and later instruction tuning.
- A 158.6M model is useful as a compact research system, but it will not match modern multi-billion-parameter assistants.
- Dataset licenses and access terms remain the responsibility of the person running the training job.
