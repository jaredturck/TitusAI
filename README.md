# TitusAI

TitusAI is a readable, from-scratch PyTorch language model for studying a modern compact LLM end to end.

The model, training loop, data pipeline, checkpointing, Discord monitoring, generation, and CPU inference are implemented in this repository. Hugging Face is used only for the tokenizer and source datasets.

## Complete workflow

Run these stages in order.

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
hf auth login
```

Accept access to `nvidia/Nemotron-CC-Math-v1` on Hugging Face before preparing data.

### 2. Configure Discord

Create a private Discord webhook, then run:

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

During training, Discord reports startup, loss, TPS, validation, snapshots, progress, ETA, and separate temperature, fan, clock, power, utilization, and thermal-throttling information for GPU 0 and GPU 1.

### 3. Verify the project

```bash
python check_setup.py
pytest -q
```

Optional dual-GPU smoke test:

```bash
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29671 --max_restarts=0 ddp_smoke.py
```

### 4. Prepare pretraining data

Set the desired budget in `PREPARE_CONFIG['max_total_tokens']` inside `config.py`, then run:

```bash
HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=120 PYTHONUNBUFFERED=1 python prepare_data.py
```

Data is written to:

```text
data/processed/train/
data/processed/validation/
```

Preparation is resumable. Existing valid shards are checked and reused, incomplete sources continue from their saved progress, and missing manifests are rebuilt automatically.

### 5. Pretrain

Confirm `TRAIN_CONFIG` in `config.py` is configured for pretraining:

```python
'run_name': 'pretrain',
'train_data_path': TRAIN_DATA_PATH,
'validation_data_path': VALIDATION_DATA_PATH,
'initial_weights': None,
```

Set `TRAIN_CONFIG['max_train_tokens']` to the desired training budget, then start both GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 --max_restarts=0 train.py
```

Press `Ctrl+C` once to save safely. Run the same command again to resume from the newest full checkpoint.

### 6. Inspect pretraining snapshots

```bash
python inference.py
```

Useful commands:

```text
/reload    Load the newest snapshot
/info      Show snapshot information
/direct    Direct-answer format
/reason    Reasoning format
/clear     Clear conversation history
/exit      Exit
```

A base-pretrained model is primarily a next-token predictor. Coherent instruction following is taught in the next stage.

### 7. Prepare instruction data

After pretraining is complete:

```bash
python prepare_instructions.py
```

Instruction shards are written to:

```text
data/processed/instructions/train/
data/processed/instructions/validation/
```

Only assistant tokens contribute to instruction-training loss.

### 8. Configure instruction tuning

The current code does not yet provide a `--run instructions` command. Before starting instruction tuning, change these values in `TRAIN_CONFIG` inside `config.py`:

```python
'run_name': 'instructions',
'train_data_path': PROCESSED_DATA_PATH / 'instructions' / 'train',
'validation_data_path': PROCESSED_DATA_PATH / 'instructions' / 'validation',
'initial_weights': SNAPSHOT_PATH / 'pretrain' / 'snapshot_XX.pt',
'max_train_tokens': YOUR_INSTRUCTION_TOKEN_BUDGET,
'resume_training': True,
```

Replace `snapshot_XX.pt` with the chosen final pretraining snapshot.

Instruction checkpoints and snapshots are stored separately under:

```text
weights/checkpoints/instructions/
weights/snapshots/instructions/
```

### 9. Train the instruction model

Use the same dual-GPU command:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 --max_restarts=0 train.py
```

### 10. Inspect the instruction-tuned model

Set this in `INFERENCE_CONFIG` inside `config.py`:

```python
'snapshot_run': 'instructions',
```

Then run:

```bash
python inference.py
```

## Model

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

Training uses BF16, native PyTorch DistributedDataParallel, AdamW, warmup plus cosine decay, gradient accumulation, gradient clipping, resumable checkpoints, and rolling inference snapshots.

## Pretraining mixture

| Source | Share |
|---|---:|
| DCLM 100BT shuffled | 80% |
| SwallowCode-v2 | 12% |
| Nemotron-CC-Math 4+ | 6% |
| Cosmopedia v2 | 2% |

Documents are deduplicated, split deterministically, tokenized with the SmolLM2 tokenizer, separated with document-end tokens, and packed into 2,048-token sequences.

## Key files

```text
config.py                Model, data, training, and inference settings
model.py                 Transformer architecture
prepare_data.py          Resumable pretraining-data preparation
prepare_instructions.py  Instruction-data preparation
train.py                 Dual-GPU training
notifications.py         Discord embeds and GPU telemetry
checkpoint.py            Snapshots and resumable checkpoints
inference.py             CPU-only model inspection
tests/                   Correctness tests
```

See `DESIGN.md` for implementation details and `REFERENCES.md` for research sources.

## Jared-PC GPU training setup

These commands are specific to the dual RTX 3090 setup on Jared-PC.

Set both cards to a temporary 300 W limit and force all four reported fans to 100%:

```bash
sudo nvidia-smi -i 0 -pl 300
sudo nvidia-smi -i 1 -pl 300

sudo --preserve-env=DISPLAY,WAYLAND_DISPLAY,XDG_RUNTIME_DIR,DBUS_SESSION_BUS_ADDRESS \
    nvidia-settings -c wayland-0 \
    -a '[gpu:0]/GPUFanControlState=1' \
    -a '[gpu:1]/GPUFanControlState=1' \
    -a '[fan:0]/GPUTargetFanSpeed=100' \
    -a '[fan:1]/GPUTargetFanSpeed=100' \
    -a '[fan:2]/GPUTargetFanSpeed=100' \
    -a '[fan:3]/GPUTargetFanSpeed=100'
```

Start training:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 --max_restarts=0 train.py
```

The power and fan settings normally reset after reboot or an NVIDIA driver/session restart.

Restore automatic fan control:

```bash
sudo --preserve-env=DISPLAY,WAYLAND_DISPLAY,XDG_RUNTIME_DIR,DBUS_SESSION_BUS_ADDRESS \
    nvidia-settings -c wayland-0 \
    -a '[gpu:0]/GPUFanControlState=0' \
    -a '[gpu:1]/GPUFanControlState=0'
```

Restore the stock 350 W limits:

```bash
sudo nvidia-smi -i 0 -pl 350
sudo nvidia-smi -i 1 -pl 350
```
