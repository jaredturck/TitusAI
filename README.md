# TitusAI

TitusAI is a readable, from-scratch PyTorch language model for studying a compact modern LLM from data preparation through pretraining, instruction tuning, and inference.

## Training workflow

Run these stages in order from the project root.

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
hf auth login
```

Accept access to `nvidia/Nemotron-CC-Math-v1` on Hugging Face before preparing pretraining data.

### 2. Configure Discord notifications

```bash
nano .env
chmod 600 .env
python notifications.py
```

Set `STATUS_WEBHOOK` in `.env`. Training confirms the startup notification before entering the training loop, then sends progress every 10 minutes.

### 3. Verify the project

```bash
python check_setup.py
pytest -q
```

The optional dual-GPU smoke test is:

```bash
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29671 --max_restarts=0 ddp_smoke.py
```

### 4. Prepare pretraining data

```bash
HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=120 PYTHONUNBUFFERED=1 python prepare_data.py
```

The token budget is set in `PREPARE_CONFIG` inside `config.py`. Prepared shards are written under `data/processed/train/` and `data/processed/validation/`.

### 5. Run pretraining

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 --max_restarts=0 train.py pretrain
```

The training mode is mandatory. Press `Ctrl+C` once to save safely, then run the same command to resume from the newest pretraining checkpoint.

### 6. Inspect the pretrained model

```bash
python inference.py
```

Use `/reload` to load the newest snapshot and `/info` to inspect it.

### 7. Prepare instruction data

```bash
python prepare_instructions.py
```

Progress is printed about every five seconds with processed conversations, percentage, speed, elapsed time, and ETA. Output is written under `data/processed/instructions/`.

### 8. Run instruction training

The instruction mode is hard-coded in `TRAIN_CONFIGS['instruction']` inside `config.py` and currently starts from `weights/snapshots/pretrain/snapshot_07.pt`.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 --max_restarts=0 train.py instruction
```

Instruction checkpoints and snapshots are stored separately under `weights/checkpoints/instructions/` and `weights/snapshots/instructions/`.

### 9. Inspect the instruction-tuned model

Set `INFERENCE_CONFIG['snapshot_run']` to `instructions` inside `config.py`, then run:

```bash
python inference.py
```

## Technical summary

| Component | Value |
|---|---:|
| Parameters | 158.6M |
| Decoder layers | 30 |
| Hidden dimension | 576 |
| Query / KV heads | 9 / 3 |
| Feed-forward | SwiGLU, width 2,000 |
| Context length | 2,048 |
| Attention | Causal grouped-query attention |
| Position encoding | RoPE |
| Normalization | Pre-RMSNorm and QK-RMSNorm |

Training uses BF16, PyTorch DistributedDataParallel, AdamW, gradient accumulation, resumable checkpoints, rolling inference snapshots, and assistant-only loss masking for instruction data.

The pretraining mixture is 80% DCLM, 12% SwallowCode-v2, 6% Nemotron-CC-Math, and 2% Cosmopedia v2.

## Key files

```text
config.py                Model, data, training-mode, and inference settings
prepare_data.py          Resumable pretraining-data preparation
prepare_instructions.py  Instruction-data preparation and progress reporting
train.py                 Pretraining and instruction training
notifications.py         Discord notifications and GPU telemetry
model.py                 Transformer architecture
checkpoint.py            Snapshots and resumable checkpoints
inference.py             CPU model inspection
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
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 --max_restarts=0 train.py pretrain
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
