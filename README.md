# TitusAI

TitusAI is a readable, from-scratch PyTorch language-model project intended for studying how a modern compact LLM works end to end.

The neural network is implemented directly in `model.py`. Hugging Face is used for the tokenizer and source datasets; it is not used to provide the model or training loop.

## Architecture

The default model is a dense causal decoder with approximately 158.6 million parameters after the six Titus control tokens are added to the SmolLM2 tokenizer.

| Component | Value |
|---|---:|
| Decoder layers | 30 |
| Hidden dimension | 576 |
| Query heads | 9 |
| Key/value heads | 3 |
| Head dimension | 64 |
| Feed-forward layer | SwiGLU, width 2,000 |
| Normalization | Pre-RMSNorm and QK-RMSNorm |
| Position encoding | RoPE |
| Context length | 2,048 |
| Attention | Full causal grouped-query attention |
| Output | Tied token embedding and full cross-entropy |
| Dropout | 0 |

The implementation includes an autoregressive KV cache, BF16 DDP training, masked instruction loss, rolling inference snapshots, and CPU-only interactive inference.

## Project layout

```text
TitusAI/
├── config.py                  Model, training, data, and inference settings
├── tokenizer.py               SmolLM2 tokenizer configuration and chat formatting
├── model.py                   Complete decoder architecture
├── data_utils.py              Deduplication, packing, and shard writing
├── prepare_data.py            Base-pretraining data pipeline
├── prepare_instructions.py    Smol-SmolTalk assistant-only SFT pipeline
├── dataset.py                 Memory-mapped runtime dataset and shard sampler
├── train.py                   BF16 DistributedDataParallel training
├── checkpoint.py              Rolling snapshots and resumable checkpoints
├── generate.py                Sampling, KV-cache generation, and response parsing
├── inference.py               CPU-only interactive inspection
├── check_setup.py             Tokenizer, model, and source-access validation
└── tests/                     Focused correctness tests
```

See `DESIGN.md` for tensor shapes and implementation details, and `REFERENCES.md` for the research sources behind the design.

## Installation

A CUDA-enabled PyTorch build is required for training. Install the PyTorch build appropriate for the CUDA stack on the Arch Linux machine, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The code requires a PyTorch version with `scaled_dot_product_attention(..., enable_gqa=True)` support.

## Hugging Face access

Authenticate before preparing data:

```bash
hf auth login
```

The two NVIDIA sources are gated. Accept their data-access terms on Hugging Face before running the setup check:

- `nvidia/Nemotron-CC-v2`
- `nvidia/Nemotron-CC-Math-v1`

Dataset licenses and redistribution conditions remain the responsibility of the person running the training job. The preparation manifest records source names and tokenizer metadata, but raw source data is not redistributed by this project.

## Validate the installation

Run:

```bash
python check_setup.py
pytest -q
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29671 --max_restarts=0 ddp_smoke.py
```

`check_setup.py` downloads and pins the SmolLM2 tokenizer locally, constructs the full model on the meta device to verify its parameter count, and reads one record from every configured dataset source. It fails immediately if credentials, source names, configurations, or text fields are incorrect.

The setup check does not shuffle source streams. SwallowCode is read from Hugging Face's normalized Parquet conversion and only the `text` column is loaded; this avoids schema conflicts in unrelated raw JSON metadata fields. The full preparation run applies its shuffle buffer after the stream has loaded successfully.

## Base-pretraining mixture

The default 13-billion-token target is defined in `config.py`:

| Source | Tokens | Share |
|---|---:|---:|
| Nemotron-CC-v2 Medium-High-Quality | 10.40B | 80% |
| SwallowCode-v2 stage 5 | 1.56B | 12% |
| Nemotron-CC-Math 4+ | 0.78B | 6% |
| Cosmopedia v2 | 0.26B | 2% |

Specialist sources are processed before general web text so cross-source exact deduplication keeps the specialist copy.

## Prepare the base dataset

```bash
python prepare_data.py
```

The pipeline:

1. Streams whole records from Hugging Face.
2. Preserves code, mathematical notation, line breaks, and punctuation.
3. Applies cross-source exact document and repeated-paragraph deduplication.
4. Makes a deterministic document-level train/validation split.
5. Tokenizes with the SmolLM2 tokenizer and appends `<|endoftext|>` document boundaries.
6. Packs documents into fixed 2,049-token stored records, yielding 2,048 inputs and shifted labels.
7. Writes immutable `uint16` memory-mapped token and segment shards.
8. Writes a reproducibility manifest.

Base-pretraining shards omit the redundant all-ones loss-mask file. At 13 billion tokens, token and segment files require roughly 52 GB before filesystem overhead and the deduplication database.

The default validation fraction is 0.1%. For a pilot corpus, reduce `PREPARE_CONFIG['max_total_tokens']` to approximately 100 million or more. The per-source targets are scaled automatically.

Prepared data is written to:

```text
data/processed/train/
data/processed/validation/
```

## Train on both RTX 3090s

```bash
torchrun --standalone --nproc_per_node=2 train.py
```

Each GPU receives one DDP process. No GPU memory is reserved for inference.

The default effective batch contains:

```text
4 sequences per GPU
× 8 gradient-accumulation steps
× 2 GPUs
× 2,048 positions
= 131,072 sequence positions per optimizer update
```

The actual token counter excludes document-boundary and masked targets.

Training uses:

- BF16 autocast
- AdamW with β1 0.9 and β2 0.95
- 2% linear warmup and cosine decay
- Gradient clipping at 1.0
- TF32 matrix multiplication where applicable
- Shard-aware distributed shuffling
- Exact resume positions inside a shuffled epoch

If the initial microbatch does not fit, reduce `micro_batch_size` and increase `gradient_accumulation_steps` by the same factor. Enable gradient checkpointing only when needed; it saves memory by trading away throughput.

### Packed-document attention

Targets that cross document boundaries are always masked. `isolate_packed_documents` is disabled by default so PyTorch can use its optimized causal SDPA kernel. Setting it to `True` additionally blocks attention across packed documents, but the explicit block mask is slower and consumes more memory.

## Rolling snapshots and checkpoints

Inference snapshots are saved approximately every 600 seconds after the current optimizer update. They contain BF16 model weights and lightweight metadata only.

```text
weights/snapshots/pretrain/snapshot_00.pt
...
weights/snapshots/pretrain/snapshot_09.pt
```

There are always at most ten snapshots. The oldest slot is atomically replaced, so inference never observes a partially written file. Each snapshot is roughly 317 MB for the default model.

Full resumable checkpoints are separate:

```text
weights/checkpoints/pretrain/checkpoint_XXXXXXXXX.pt
```

They include model, optimizer, scheduler, RNG states, global token count, epoch, and exact committed sample position. A keyboard interruption writes both a resumable checkpoint and an inference snapshot.

## Inspect the current model on CPU

In another terminal:

```bash
python inference.py
```

The script uses 24 CPU threads and does not touch either GPU. Commands:

```text
/reload      Load the newest completed snapshot
/info        Show step, token count, validation loss, and save time
/clear       Clear conversation history
/direct      Select direct-answer mode
/reason      Select reasoning mode
/thinking    Toggle display of extracted thinking text
/help        Show commands
/exit        Exit
```

The tokenizer keeps raw-document `<|endoftext|>` boundaries separate from the chat `<|im_end|>` token. The model must learn the control-token formats during post-training before reasoning mode becomes meaningful. The generation code enforces a configurable thinking-token budget to prevent an endless reasoning loop.

## Instruction tuning

Prepare Smol-SmolTalk separately:

```bash
python prepare_instructions.py
```

This writes the same shard format under:

```text
data/processed/instructions/train/
data/processed/instructions/validation/
```

Only assistant tokens contribute to the loss. User, system, and formatting prefixes are masked.

To start an instruction run, edit the following settings in `config.py`:

```python
TRAIN_CONFIG['run_name'] = 'instructions'
TRAIN_CONFIG['train_data_path'] = INSTRUCTION_CONFIG['output_path'] / 'train'
TRAIN_CONFIG['validation_data_path'] = INSTRUCTION_CONFIG['output_path'] / 'validation'
TRAIN_CONFIG['initial_weights'] = SNAPSHOT_PATH / 'pretrain' / 'snapshot_XX.pt'
TRAIN_CONFIG['max_train_tokens'] = YOUR_ASSISTANT_TOKEN_BUDGET
TRAIN_CONFIG['resume_training'] = True
```

Then launch the same command:

```bash
torchrun --standalone --nproc_per_node=2 train.py
```

Set `INFERENCE_CONFIG['snapshot_run']` to `instructions` to inspect the instruction-tuned snapshots.

Smol-SmolTalk examples are treated as direct-answer supervision. The code supports `<|think|>` and `<|final|>` blocks, but no unverified chain-of-thought corpus is silently mixed into SFT. A future reasoning dataset should contain concise, validated traces and can use the same assistant-token loss mask.

## Tests

The included tests cover:

- Forward and backward passes
- Causal attention
- Grouped-query KV caching
- Cached versus full-sequence logits
- RoPE-compatible generation positions
- Packed-document isolation
- Assistant-only loss masking
- Boundary-target masking
- Shard rotation and mmap loading
- Distributed sampler partitioning and resume offsets
- Rolling ten-file snapshot replacement
- Atomic snapshot round trips
- Sampling filters and response extraction

These tests validate implementation behaviour, not final language quality. Model quality still depends on completing pretraining and post-training with suitable data and hyperparameters.

## Practical limitations

- The full 13B-token run was not executed as part of packaging.
- Dataset repositories can change; run `check_setup.py` before committing storage and compute.
- NVIDIA dataset access terms must be reviewed directly.
- A 159M model can become useful for constrained tasks, but it will not match modern multi-billion-parameter assistants across broad knowledge and reasoning.
