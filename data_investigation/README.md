# Data Investigation

This folder contains the pre-training dataset investigation and the production-quality corpus builder. Nothing here is wired into TitusAI training yet.

## Finalized pre-training corpus

The selected corpus is 460,000,000 GPT-2 tokens:

- 65% / 299,000,000 tokens: FineWeb-Edu
- 25% / 115,000,000 tokens: Cosmopedia v2 middle-school textbooks
- 10% / 46,000,000 tokens: TinyStories V2 GPT-4

Each document is limited to 511 content tokens and terminated with GPT-2 EOS token 50256, giving a maximum stored document length of 512 tokens. The final corpus is raw `uint16`, so 460,000,000 tokens occupy exactly 920,000,000 bytes before filesystem accounting.

### FineWeb-Edu

Production source: `HuggingFaceFW/fineweb_edu_100BT-shuffled`

Filter:

- `int_score >= 4`
- `language_score >= 0.95`
- `100 <= token_count <= 1024`

The 100B shuffled subset provides enough headroom for the strict filter while preserving randomized source coverage.

### Cosmopedia v2

Production source: `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2/` Parquet files.

Filter:

- `audience == middle_school_students`
- `format` starts with `textbook`

### TinyStories V2 GPT-4

Production source: `maveriq/tinystoriesv2_gpt4`.

All non-empty training stories are eligible. Any literal `<|endoftext|>` marker is removed before exactly one EOS token is appended by the builder.

## Production corpus builder

`build_pretrain_corpus.py` builds the finalized corpus without touching the live TitusAI data pipeline.

It:

- freezes exact Hugging Face dataset revisions and deterministic shard order in `pretrain_corpus/plan.json`
- downloads Parquet shards only when needed and keeps them under `data_investigation/data/`
- reuses already-downloaded shards on later runs
- reads Parquet in bounded batches instead of loading whole datasets into RAM
- applies finalized filters with vectorized PyArrow operations before tokenization
- uses roughly one tokenizer worker per physical CPU core through multiprocessing
- tokenizes in batches with the GPT-2 fast tokenizer
- truncates documents to 511 content tokens and appends EOS
- writes raw `uint16` shard parts atomically
- resumes from completed shard parts after interruption
- fills the source quotas exactly to 299M / 115M / 46M tokens
- concatenates the source parts into one `pretrain.bin`
- writes `pretrain.json` with revisions, filters, document counts, sizes, and SHA-256

GPUs are intentionally unused because Parquet filtering and GPT-2 BPE tokenization are CPU/storage workloads. The RTX 3090s remain available for actual model training.

Run from the repository root:

```bash
python data_investigation/build_pretrain_corpus.py
```

Generated build state is written under:

```text
data_investigation/pretrain_corpus/
```

The important final artifacts are:

```text
data_investigation/pretrain_corpus/pretrain.bin
data_investigation/pretrain_corpus/pretrain.json
```

The generated corpus, source parts, plans, downloaded Parquet data, and investigation outputs are ignored by Git.

## Corpus verification

After the build finishes:

```bash
python data_investigation/verify_pretrain_corpus.py
```

The verifier memory-maps the final binary and checks:

- byte size against the manifest
- source token quotas
- valid GPT-2 token ID range
- EOS count against the recorded document count
- SHA-256 against the build manifest

## Investigation scripts

`sample_datasets.py` downloads one Parquet shard from each candidate dataset and writes readable samples under `data_investigation/output/`.

`quality_sweep.py` compares DCLM-Edu and FineWeb-Edu educational score bands.

`final_sweep.py` compares the strongest filtered slices and was used to choose the finalized corpus above.

The earlier investigation also contains samples from DCLM-Edu, Cosmopedia v1, ClimbMix, TinyStories, FineWeb-Edu, and Cosmopedia v2 for auditability.
