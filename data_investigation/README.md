# Data Investigation

This folder is for manually inspecting candidate pre-training datasets before changing TitusAI's training data.

`sample_datasets.py` downloads one Parquet shard from each candidate dataset into `data_investigation/data/`, then samples entirely from those local files. Once a shard exists locally, later runs reuse it without contacting Hugging Face for that dataset.

It currently examines:

- `HuggingFaceTB/dclm-edu`, filtered to educational score 3+
- `HuggingFaceFW/fineweb-edu`, using one shard from the 10B-token sample and filtering to educational score 3+
- `HuggingFaceTB/cosmopedia-100k`
- `maveriq/tinystoriesv2_gpt4`, a convenient parquet mirror of TinyStories V2 for inspection
- `karpathy/climbmix-400b-shuffle`, included as a modern per-token-efficiency wildcard

The script collects 20 randomized examples from each local shard. Each text preview is capped at 5,000 characters so the inspection output stays small.

Run from the repository root:

```bash
python data_investigation/sample_datasets.py
```

Downloaded shards are kept under `data_investigation/data/` and are ignored by Git. Delete a dataset's local folder only if you explicitly want that shard downloaded again.

Outputs are written to `data_investigation/output/`:

- one Markdown file per dataset for manual reading
- `summary.md` with basic length statistics
- `samples.jsonl` with the combined sample and useful source metadata

The generated data and output are investigation artifacts and should not be committed.
