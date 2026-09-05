# Data Investigation

This folder is for manually inspecting candidate pre-training datasets before changing TitusAI's training data.

`sample_datasets.py` streams a small randomized sample from each candidate rather than downloading full corpora. It currently examines:

- `HuggingFaceTB/dclm-edu`, filtered to educational score 3+
- `HuggingFaceFW/fineweb-edu`, using the 10B-token sample and filtering to educational score 3+
- `HuggingFaceTB/cosmopedia-100k`
- `maveriq/tinystoriesv2_gpt4`, a convenient parquet mirror of TinyStories V2 for inspection
- `karpathy/climbmix-400b-shuffle`, included as a modern per-token-efficiency wildcard

The script collects 20 examples from each source using Hugging Face streaming and a small shuffle buffer. Each text preview is capped at 5,000 characters so the inspection output stays small.

Run from the repository root:

```bash
python data_investigation/sample_datasets.py
```

Outputs are written to `data_investigation/output/`:

- one Markdown file per dataset for manual reading
- `summary.md` with basic length statistics
- `samples.jsonl` with the combined sample and useful source metadata

The generated output is investigation data and should not be committed.
