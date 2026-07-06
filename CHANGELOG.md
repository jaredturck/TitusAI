# Changelog

## 1.0.12

- Replace assistant-style instruction tuning with a 50-million-token conversational mixture of SODA, Topical-Chat, and DailyDialog.
- Train packed newline-separated conversations with full next-token loss, ordinary causal attention, pretraining-sized batches, and no stored loss masks or document-isolation attention masks.
- Resume the current conversation run normally while allowing a fresh run to inherit the newest earlier checkpoint, and align inference with plain conversational turns.

## 1.0.11

- Report GPU 0 and GPU 1 separately in Discord startup and ten-minute progress embeds.
- Show temperature, fan speed, SM clock, power draw and limit, utilization, and thermal-throttling state for every NVIDIA GPU.
- Make tokens per second explicit and turn progress embeds orange when any GPU reports active thermal slowdown, while keeping all `nvidia-smi` work on the notifier thread.

## 1.0.8

- Replace plain Discord status messages with structured embeds for startup, progress, validation, completion, interruption, and failure events.
- Add progress bars, smoothed loss, ETA, throughput, learning rate, snapshot details, and rank-zero peak GPU memory to remote training updates.
- Keep notification formatting and HTTP delivery on the existing background worker so the training hot path performs no network or JSON work.

## 1.0.7

- Fix manifest generation to use the actual `SequencePacker` statistics and shared manifest writer.
- Validate and recover existing shard pairs before preprocessing, rebuilding missing manifests without retokenizing completed data.
- Resume incomplete sources by appending new shards while preserving the existing deduplication state.

## 1.0.6

- Replace the custom Discord webhook text file with project-root `.env` loading through `python-dotenv`.
- Ignore local environment files while keeping a credential-free `.env.example` template under version control.
- Remove the obsolete webhook-path configuration and update notifier tests and setup documentation.

## 1.0.5

- Restore Discord training monitoring through a dedicated non-blocking notifier used only by DDP rank zero.
- Send startup, ten-minute progress, validation, completion, interruption, and fatal-error messages without allowing webhook failures to interrupt training.
- Keep webhook credentials in a Git-ignored local file, add a standalone webhook test command, and document the setup flow.

## 1.0.4

- Avoid the PyArrow interpreter-finalization crash after successful dataset streaming.
- Flush console output and terminate cleanly after `check_setup.py` and `prepare_data.py` complete.
- Preserve normal exception handling and non-zero failures when either script raises an error.

## 1.0.3

- Replaced the gated Nemotron-CC-v2 general-web source with the public, globally shuffled DCLM 100BT dataset.
- Preserve the existing 10.4-billion-token general-text allocation and skip redundant buffered shuffling for pre-shuffled sources.
- Updated source documentation and added regression coverage for the configured general-web dataset.

## 1.0.2

- Replaced the incomplete SwallowCode Parquet-viewer path with direct streaming from the official Stage 5 JSONL files.
- Parse only SwallowCode's `text` field, so inconsistent metadata schemas cannot affect preprocessing.
- Added a deterministic bounded shuffle for schema-free JSONL streams.
- Explicitly close setup-check iterators after reading one record.
- Added tests for the real Stage 5 path, mixed metadata shapes, stream closure, and deterministic shuffling.

## 1.0.1

- Prevented `check_setup.py` from filling the 10,000-record shuffle buffer while checking one sample.
- Limited Parquet-backed sources to the `text` column during streaming.
- Switched SwallowCode-v2 from heterogeneous raw JSON shards to the Hub's normalized Parquet conversion.
- Added source-loader tests covering no-shuffle setup checks and converted-Parquet loading.
