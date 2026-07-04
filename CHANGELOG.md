# Changelog

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
