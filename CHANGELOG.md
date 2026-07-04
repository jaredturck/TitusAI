# Changelog

## 1.0.1

- Prevented `check_setup.py` from filling the 10,000-record shuffle buffer while checking one sample.
- Limited Parquet-backed sources to the `text` column during streaming.
- Switched SwallowCode-v2 from heterogeneous raw JSON shards to the Hub's normalized Parquet conversion.
- Added source-loader tests covering no-shuffle setup checks and converted-Parquet loading.
