# Pre-training Corpus Decisions

This file records the decisions reached during TitusAI's pre-training dataset investigation so they remain explicit when the experimental builder is eventually moved into the live training pipeline.

## Goal

TitusAI is a deliberately tiny language model. The pre-training corpus should optimize useful learning per token rather than imitate the scale or breadth strategy of billion-parameter models.

The target behavior is coherent English with broad but shallow general knowledge. Breadth is more important than spending thousands of tokens on individual topics.

## Final mixture

The first serious pre-training corpus is fixed at 460,000,000 GPT-2 tokens:

| Source | Share | Token quota | Role |
| --- | ---: | ---: | --- |
| FineWeb-Edu | 65% | 299,000,000 | Natural English and broad factual/educational coverage |
| Cosmopedia v2 | 25% | 115,000,000 | Structured conceptual explanations |
| TinyStories V2 GPT-4 | 10% | 46,000,000 | Simple grammar, causality, narrative coherence, and clean endings |

These percentages are token quotas after GPT-2 tokenization, not source document counts or source byte sizes.

## FineWeb-Edu

FineWeb-Edu became the backbone only after aggressive filtering. Unfiltered and score-3 samples contained too much low-value web material, opinion, marketing copy, SEO content, and page cruft.

The retained slice is:

- educational integer score 4 or higher
- language score 0.95 or higher
- upstream document token count between 100 and 1024

The investigation found this slice consistently contained compact educational material across science, history, civics, nature, technology, and ordinary factual topics while retaining naturally written English.

The production builder uses the globally shuffled 100B-token FineWeb-Edu subset so the strict filter has enough source material without downloading the full multi-terabyte corpus.

## Cosmopedia v2

Cosmopedia v2 was useful but should not dominate the corpus. Its structured educational generation frequently follows a helpful concept-definition-example-relationship pattern, but its synthetic voice is repetitive and some samples contained questionable or overconfident claims.

Only the middle-school textbook slice is retained:

- `audience == middle_school_students`
- `format` begins with `textbook`

At 25% it supplies pedagogical structure without making synthetic educational prose the model's primary language distribution.

Cosmopedia v1 is not used.

## TinyStories V2 GPT-4

TinyStories is extremely narrow in subject matter but unusually clean and relevant to models at TitusAI's scale. It supplies short grammatical sequences, dialogue, entities, cause and effect, simple common sense, and explicit endings.

Its role is coherence rather than factual knowledge, so it is limited to 10% of total tokens.

## Rejected sources

### DCLM-Edu

DCLM-Edu was removed after manual inspection. Even aggressive educational/language filtering retained inconsistent material such as commercial pages, low-quality essays, and assorted web noise. FineWeb-Edu's strict slice provided the same desired role more consistently.

### ClimbMix

ClimbMix samples contained useful material but also forum fragments, malformed page text, appended question-answer artifacts, and mixed synthetic material. It did not provide enough value over the selected sources for a 23M-scale model.

### Wikipedia / FineWiki

Wikipedia was not selected as a foundational corpus. Its factual quality is useful, but long entity-centered articles are a poor match for the project's breadth-per-token objective. The selected educational web and synthetic textbook slices provide more varied compact contexts.

### General web, academic-paper, code, and specialist math corpora

These are intentionally excluded from the first corpus. They either spend too many tokens on noise/specialization or target capabilities outside the current goal. They can be reconsidered only after measuring the first model trained on this corpus.

## Document policy

Every source document is independently GPT-2-tokenized.

- maximum content tokens: 511
- exactly one GPT-2 EOS token is appended
- maximum stored document length: 512 tokens
- literal `<|endoftext|>` strings in source text are removed before tokenization

The cap deliberately prevents a single long article from consuming the token budget of many shorter concepts.

## Storage policy

The final corpus is a headerless stream of GPT-2 token IDs stored as `uint16`.

GPT-2 has 50,257 vocabulary entries, so `uint8` is insufficient while `uint16` is sufficient without the overhead of 32-bit integer or floating-point storage.

460,000,000 tokens therefore occupy exactly 920,000,000 bytes.

A JSON manifest accompanies the binary and records source revisions, filters, document counts, source quotas, checksum, tokenizer, and storage information.

## Build policy

Corpus generation is a CPU, storage, and network workload. GPUs are intentionally not used.

The builder should:

- freeze exact upstream dataset revisions before building
- persist downloaded Parquet shards locally
- never redownload an existing shard
- apply cheap vectorized metadata filters before tokenization
- tokenize filtered documents in parallel across CPU cores
- process bounded Parquet batches rather than materializing full datasets
- checkpoint completed source shards atomically
- resume completed work after interruption
- produce exact source token quotas
- generate and verify a SHA-256 for the final binary

The experimental implementation lives in `data_investigation/build_pretrain_corpus.py`. It must remain isolated from the live training pipeline until the training architecture and post-training data decisions are complete.
